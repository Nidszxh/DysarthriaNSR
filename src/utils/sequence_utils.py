"""
src/utils/sequence_utils.py — Shared sequence-alignment utilities.

Extracted from train.py and evaluate.py (Q1 deduplication fix) to provide
a single canonical implementation that both modules import.

# REFACTOR LOG
# [PERF] Vectorized -100 tail propagation: replaced Python for-loop over batch
#        items with a single broadcasted tensor mask operation — eliminates O(B)
#        sequential Python iterations, merges into one CUDA kernel launch.
"""

import torch


def align_labels_to_logits(
    labels: torch.Tensor,
    time_steps_logits: int,
    center_weight: float = 0.7,
) -> torch.Tensor:
    """
    Align a label tensor's time dimension to match the logits time dimension.

    Uses proportional nearest-neighbor interpolation rather than simple
    pad/truncate so supervision is distributed across the full logit timeline.
    This is still an approximation (not forced alignment), but it avoids
    concentrating CE supervision in the earliest frames only.

    Args:
        labels:             Label tensor [batch, seq_len]
        time_steps_logits:  Target time dimension (from logits.size(1))
        center_weight:      Boundary de-emphasis factor in [0, 1]. Higher values
                            mask more segment edges with -100 and keep only the
                            central fraction for frame-CE supervision.

    Returns:
        Aligned labels [batch, time_steps_logits]
    """
    batch_size, time_steps_labels = labels.shape

    if time_steps_labels == time_steps_logits:
        return labels

    # Map each logit frame to a proportional source-label index.
    indices = (
        torch.arange(time_steps_logits, device=labels.device, dtype=torch.float32)
        * (time_steps_labels / float(time_steps_logits))
    ).long().clamp(0, time_steps_labels - 1)
    aligned = labels[:, indices]

    if center_weight > 0.0 and time_steps_labels > 1 and time_steps_logits > 2:
        keep_frac = max(0.0, min(1.0, 1.0 - float(center_weight)))
        if keep_frac < 1.0:
            seg_edges = torch.linspace(
                0,
                time_steps_logits,
                steps=time_steps_labels + 1,
                device=labels.device,
            ).floor().long()
            for seg_idx in range(time_steps_labels):
                start = int(seg_edges[seg_idx].item())
                end = int(seg_edges[seg_idx + 1].item())
                seg_len = end - start
                if seg_len <= 0:
                    continue
                keep_len = max(1, int(round(seg_len * keep_frac)))
                center_start = start + max(0, (seg_len - keep_len) // 2)
                center_end = min(end, center_start + keep_len)
                if center_start > start:
                    aligned[:, start:center_start] = -100
                if center_end < end:
                    aligned[:, center_end:end] = -100

    # [PERF] Vectorized -100 tail propagation.
    # Previous: Python for-loop over batch items (O(B) sequential tensor ops).
    # Now: single broadcasted comparison — one CUDA kernel, no host-side loop.
    pad_mask = (labels == -100)
    if pad_mask.any():
        # Fraction of -100 tokens per sample in the source label sequence [B]
        pad_fraction = pad_mask.float().mean(dim=1)
        # Number of trailing positions to fill with -100 in the aligned tensor [B]
        n_pad = (pad_fraction * time_steps_logits).long()
        # Build a boolean mask: True where position >= (T - n_pad[b]) [B, T]
        positions = torch.arange(
            time_steps_logits, device=labels.device
        ).unsqueeze(0)  # [1, T]
        cutoffs = (time_steps_logits - n_pad).unsqueeze(1)  # [B, 1]
        tail_mask = positions >= cutoffs  # [B, T] – broadcasted, no Python loop
        aligned = aligned.masked_fill(tail_mask, -100)

    return aligned
