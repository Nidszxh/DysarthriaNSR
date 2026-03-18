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


def align_labels_to_logits(labels: torch.Tensor, time_steps_logits: int) -> torch.Tensor:
    """
    Align a label tensor's time dimension to match the logits time dimension.

    Uses proportional nearest-neighbor interpolation rather than simple
    pad/truncate so supervision is distributed across the full logit timeline.
    This is still an approximation (not forced alignment), but it avoids
    concentrating CE supervision in the earliest frames only.

    Args:
        labels:             Label tensor [batch, seq_len]
        time_steps_logits:  Target time dimension (from logits.size(1))

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
