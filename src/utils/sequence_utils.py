"""
src/utils/sequence_utils.py — Shared sequence-alignment utilities.

Extracted from train.py and evaluate.py (Q1 deduplication fix) to provide
a single canonical implementation that both modules import.
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

    # Preserve ignore-index tail semantics when source labels contain -100.
    pad_mask = (labels == -100)
    if pad_mask.any():
        pad_fraction = pad_mask.float().mean(dim=1)
        n_pad = (pad_fraction * time_steps_logits).long()
        for b in range(batch_size):
            n = int(n_pad[b].item())
            if n > 0:
                aligned[b, time_steps_logits - n:] = -100

    return aligned
