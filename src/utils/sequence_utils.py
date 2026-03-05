"""
src/utils/sequence_utils.py — Shared sequence-alignment utilities.

Extracted from train.py and evaluate.py (Q1 deduplication fix) to provide
a single canonical implementation that both modules import.
"""

import torch


def align_labels_to_logits(labels: torch.Tensor, time_steps_logits: int) -> torch.Tensor:
    """
    Align a label tensor's time dimension to match the logits time dimension.

    Pads with -100 (CTC/CE ignore index) if labels are shorter, or truncates
    if labels are longer than the logits sequence length.

    Args:
        labels:             Label tensor [batch, seq_len]
        time_steps_logits:  Target time dimension (from logits.size(1))

    Returns:
        Aligned labels [batch, time_steps_logits]
    """
    batch_size = labels.size(0)
    time_steps_labels = labels.size(1)

    if time_steps_labels < time_steps_logits:
        padding = torch.full(
            (batch_size, time_steps_logits - time_steps_labels),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return torch.cat([labels, padding], dim=1)
    return labels[:, :time_steps_logits]
