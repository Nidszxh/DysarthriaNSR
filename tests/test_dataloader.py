"""
Unit tests for the data collator.

Tests verify:
- Label padding uses -100 sentinel (never 0 or 1)
- Output tensor shapes are consistent across varying sequence lengths
- Attention mask correctly marks valid vs. padding regions
- Articulatory labels are padded with -100
"""
import pytest
import torch

from src.data.dataloader import NeuroSymbolicCollator


def _make_sample(audio_len: int, label_len: int, speaker: str = "M01", is_dysarthric: int = 1):
    """Build a minimal collator-compatible sample dict (no file I/O required)."""
    return {
        "input_values": torch.zeros(audio_len),
        "labels": torch.zeros(label_len, dtype=torch.long) + 5,  # phoneme ID=5
        "articulatory_labels": {
            "manner": torch.zeros(label_len, dtype=torch.long),
            "place": torch.zeros(label_len, dtype=torch.long),
            "voice": torch.zeros(label_len, dtype=torch.long),
        },
        "metadata": {
            "is_dysarthric": torch.tensor(is_dysarthric, dtype=torch.long),
            "speaker": speaker,
            "transcript": "AH B C",
        },
    }


class TestNeuroSymbolicCollator:
    def setup_method(self):
        # processor is only stored, never called by __call__
        self.collator = NeuroSymbolicCollator(processor=None, pad_id=1, ctc_stride=320)

    def test_label_padding_sentinel(self):
        """Labels must be padded with -100, not 0 or 1."""
        samples = [_make_sample(3200, 5), _make_sample(6400, 10)]
        batch = self.collator(samples)
        labels = batch["labels"]
        # Short sample (len=5) should have padding in positions 5..9
        padded_positions = labels[0, 5:]
        assert (padded_positions == -100).all(), "Expected -100 padding, got non-sentinel values"

    def test_labels_not_padded_with_zero(self):
        """Padding must never be 0 (that's the CTC blank token ID)."""
        samples = [_make_sample(3200, 3), _make_sample(6400, 8)]
        batch = self.collator(samples)
        # All padding positions should be -100
        short_label = batch["labels"][0]
        pad_values = short_label[3:]  # positions beyond original length
        assert (pad_values != 0).all(), "Found 0-padded labels — must be -100"

    def test_batch_shapes_consistent(self):
        """All tensor outputs should have consistent batch dimension."""
        samples = [_make_sample(3200, 4), _make_sample(4800, 6), _make_sample(1600, 2)]
        batch = self.collator(samples)
        B = len(samples)
        assert batch["input_values"].shape[0] == B
        assert batch["labels"].shape[0] == B
        assert batch["attention_mask"].shape[0] == B

    def test_attention_mask_marks_valid_region(self):
        """Attention mask should be 1 for valid frames, 0 for padding."""
        audio_len_0 = 3200  # 10 frames at stride 320
        audio_len_1 = 6400  # 20 frames
        samples = [_make_sample(audio_len_0, 4), _make_sample(audio_len_1, 6)]
        batch = self.collator(samples)
        mask = batch["attention_mask"]
        # Shorter sample: valid region is audio_len_0 samples
        assert mask[0, :audio_len_0].sum() == audio_len_0
        assert mask[0, audio_len_0:].sum() == 0  # padding is zero

    def test_articulatory_labels_padded_with_neg_100(self):
        """Articulatory labels for shorter samples must be padded with -100."""
        samples = [_make_sample(3200, 3), _make_sample(6400, 7)]
        batch = self.collator(samples)
        art = batch["articulatory_labels"]
        for key in ("manner", "place", "voice"):
            short_art = art[key][0]
            padding = short_art[3:]  # beyond original length=3
            assert (padding == -100).all(), f"Articulatory '{key}' not padded with -100"

    def test_input_lengths_based_on_ctc_stride(self):
        """input_lengths = audio_samples // ctc_stride."""
        samples = [_make_sample(3200, 4), _make_sample(6400, 6)]
        batch = self.collator(samples)
        assert batch["input_lengths"][0].item() == 3200 // 320  # = 10
        assert batch["input_lengths"][1].item() == 6400 // 320  # = 20

    def test_label_lengths_correct(self):
        samples = [_make_sample(3200, 4), _make_sample(6400, 7)]
        batch = self.collator(samples)
        assert batch["label_lengths"][0].item() == 4
        assert batch["label_lengths"][1].item() == 7

    def test_speakers_list_preserved(self):
        samples = [_make_sample(3200, 4, speaker="M01"), _make_sample(3200, 4, speaker="F01")]
        batch = self.collator(samples)
        assert batch["speakers"] == ["M01", "F01"]

    def test_status_tensor(self):
        samples = [_make_sample(3200, 4, is_dysarthric=1), _make_sample(3200, 4, is_dysarthric=0)]
        batch = self.collator(samples)
        assert batch["status"][0].item() == 1
        assert batch["status"][1].item() == 0

    def test_single_sample_batch(self):
        """Single-sample batch should produce [1, ...] tensors."""
        samples = [_make_sample(3200, 5)]
        batch = self.collator(samples)
        assert batch["input_values"].shape[0] == 1
        assert batch["labels"].shape[0] == 1
