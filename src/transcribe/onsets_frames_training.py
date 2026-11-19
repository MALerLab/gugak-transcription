"""Fine-tuning Onsets and Frames on single stems: label rolls, batches, the masked loss, one training step.

Follows Jong Wook Kim's repo (``train.py``, ``onsets_and_frames/transcriber.py``) where it can: random
fixed-length segments, binary cross-entropy on the onset, offset and frame heads, gradient clipping.
Two changes for gugak:
- frames inside a time mask (the six dragon grace-note windows) count for no loss term;
- no velocity loss: the six dragon MIDI's velocity is a constant (MusicXML has no dynamics).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.notes.roll import note_table_to_label_rolls, windows_to_frame_mask

LOSS_HEADS = ("onset", "offset", "frame")


def frame_count(num_samples: int, hop_length: int) -> int:
    """Frames the mel front end makes from ``num_samples`` (it drops the last sample, then a centred STFT)."""
    return (num_samples - 1) // hop_length + 1


def stem_labels(notes: pd.DataFrame, windows: np.ndarray, num_samples: int, transcribe_config,
                training_config) -> dict[str, np.ndarray]:
    """One stem's label rolls (onset, offset, frame: ``(frames, keys)`` uint8) and its frame mask (``(frames,)`` bool).

    Args:
        notes: the stem's reference note table (one part).
        windows: ``(n, 2)`` seconds left out of the loss.
        num_samples: stem length at the model's sample rate.
        transcribe_config: `configs/transcribe/onsets_frames.yaml` settings.
        training_config: `configs/transcribe/onsets_frames_training.yaml` settings.
    """
    hop_seconds = transcribe_config.hop_length / transcribe_config.sample_rate
    num_frames = frame_count(num_samples, transcribe_config.hop_length)
    rolls = note_table_to_label_rolls(notes, num_frames, hop_seconds, transcribe_config.min_midi,
                                      transcribe_config.max_midi - transcribe_config.min_midi + 1,
                                      training_config.onset_label_frames, training_config.offset_label_frames)
    return {**rolls, "mask": windows_to_frame_mask(windows, num_frames, hop_seconds)}


def pad_to_segment(audio: np.ndarray, labels: dict[str, np.ndarray], segment_frames: int,
                   hop_length: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A stem no longer than a segment, padded with silence to ``segment_frames + 1`` frames; padding masked out of the loss."""
    num_frames = segment_frames + 1
    extra = num_frames - len(labels["mask"])
    padded = {name: np.pad(array, [(0, extra)] + [(0, 0)] * (array.ndim - 1)) for name, array in labels.items()}
    return np.pad(audio, (0, max(0, num_frames * hop_length - len(audio)))), padded


class SegmentSampler:
    """All training stems held on one device; draws random fixed-length segments, stems weighted by length.

    A stem no longer than a segment (a solo clip) is padded with silence and masked there, then always drawn
    whole; its weight is its real length, so segments come from each dataset in proportion to its audio time.
    """

    def __init__(self, audios: list[np.ndarray], labels: list[dict[str, np.ndarray]], segment_frames: int,
                 hop_length: int, device: torch.device, seed: int):
        self.audios, self.labels, frames = [], [], []
        for audio, labels_i in zip(audios, labels):
            frames.append(len(labels_i["mask"]))
            if len(labels_i["mask"]) <= segment_frames:
                audio, labels_i = pad_to_segment(audio, labels_i, segment_frames, hop_length)
            self.audios.append(torch.from_numpy(audio).to(device))
            self.labels.append({name: torch.from_numpy(array).to(device) for name, array in labels_i.items()})
        frames = np.array(frames, dtype=float)
        self.padded_count = int((frames <= segment_frames).sum())
        self.stem_weights = frames / frames.sum()
        self.segment_frames, self.hop_length = segment_frames, hop_length
        self.generator = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """A batch: ``audio`` (batch, samples), label rolls (batch, frames, keys) float, ``mask`` (batch, frames) float."""
        stems = self.generator.choice(len(self.audios), size=batch_size, p=self.stem_weights)
        batch = {"audio": [], "onset": [], "offset": [], "frame": [], "mask": []}
        for stem in stems:
            num_frames = len(self.labels[stem]["mask"])
            start = int(self.generator.integers(0, num_frames - self.segment_frames))
            # segment_frames * hop samples → segment_frames mel frames (the front end drops the last sample)
            sample_start = start * self.hop_length
            batch["audio"].append(self.audios[stem][sample_start:sample_start + self.segment_frames * self.hop_length])
            for name in ("onset", "offset", "frame", "mask"):
                batch[name].append(self.labels[stem][name][start:start + self.segment_frames])
        return {name: torch.stack(values).float() for name, values in batch.items()}


def masked_loss(predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                head_weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-head binary cross-entropy, averaged over unmasked (frame, key) cells only."""
    mask = batch["mask"][..., None]
    cells = (mask.sum() * predictions["onset"].shape[-1]).clamp(min=1.0)
    total, parts = 0.0, {}
    for head in LOSS_HEADS:
        loss = F.binary_cross_entropy(predictions[head], batch[head], reduction="none")
        head_loss = (loss * mask).sum() / cells
        total = total + head_weights[head] * head_loss
        parts[head] = float(head_loss.detach())
    return total, parts


def training_step(model, log_mel, optimizer, batch: dict[str, torch.Tensor], training_config) -> dict[str, float]:
    """One optimiser step on a batch; returns the loss terms."""
    model.train()
    mel = log_mel(batch["audio"][:, :-1])  # the repo drops the last sample before the STFT
    onset, offset, _, frame, _ = model(mel)
    predictions = {"onset": onset, "offset": offset, "frame": frame}
    loss, parts = masked_loss(predictions, batch, dict(training_config.loss_weights))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.clip_gradient_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), **parts}
