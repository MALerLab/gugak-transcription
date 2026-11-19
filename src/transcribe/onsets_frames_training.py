"""Fine-tuning Onsets and Frames on single stems: label rolls, batches, the masked loss, one training step.

Follows Jong Wook Kim's repo (``train.py``, ``onsets_and_frames/transcriber.py``) where it can: random
fixed-length segments, binary cross-entropy on the onset, offset and frame heads, gradient clipping.
Two changes for gugak:
- frames inside a time mask (the six dragon grace-note windows) count for no loss term;
- no velocity loss: the six dragon MIDI's velocity is a constant (MusicXML has no dynamics).
Optional augmentation on the magnitude spectrum (``augmentation.enabled``, off when missing): a random
tuning offset, whole-semitone shifts with the labels moved along, gain and spectral tilt.
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


def warp_frequency(spectrum: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """Each example's magnitude spectrum ``(batch, bins, frames)`` stretched along frequency by its factor.

    Bin ``j`` takes the (linearly interpolated) magnitude of bin ``j / factor``: every partial moves to
    ``factor`` × its frequency, so the pitch moves by ``1200 log2(factor)`` cents while timing stays put.
    Bins whose source lies past the top bin go silent.
    """
    num_bins, num_frames = spectrum.shape[1], spectrum.shape[2]
    source = torch.arange(num_bins, device=spectrum.device, dtype=spectrum.dtype)[None, :] / factors[:, None]
    low = source.floor().long()
    weight = (source - low)[..., None]
    pick = lambda index: torch.gather(spectrum, 1, index.clamp(0, num_bins - 1)[..., None].expand(-1, -1, num_frames))
    warped = pick(low) * (1 - weight) + pick(low + 1) * weight
    return warped * (source <= num_bins - 1)[..., None]


def shift_keys(roll: torch.Tensor, shifts: list[int]) -> torch.Tensor:
    """Label rolls ``(batch, frames, keys)`` moved up by each example's whole number of keys; keys pushed off the end drop."""
    shifted = torch.zeros_like(roll)
    for index, shift in enumerate(shifts):
        if shift > 0:
            shifted[index, :, shift:] = roll[index, :, :-shift]
        elif shift < 0:
            shifted[index, :, :shift] = roll[index, :, -shift:]
        else:
            shifted[index] = roll[index]
    return shifted


def augment(spectrum: torch.Tensor, batch: dict[str, torch.Tensor], sample_rate: int,
            settings) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Random per-example augmentation of a magnitude spectrum ``(batch, bins, frames)`` and its labels.

    - tuning: pitch moved by a uniform amount within ±``tuning_cents``; labels unchanged (the note stays on its key)
    - semitone shift: with ``semitone_shift_probability``, also a whole number of semitones within
      ±``semitone_shift_max`` (never 0), labels moved by the same number of keys
    - gain: uniform within ±``gain_db``
    - tilt: uniform within ±``tilt_db_per_octave``, pivoting at 1 kHz (a crude random EQ)
    """
    size, device = spectrum.shape[0], spectrum.device
    uniform = lambda limit: (torch.rand(size, device=device) * 2 - 1) * limit
    semitones = [0] * size
    if settings.semitone_shift_max > 0:
        # torch's global generator (seeded by the training command), so runs repeat
        choices = [s for s in range(-settings.semitone_shift_max, settings.semitone_shift_max + 1) if s != 0]
        picks = torch.randint(len(choices), (size,)).tolist()
        shifted = (torch.rand(size) < settings.semitone_shift_probability).tolist()
        semitones = [choices[pick] if on else 0 for pick, on in zip(picks, shifted)]
    cents = uniform(settings.tuning_cents) + 100 * torch.tensor(semitones, device=device, dtype=spectrum.dtype)
    spectrum = warp_frequency(spectrum, 2 ** (cents / 1200))
    num_bins = spectrum.shape[1]
    frequencies = torch.arange(num_bins, device=device, dtype=spectrum.dtype) * sample_rate / (2 * (num_bins - 1))
    octaves_from_pivot = torch.log2(frequencies.clamp(min=30.0) / 1000.0)
    decibels = uniform(settings.gain_db)[:, None] + uniform(settings.tilt_db_per_octave)[:, None] * octaves_from_pivot[None, :]
    spectrum = spectrum * (10 ** (decibels / 20))[..., None]
    if any(semitones):
        batch = {**batch, **{head: shift_keys(batch[head], semitones) for head in LOSS_HEADS}}
    return spectrum, batch


def training_step(model, log_mel, optimizer, batch: dict[str, torch.Tensor], training_config) -> dict[str, float]:
    """One optimiser step on a batch (augmented first if the recipe asks); returns the loss terms."""
    model.train()
    spectrum = log_mel.spectrum(batch["audio"][:, :-1])  # the repo drops the last sample before the STFT
    augmentation = training_config.get("augmentation")
    if augmentation is not None and augmentation.enabled:
        spectrum, batch = augment(spectrum, batch, log_mel.sample_rate, augmentation)
    mel = log_mel.mel_from_spectrum(spectrum)
    onset, offset, _, frame, _ = model(mel)
    predictions = {"onset": onset, "offset": offset, "frame": frame}
    loss, parts = masked_loss(predictions, batch, dict(training_config.loss_weights))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.clip_gradient_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), **parts}
