"""Onsets and Frames (Hawthorne et al. 2018) run as-is on any audio → note table (zero-shot: no fine-tuning).

PyTorch reimplementation by Jong Wook Kim (github.com/jongwook/onsets-and-frames, MIT licence,
commit ``783ca08``), weights trained on MAESTRO piano for 500k steps and shared by the author in
the repo's issues #17 / #23. The model below is that repo's architecture, copied so the state
dict keys match; the mel front end is rebuilt on ``torch.stft`` + ``torchaudio`` (the repo's
uses ``librosa``, whose current API it no longer runs on).

The checkpoint is a whole pickled model from an old PyTorch (≈ 1.0), saved with each class's
source code. It is never unpickled as code: ``read_pickled_weights`` reads it with a restricted
unpickler that imports nothing and rebuilds the tensors only.
"""
from __future__ import annotations

import collections
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn

# Storage types a legacy checkpoint names → their element type
_STORAGE_DTYPES = {"FloatStorage": torch.float32, "LongStorage": torch.int64}


class _Placeholder:
    """Stands in for any class the checkpoint names: keeps the object's state as attributes, runs no code."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


def _rebuild_tensor(storage: torch.Tensor, storage_offset: int, size, stride, *_) -> torch.Tensor:
    """Our own stand-in for torch._utils._rebuild_tensor_v2: a strided view of a flat storage."""
    return storage.as_strided(size, stride, storage_offset)


def _rebuild_parameter(data: torch.Tensor, *_) -> torch.Tensor:
    """Our own stand-in for torch._utils._rebuild_parameter: the plain tensor."""
    return data


class _RestrictedUnpickler(pickle.Unpickler):
    """Reads a legacy (pre-1.6) torch pickle while importing nothing.

    Classes become ``_Placeholder``s, the tensor rebuild functions become our own, and storages
    become empty tensors, filled from the file after the object is read. The saved module source
    code old PyTorch stored alongside each class is dropped unread.
    """

    def __init__(self, file):
        super().__init__(file)
        self.storages: dict[str, torch.Tensor] = {}

    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if (module, name) == ("torch._utils", "_rebuild_parameter"):
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _STORAGE_DTYPES[name]
        return type(name, (_Placeholder,), {})

    def persistent_load(self, persistent_id):
        kind = persistent_id[0]
        if kind == "module":  # ('module', class, source file, source code)
            return persistent_id[1]
        if kind == "storage":  # ('storage', dtype, key, location, element count, view)
            _, dtype, key, _, element_count, view = persistent_id
            if key not in self.storages:
                self.storages[key] = torch.empty(element_count, dtype=dtype)
            if view is not None:
                _, view_offset, view_size = view
                return self.storages[key][view_offset:view_offset + view_size]
            return self.storages[key]
        raise pickle.UnpicklingError(f"unexpected persistent id {kind!r}")


def read_pickled_weights(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """The parameters and buffers of a whole pickled ``nn.Module`` (legacy format), as a state dict, without running its code."""
    with open(checkpoint_path, "rb") as file:
        # Header pickles: magic number, protocol version, system info
        for _ in range(3):
            _RestrictedUnpickler(file).load()
        unpickler = _RestrictedUnpickler(file)
        model = unpickler.load()
        # Then the storage keys, then each storage's raw data: element count (int64) + values, little-endian
        storage_keys = _RestrictedUnpickler(file).load()
        for key in storage_keys:
            storage = unpickler.storages[key]
            element_count = int(np.frombuffer(file.read(8), dtype="<i8")[0])
            if element_count != storage.numel():
                raise ValueError(f"storage {key}: {element_count} values in the file, {storage.numel()} expected")
            values = np.frombuffer(file.read(element_count * storage.element_size()),
                                   dtype=torch.empty(0, dtype=storage.dtype).numpy().dtype.newbyteorder("<"))
            storage.copy_(torch.from_numpy(values.copy()))

    def walk(module, prefix: str) -> dict[str, torch.Tensor]:
        state = {f"{prefix}{name}": tensor for group in ("_parameters", "_buffers")
                 for name, tensor in module.__dict__.get(group, {}).items() if tensor is not None}
        for name, child in module.__dict__.get("_modules", {}).items():
            state.update(walk(child, f"{prefix}{name}."))
        return state

    return walk(model, "")


class ConvStack(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(nn.Linear((output_features // 8) * (input_features // 4), output_features),
                                nn.Dropout(0.5))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.cnn(mel.view(mel.size(0), 1, mel.size(1), mel.size(2)))
        return self.fc(x.transpose(1, 2).flatten(-2))


class BiLSTM(nn.Module):
    """Bidirectional LSTM; at inference each direction runs in chunks, carrying its state across them."""

    def __init__(self, input_features: int, recurrent_features: int, inference_chunk_frames: int):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)
        self.inference_chunk_frames = inference_chunk_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.rnn(x)[0]
        batch_size, sequence_length, _ = x.shape
        hidden_size = self.rnn.hidden_size
        h = torch.zeros(2, batch_size, hidden_size, device=x.device)
        c = torch.zeros(2, batch_size, hidden_size, device=x.device)
        output = torch.zeros(batch_size, sequence_length, 2 * hidden_size, device=x.device)
        starts = range(0, sequence_length, self.inference_chunk_frames)
        for start in starts:  # forward direction
            end = start + self.inference_chunk_frames
            output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))
        h.zero_()
        c.zero_()
        for start in reversed(starts):  # reverse direction
            end = start + self.inference_chunk_frames
            result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
            output[:, start:end, hidden_size:] = result[:, :, hidden_size:]
        return output


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features: int, output_features: int, model_complexity: int, inference_chunk_frames: int):
        super().__init__()
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2, inference_chunk_frames)
        self.onset_stack = nn.Sequential(ConvStack(input_features, model_size), sequence_model(model_size, model_size),
                                         nn.Linear(model_size, output_features), nn.Sigmoid())
        self.offset_stack = nn.Sequential(ConvStack(input_features, model_size), sequence_model(model_size, model_size),
                                          nn.Linear(model_size, output_features), nn.Sigmoid())
        self.frame_stack = nn.Sequential(ConvStack(input_features, model_size),
                                         nn.Linear(model_size, output_features), nn.Sigmoid())
        self.combined_stack = nn.Sequential(sequence_model(output_features * 3, model_size),
                                            nn.Linear(model_size, output_features), nn.Sigmoid())
        self.velocity_stack = nn.Sequential(ConvStack(input_features, model_size), nn.Linear(model_size, output_features))

    def forward(self, mel: torch.Tensor):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class LogMel(nn.Module):
    """The repo's front end: magnitude STFT (Hann, reflect-padded, centred) → HTK mel, Slaney-normalised → log."""

    def __init__(self, config):
        super().__init__()
        self.window_length, self.hop_length = config.window_length, config.hop_length
        self.register_buffer("window", torch.hann_window(config.window_length, periodic=True))
        filters = torchaudio.functional.melscale_fbanks(config.window_length // 2 + 1, config.mel_min_hz, config.mel_max_hz,
                                                        config.mel_bands, config.sample_rate, norm="slaney", mel_scale="htk")
        self.register_buffer("mel_filters", filters.T.contiguous())  # (mels, frequency bins)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(batch, samples) in [-1, 1] → (batch, frames, mels)."""
        spectrum = torch.stft(audio, self.window_length, self.hop_length, window=self.window, center=True,
                              pad_mode="reflect", return_complex=True).abs()
        return torch.log(torch.clamp(self.mel_filters @ spectrum, min=1e-5)).transpose(-1, -2)


class OnsetsFramesTranscriber:
    """Mel front end + model + note decoding, with the recipe's settings."""

    def __init__(self, config, device: str):
        self.config, self.device = config, torch.device(device)
        self.log_mel = LogMel(config).to(self.device)
        self.model = load_onsets_frames(config).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, audio: np.ndarray, part: str) -> pd.DataFrame:
        """Note table of mono audio at ``config.sample_rate`` (columns part, pitch, velocity, start, duration)."""
        samples = torch.from_numpy(np.clip(audio, -1.0, 1.0).astype(np.float32)).to(self.device)
        mel = self.log_mel(samples[None, :-1])  # the repo drops the last sample before the STFT
        onset, _, _, frame, velocity = self.model(mel)
        return decode_notes(onset[0].cpu().numpy(), frame[0].cpu().numpy(), velocity[0].cpu().numpy(), part, self.config)


def load_onsets_frames(config) -> OnsetsAndFrames:
    """The pretrained model, after checking the checkpoint's md5 and loading every weight with strict key matching.

    Args:
        config: the `configs/transcribe/onsets_frames.yaml` settings.
    """
    checkpoint_md5 = hashlib.md5(Path(config.checkpoint_path).read_bytes()).hexdigest()
    if checkpoint_md5 != config.checkpoint_md5:
        raise RuntimeError(f"{config.checkpoint_path} has md5 {checkpoint_md5}, the recipe pins {config.checkpoint_md5}")
    model = OnsetsAndFrames(config.mel_bands, config.max_midi - config.min_midi + 1, config.model_complexity,
                            config.lstm_inference_chunk_frames)
    model.load_state_dict(read_pickled_weights(config.checkpoint_path), strict=True)
    return model


def decode_notes(onset: np.ndarray, frame: np.ndarray, velocity: np.ndarray, part: str, config) -> pd.DataFrame:
    """The repo's ``extract_notes``: a note starts where the onset head rises above its threshold and
    lasts while the onset or frame head stays above theirs. Velocity = mean over its onset frames.

    Args:
        onset, frame, velocity: (frames, keys) model outputs.
        part: written into the ``part`` column.
        config: the `configs/transcribe/onsets_frames.yaml` settings.
    """
    onsets = onset > config.onset_threshold
    active = onsets | (frame > config.frame_threshold)
    rising = onsets & ~np.vstack([np.zeros((1, onsets.shape[1]), bool), onsets[:-1]])
    seconds_per_frame = config.hop_length / config.sample_rate
    rows = []
    for start, key in zip(*np.nonzero(rising)):
        end = start
        while end < len(active) and active[end, key]:
            end += 1
        onset_velocities = velocity[start:end, key][onsets[start:end, key]]
        rows.append({"part": part, "pitch": int(config.min_midi + key),
                     "velocity": int(np.clip(onset_velocities.mean() * 127, 0, 127)) if len(onset_velocities) else 0,
                     "start": start * seconds_per_frame, "duration": (end - start) * seconds_per_frame})
    table = pd.DataFrame(rows, columns=["part", "pitch", "velocity", "start", "duration"])
    return table.sort_values("start", kind="stable").reset_index(drop=True)
