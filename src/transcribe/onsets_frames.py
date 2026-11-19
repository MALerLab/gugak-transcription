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
    def __init__(self, input_features: int, output_features: int, model_complexity: int, inference_chunk_frames: int,
                 num_instruments: int = 0):
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
        self.num_instruments = num_instruments
        if num_instruments:
            # Instrument conditioning, all zero at the start so the model begins as the pretrained one:
            # FiLM (scale, shift) on each note head's conv features, and a per-key bias on the onset and frame outputs
            zero_embedding = lambda size: torch.zeros(num_instruments, size)
            self.instrument_film = nn.ParameterDict({f"{head}_{kind}": nn.Parameter(zero_embedding(model_size))
                                                     for head in ("onset", "offset", "frame") for kind in ("scale", "shift")})
            self.instrument_key_bias = nn.ParameterDict({head: nn.Parameter(zero_embedding(output_features))
                                                         for head in ("onset", "frame")})

    def _conditioned_head(self, stack: nn.Sequential, features: torch.Tensor, instrument: torch.Tensor, film: str | None,
                          key_bias: str | None, conv_first: bool = True) -> torch.Tensor:
        """One head's ``nn.Sequential`` run layer by layer, with the instrument's FiLM after its conv stack and its key bias before the sigmoid."""
        layers = list(stack)
        x = layers[0](features)
        if film is not None and conv_first:
            x = x * (1 + self.instrument_film[f"{film}_scale"][instrument][:, None, :]) + self.instrument_film[f"{film}_shift"][instrument][:, None, :]
        for layer in layers[1:-1]:
            x = layer(x)
        if key_bias is not None:
            x = x + self.instrument_key_bias[key_bias][instrument][:, None, :]
        return layers[-1](x)

    def forward(self, mel: torch.Tensor, instrument: torch.Tensor | None = None):
        """Outputs for a batch of log mels; ``instrument`` (batch,) indexes the conditioning instruments (None: unconditioned model)."""
        if not self.num_instruments:
            onset_pred = self.onset_stack(mel)
            offset_pred = self.offset_stack(mel)
            activation_pred = self.frame_stack(mel)
            combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
            frame_pred = self.combined_stack(combined_pred)
            velocity_pred = self.velocity_stack(mel)
            return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
        if instrument is None:
            raise ValueError("this model is instrument-conditioned: pass the instrument index")
        onset_pred = self._conditioned_head(self.onset_stack, mel, instrument, "onset", "onset")
        offset_pred = self._conditioned_head(self.offset_stack, mel, instrument, "offset", None)
        activation_pred = self._conditioned_head(self.frame_stack, mel, instrument, "frame", None)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self._conditioned_head(self.combined_stack, combined_pred, instrument, None, "frame", conv_first=False)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class LogMel(nn.Module):
    """The repo's front end: magnitude STFT (Hann, reflect-padded, centred) → HTK mel, Slaney-normalised → log."""

    def __init__(self, config):
        super().__init__()
        self.window_length, self.hop_length, self.sample_rate = config.window_length, config.hop_length, config.sample_rate
        self.register_buffer("window", torch.hann_window(config.window_length, periodic=True))
        filters = torchaudio.functional.melscale_fbanks(config.window_length // 2 + 1, config.mel_min_hz, config.mel_max_hz,
                                                        config.mel_bands, config.sample_rate, norm="slaney", mel_scale="htk")
        self.register_buffer("mel_filters", filters.T.contiguous())  # (mels, frequency bins)

    def spectrum(self, audio: torch.Tensor) -> torch.Tensor:
        """(batch, samples) in [-1, 1] → magnitude STFT (batch, frequency bins, frames)."""
        return torch.stft(audio, self.window_length, self.hop_length, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True).abs()

    def mel_from_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Magnitude STFT (batch, frequency bins, frames) → log mel (batch, frames, mels)."""
        return torch.log(torch.clamp(self.mel_filters @ spectrum, min=1e-5)).transpose(-1, -2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(batch, samples) in [-1, 1] → (batch, frames, mels)."""
        return self.mel_from_spectrum(self.spectrum(audio))


class OnsetsFramesTranscriber:
    """Mel front end + model + note decoding, with the recipe's settings.

    Loads the MAESTRO weights, or fine-tuned ones when ``config.finetuned_weights_path`` is set.
    """

    def __init__(self, config, device: str):
        self.config, self.device = config, torch.device(device)
        self.log_mel = LogMel(config).to(self.device)
        self.model = load_onsets_frames(config).to(self.device).eval()

    @torch.no_grad()
    def predict(self, audio: np.ndarray, instrument: str | None = None) -> dict[str, np.ndarray]:
        """The model's onset, offset, frame and velocity outputs, each ``(frames, keys)``, for mono audio at ``config.sample_rate``.

        ``instrument`` (English name) is needed only by an instrument-conditioned model.
        """
        samples = torch.from_numpy(np.clip(audio, -1.0, 1.0).astype(np.float32)).to(self.device)
        mel = self.log_mel(samples[None, :-1])  # the repo drops the last sample before the STFT
        onset, offset, _, frame, velocity = self.model(mel, instrument_indices(self.config, [instrument], self.device))
        return {name: output[0].cpu().numpy() for name, output in
                (("onset", onset), ("offset", offset), ("frame", frame), ("velocity", velocity))}

    def __call__(self, audio: np.ndarray, part: str, instrument: str | None = None) -> pd.DataFrame:
        """Note table of mono audio at ``config.sample_rate`` (columns part, pitch, velocity, start, duration)."""
        outputs = self.predict(audio, instrument)
        return decode_notes(outputs["onset"], outputs["frame"], outputs["velocity"], part, self.config)


def conditioning_instruments(config) -> list[str]:
    """The instruments an instrument-conditioned model knows, in index order; empty = unconditioned (missing key too)."""
    return list(config.get("conditioning_instruments") or [])


def instrument_indices(config, instruments: list[str | None], device) -> torch.Tensor | None:
    """The conditioning index of each instrument as a tensor, or None for an unconditioned model."""
    known = conditioning_instruments(config)
    if not known:
        return None
    missing = [name for name in instruments if name not in known]
    if missing:
        raise ValueError(f"instruments {missing} are not among the conditioning instruments {known}")
    return torch.tensor([known.index(name) for name in instruments], device=device)


def build_onsets_frames(config) -> OnsetsAndFrames:
    """The architecture with the recipe's sizes, untrained."""
    return OnsetsAndFrames(config.mel_bands, config.max_midi - config.min_midi + 1, config.model_complexity,
                           config.lstm_inference_chunk_frames, len(conditioning_instruments(config)))


def load_pretrained_onsets_frames(config) -> OnsetsAndFrames:
    """The MAESTRO model, after checking the checkpoint's md5 and loading every weight with strict key matching.

    Args:
        config: the `configs/transcribe/onsets_frames.yaml` settings.
    """
    checkpoint_md5 = hashlib.md5(Path(config.checkpoint_path).read_bytes()).hexdigest()
    if checkpoint_md5 != config.checkpoint_md5:
        raise RuntimeError(f"{config.checkpoint_path} has md5 {checkpoint_md5}, the recipe pins {config.checkpoint_md5}")
    model = build_onsets_frames(config)
    # strict on every pretrained weight; only the (zero-initialised) instrument conditioning may be missing
    missing, unexpected = model.load_state_dict(read_pickled_weights(config.checkpoint_path), strict=False)
    missing = [key for key in missing if not key.startswith(("instrument_film.", "instrument_key_bias."))]
    if missing or unexpected:
        raise RuntimeError(f"pretrained weights don't match the model: missing {missing}, unexpected {unexpected}")
    return model


def load_onsets_frames(config) -> OnsetsAndFrames:
    """The fine-tuned model if ``config.finetuned_weights_path`` is set (a state dict saved by training), else the MAESTRO one."""
    finetuned_weights_path = config.get("finetuned_weights_path")
    if not finetuned_weights_path:
        return load_pretrained_onsets_frames(config)
    model = build_onsets_frames(config)
    model.load_state_dict(torch.load(finetuned_weights_path, map_location="cpu", weights_only=True), strict=True)
    return model


def onset_peak_time(onset: np.ndarray, start: int, run_end: int, key: int, interpolate: bool) -> float:
    """Frame position (fractional if ``interpolate``) of the onset head's peak inside one above-threshold run.

    ``interpolate``: fit a parabola through the peak frame and its two neighbours and take its top,
    clipped to ±½ frame: the head's 32 ms frames are coarse next to a 50 ms onset tolerance.
    """
    peak = start + int(np.argmax(onset[start:run_end, key]))
    if not interpolate or not 0 < peak < len(onset) - 1:
        return float(peak)
    before, at, after = onset[peak - 1, key], onset[peak, key], onset[peak + 1, key]
    curvature = before - 2 * at + after
    if curvature >= 0:
        return float(peak)
    return peak + float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))


def decode_notes(onset: np.ndarray, frame: np.ndarray, velocity: np.ndarray, part: str, config) -> pd.DataFrame:
    """The repo's ``extract_notes``: a note starts where the onset head rises above its threshold and
    lasts while the onset or frame head stays above theirs. Velocity = mean over its onset frames.

    ``config.onset_time`` (missing = ``edge``, the repo's) sets where inside the onset head's above-threshold
    run the note starts: ``edge`` its first frame, ``peak`` the head's highest frame, ``peak_interp`` that
    peak refined between frames. The note's end is the same in all three.

    Args:
        onset, frame, velocity: (frames, keys) model outputs.
        part: written into the ``part`` column.
        config: the `configs/transcribe/onsets_frames.yaml` settings.
    """
    onset_time = config.get("onset_time", "edge")
    if onset_time not in ("edge", "peak", "peak_interp"):
        raise ValueError(f"unknown onset_time {onset_time!r}")
    onsets = onset > config.onset_threshold
    active = onsets | (frame > config.frame_threshold)
    rising = onsets & ~np.vstack([np.zeros((1, onsets.shape[1]), bool), onsets[:-1]])
    seconds_per_frame = config.hop_length / config.sample_rate
    rows = []
    for start, key in zip(*np.nonzero(rising)):
        run_end = start
        while run_end < len(onsets) and onsets[run_end, key]:
            run_end += 1
        end = run_end
        while end < len(active) and active[end, key]:
            end += 1
        onset_velocities = velocity[start:end, key][onsets[start:end, key]]
        note_start = (float(start) if onset_time == "edge"
                      else onset_peak_time(onset, start, run_end, key, interpolate=onset_time == "peak_interp"))
        rows.append({"part": part, "pitch": int(config.min_midi + key),
                     "velocity": int(np.clip(onset_velocities.mean() * 127, 0, 127)) if len(onset_velocities) else 0,
                     "start": note_start * seconds_per_frame, "duration": (end - note_start) * seconds_per_frame})
    table = pd.DataFrame(rows, columns=["part", "pitch", "velocity", "start", "duration"])
    return table.sort_values("start", kind="stable").reset_index(drop=True)
