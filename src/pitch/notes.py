"""Note-ify: an f0 curve (+ the stem's energy) → note events → MIDI.

Steps: frame energy → voicing → tuning offset from A440 → segmentation into notes → MIDI.
Two segmentation methods: ``naive`` (snap every frame to a semitone, merge equal runs) and
``stability`` (hysteresis on a smoothed pitch curve, so vibrato and short bends stay inside
one note, plus energy onsets to split repeated same-pitch notes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pretty_midi
from scipy.ndimage import median_filter


@dataclass
class Note:
    start_frame: int
    end_frame: int  # exclusive
    pitch: int      # MIDI note number on the tuning-corrected grid


# --- frame features ---------------------------------------------------------

def frame_rms_db(mono: np.ndarray, sample_rate: int, num_frames: int, hop_seconds: float, config) -> np.ndarray:
    """RMS in dBFS of a window centred on each f0 frame; silence (below the floor) → -240 dB."""
    hop = int(round(hop_seconds * sample_rate))
    half_window = int(round(config.energy_window_seconds * sample_rate)) // 2
    padded = np.pad(mono.astype(np.float64), half_window)
    prefix = np.concatenate(([0.0], np.cumsum(padded * padded)))
    centres = np.arange(num_frames) * hop
    starts = np.clip(centres, 0, len(padded) - 1)
    ends = np.clip(centres + 2 * half_window, 0, len(padded))
    rms = np.sqrt(np.maximum(prefix[ends] - prefix[starts], 0) / np.maximum(ends - starts, 1))
    return np.where(rms > config.silence_floor, 20 * np.log10(np.maximum(rms, 1e-12)), -240.0)


def voiced_mask(confidence: np.ndarray, rms_db: np.ndarray, hop_seconds: float, config) -> np.ndarray:
    """Frames that carry a note: confident f0 and loud enough, with short gaps bridged."""
    audible = rms_db > -240.0
    loud_level = np.percentile(rms_db[audible], 95) if audible.any() else 0.0
    voiced = (confidence >= config.confidence_threshold) & (rms_db >= loud_level - config.energy_gate_db)
    return bridge_short_gaps(voiced, int(round(config.gap_fill_seconds / hop_seconds)))


def bridge_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Set False runs of at most ``max_gap_frames`` that sit between True frames to True."""
    mask = mask.copy()
    runs = runs_of_true(~mask)
    for start, end in runs:
        if start > 0 and end < len(mask) and end - start <= max_gap_frames:
            mask[start:end] = True
    return mask


def runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each contiguous True run, end exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def hz_to_cents_from_a440(f0_hz: np.ndarray) -> np.ndarray:
    """Pitch in cents relative to A4 = 440 Hz."""
    return 1200 * np.log2(np.maximum(f0_hz, 1e-3) / 440.0)


# --- tuning -----------------------------------------------------------------

def estimate_tuning_offset(cents: np.ndarray, voiced: np.ndarray, config) -> float:
    """The recording's offset from the A440 semitone grid, in cents, in [-50, 50).

    Takes stable voiced frames, folds their pitch onto one semitone (cents mod 100), and
    returns the peak of the circularly smoothed histogram — the grid the player sits on.
    """
    slope = np.abs(np.gradient(cents))
    stable = voiced & (slope < config.tuning_stable_slope_cents_per_frame)
    if stable.sum() < 10:
        return 0.0
    folded = np.mod(cents[stable], 100.0)
    histogram = np.bincount(np.floor(folded).astype(int) % 100, minlength=100).astype(float)
    kernel_offsets = np.arange(-50, 50)
    kernel = np.exp(-0.5 * (kernel_offsets / config.tuning_smoothing_cents) ** 2)
    # circular convolution: the fold wraps, so 99 cents sits next to 0 cents
    smoothed = np.real(np.fft.ifft(np.fft.fft(histogram) * np.fft.fft(np.roll(kernel, -50))))
    peak = float(np.argmax(smoothed)) + 0.5
    return peak - 100.0 if peak >= 50.0 else peak


# --- segmentation -----------------------------------------------------------

def segment_naive(semitones: np.ndarray, voiced: np.ndarray, min_note_frames: int) -> list[Note]:
    """Snap every voiced frame to the nearest semitone and merge equal neighbours."""
    notes = []
    snapped = np.round(semitones).astype(int)
    for region_start, region_end in runs_of_true(voiced):
        start = region_start
        for frame in range(region_start + 1, region_end + 1):
            if frame == region_end or snapped[frame] != snapped[start]:
                notes.append(Note(start, frame, int(snapped[start])))
                start = frame
    return [n for n in notes if n.end_frame - n.start_frame >= min_note_frames]


def segment_region_by_hysteresis(curve: np.ndarray, offset: int, hysteresis: float, min_switch_frames: int) -> list[Note]:
    """Split one voiced region where the pitch leaves the current note far enough, for long enough.

    The new note is back-dated to the frame where the departure began.
    """
    notes = []
    current = int(np.round(curve[0]))
    note_start = 0
    candidate, departure_start = None, None
    for frame, value in enumerate(curve):
        if abs(value - current) <= hysteresis:
            candidate = None
            continue
        rounded = int(np.round(value))
        if rounded != candidate:
            candidate, departure_start = rounded, frame
        if frame - departure_start + 1 >= min_switch_frames:
            notes.append(Note(offset + note_start, offset + departure_start, current))
            current, note_start, candidate = candidate, departure_start, None
    notes.append(Note(offset + note_start, offset + len(curve), current))
    return notes


def onset_frames(rms_db: np.ndarray, hop_seconds: float, config) -> np.ndarray:
    """Frames where the energy rises by ``onset_rise_db`` within a short window (local peaks only)."""
    window = max(1, int(round(config.onset_rise_window_seconds / hop_seconds)))
    floor = np.maximum(rms_db, -120.0)
    previous_min = np.array([floor[max(0, i - window):i].min() if i > 0 else floor[0] for i in range(len(floor))])
    rise = floor - previous_min
    min_gap = max(1, int(round(config.min_inter_onset_seconds / hop_seconds)))
    candidates = np.flatnonzero(rise >= config.onset_rise_db)
    onsets = []
    for frame in candidates:
        if onsets and frame - onsets[-1] < min_gap:
            if rise[frame] > rise[onsets[-1]]:
                onsets[-1] = frame
            continue
        onsets.append(frame)
    # place the onset at the start of the rise, not its top
    return np.array([max(0, f - window // 2) for f in onsets], dtype=int)


def split_at_onsets(notes: list[Note], onsets: np.ndarray, min_note_frames: int) -> list[Note]:
    """Cut notes at energy onsets that fall inside them (not too close to either end)."""
    result = []
    for note in notes:
        inside = onsets[(onsets >= note.start_frame + min_note_frames) & (onsets <= note.end_frame - min_note_frames)]
        bounds = [note.start_frame, *inside.tolist(), note.end_frame]
        result += [Note(a, b, note.pitch) for a, b in zip(bounds[:-1], bounds[1:])]
    return result


def merge_short_notes(notes: list[Note], min_note_frames: int) -> list[Note]:
    """Fold notes shorter than the minimum into a touching neighbour; drop isolated ones."""
    notes = list(notes)
    changed = True
    while changed:
        changed = False
        for index, note in enumerate(notes):
            if note.end_frame - note.start_frame >= min_note_frames:
                continue
            previous = notes[index - 1] if index > 0 and notes[index - 1].end_frame == note.start_frame else None
            following = notes[index + 1] if index + 1 < len(notes) and notes[index + 1].start_frame == note.end_frame else None
            if previous is not None and (following is None or
                                         abs(previous.pitch - note.pitch) <= abs(following.pitch - note.pitch)):
                previous.end_frame = note.end_frame
            elif following is not None:
                following.start_frame = note.start_frame
            del notes[index]
            changed = True
            break
    return notes


def repitch_and_merge(notes: list[Note], semitones: np.ndarray) -> list[Note]:
    """Give each note the median pitch of its frames, then merge touching notes that now share a pitch."""
    for note in notes:
        note.pitch = int(np.round(np.median(semitones[note.start_frame:note.end_frame])))
    merged = []
    for note in notes:
        if merged and merged[-1].end_frame == note.start_frame and merged[-1].pitch == note.pitch:
            merged[-1].end_frame = note.end_frame
        else:
            merged.append(note)
    return merged


def segment_stability(semitones: np.ndarray, voiced: np.ndarray, rms_db: np.ndarray, hop_seconds: float, config) -> list[Note]:
    """Hysteresis segmentation of each voiced region, then onset splits and short-note merging."""
    median_frames = max(1, int(round(config.median_filter_seconds / hop_seconds)))
    min_switch_frames = max(1, int(round(config.min_switch_seconds / hop_seconds)))
    min_note_frames = max(1, int(round(config.min_note_seconds / hop_seconds)))
    notes = []
    for region_start, region_end in runs_of_true(voiced):
        curve = median_filter(semitones[region_start:region_end], size=median_frames, mode="nearest")
        notes += segment_region_by_hysteresis(curve, region_start, config.hysteresis_semitones, min_switch_frames)
    notes = repitch_and_merge(notes, semitones)
    if config.onset_split:
        notes = split_at_onsets(notes, onset_frames(rms_db, hop_seconds, config), min_note_frames)
    notes = merge_short_notes(notes, min_note_frames)
    # merging keeps the survivor's pitch; re-pitch over the grown span but keep onset splits
    for note in notes:
        note.pitch = int(np.round(np.median(semitones[note.start_frame:note.end_frame])))
    return notes


# --- top level --------------------------------------------------------------

def notes_from_f0(track: dict, mono_16k: np.ndarray, sample_rate: int, hop_seconds: float, config) -> dict:
    """Run the whole note-ify step on one f0 track.

    Args:
        track: ``time``, ``f0_hz``, ``confidence`` arrays from src/pitch/track.py.
        mono_16k: the same audio the f0 was tracked on (for energy).
        sample_rate: its sample rate.
        hop_seconds: the f0 hop.
        config: the `configs/pitch/notes.yaml` settings.

    Returns:
        dict with ``notes`` (list[Note]), ``tuning_offset_cents``, ``semitones`` (tuned, fractional MIDI
        numbers per frame), ``voiced`` and ``rms_db`` per frame.
    """
    num_frames = len(track["f0_hz"])
    rms_db = frame_rms_db(mono_16k, sample_rate, num_frames, hop_seconds, config)
    voiced = voiced_mask(track["confidence"], rms_db, hop_seconds, config)
    cents = hz_to_cents_from_a440(track["f0_hz"])
    tuning_offset_cents = estimate_tuning_offset(cents, voiced, config)
    snap_offset_cents = tuning_offset_cents if config.tuning_correction else 0.0
    semitones = 69 + (cents - snap_offset_cents) / 100.0
    if config.method == "naive":
        notes = segment_naive(semitones, voiced, max(1, int(round(config.min_note_seconds / hop_seconds))))
    else:
        notes = segment_stability(semitones, voiced, rms_db, hop_seconds, config)
    return {"notes": notes, "tuning_offset_cents": tuning_offset_cents, "semitones": semitones,
            "voiced": voiced, "rms_db": rms_db}


def note_velocities(notes: list[Note], rms_db: np.ndarray, velocity_range) -> list[int]:
    """Map each note's peak RMS (dB) linearly onto the velocity range, per recording."""
    if not notes:
        return []
    peaks = np.array([rms_db[n.start_frame:n.end_frame].max() for n in notes])
    low, high = np.percentile(peaks, 5), np.percentile(peaks, 95)
    scaled = np.clip((peaks - low) / max(high - low, 1e-6), 0, 1)
    return [int(round(velocity_range[0] + s * (velocity_range[1] - velocity_range[0]))) for s in scaled]


def notes_to_midi(result: dict, hop_seconds: float, program: int, config, with_pitch_bends: bool = False,
                  time_offset_seconds: float = 0.0) -> pretty_midi.PrettyMIDI:
    """Write note events (and optionally the f0 detail as pitch bends) as a one-track MIDI.

    Note pitches sit on the tuning-corrected grid; with bends, the bend follows the tuned curve,
    so the rendered pitch is the performed pitch minus the recording's tuning offset.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=program)
    notes, semitones = result["notes"], result["semitones"]
    for note, velocity in zip(notes, note_velocities(notes, result["rms_db"], config.velocity_range)):
        instrument.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=note.pitch,
            start=time_offset_seconds + note.start_frame * hop_seconds,
            end=time_offset_seconds + note.end_frame * hop_seconds))
    if with_pitch_bends:
        add_pitch_bends(instrument, notes, semitones, hop_seconds, config, time_offset_seconds)
    midi.instruments.append(instrument)
    return midi


def add_pitch_bends(instrument: pretty_midi.Instrument, notes: list[Note], semitones: np.ndarray,
                    hop_seconds: float, config, time_offset_seconds: float) -> None:
    """Pitch-bend events that make each note follow the f0 curve around its snapped pitch."""
    bend_range = config.pitch_bend_range_semitones
    # RPN 0 (pitch-bend sensitivity) = bend_range semitones. pretty_midi sorts same-time control changes
    # by number, which would put the data entry (CC 6) before the RPN select (CC 101/100) and the synth
    # would drop it, so the data entry goes 2 ms later
    for number, value, delay in ((101, 0, 0.0), (100, 0, 0.0), (6, bend_range, 0.002), (38, 0, 0.002)):
        instrument.control_changes.append(pretty_midi.ControlChange(number, value, time_offset_seconds + delay))
    min_step = config.pitch_bend_min_step_cents / 100.0
    for note in notes:
        last = None
        for frame in range(note.start_frame, note.end_frame):
            bend = float(np.clip(semitones[frame] - note.pitch, -bend_range, bend_range))
            if last is None or abs(bend - last) >= min_step:
                value = int(np.clip(round(bend / bend_range * 8191), -8192, 8191))
                instrument.pitch_bends.append(pretty_midi.PitchBend(value, time_offset_seconds + frame * hop_seconds))
                last = bend
    instrument.pitch_bends.append(pretty_midi.PitchBend(0, time_offset_seconds + (notes[-1].end_frame if notes else 0) * hop_seconds))
