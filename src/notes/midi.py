"""Note table ↔ MIDI. MIDI holds a slice of the note table: pitch, timing, velocity, one track per part.

Pitch bends (the f0 detail around each note) come from a separate curve, not from the table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from src.notes.table import note_table_to_frames


def note_table_to_pretty_midi(table: pd.DataFrame, gm_programs: dict[str, int], velocity: int | None = None,
                              pitch_shift: dict[str, int] | None = None,
                              time_offset_seconds: float = 0.0) -> pretty_midi.PrettyMIDI:
    """A note table (``start``/``duration`` in seconds) as MIDI, one track per part.

    Args:
        table: note table with ``part``, ``pitch``, ``start``, ``duration`` (+ ``velocity``).
        gm_programs: part → General MIDI program (0-based).
        velocity: one velocity for every note; None uses the table's ``velocity`` column.
        pitch_shift: part → semitones added to the pitch (sounding-pitch correction).
        time_offset_seconds: added to every note time.
    """
    pitch_shift = pitch_shift or {}
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for part_name, notes in table.groupby("part", sort=False):
        track = pretty_midi.Instrument(program=gm_programs[part_name], name=part_name)
        shift = pitch_shift.get(part_name, 0)
        for row in notes.itertuples():
            start = time_offset_seconds + float(row.start)
            track.notes.append(pretty_midi.Note(velocity=int(velocity if velocity is not None else row.velocity),
                                                pitch=int(row.pitch + shift),
                                                start=start, end=start + float(row.duration)))
        midi.instruments.append(track)
    return midi


def note_table_to_midi(table: pd.DataFrame, path: Path, gm_programs: dict[str, int], velocity: int | None = None,
                       pitch_shift: dict[str, int] | None = None) -> None:
    """Write a note table as a ``.mid`` file (see ``note_table_to_pretty_midi``)."""
    midi = note_table_to_pretty_midi(table, gm_programs, velocity, pitch_shift)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def midi_to_note_table(midi_path: str | Path) -> pd.DataFrame:
    """Every non-drum note of a MIDI file as a note table, sorted by start.

    Columns: ``part`` (track name), ``pitch``, ``velocity``, ``start``, ``duration``.
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    rows = [{"part": instrument.name, "pitch": note.pitch, "velocity": note.velocity,
             "start": note.start, "duration": note.end - note.start}
            for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes]
    table = pd.DataFrame(rows, columns=["part", "pitch", "velocity", "start", "duration"])
    return table.sort_values("start", kind="stable").reset_index(drop=True)


def add_pitch_bends(instrument: pretty_midi.Instrument, table: pd.DataFrame, semitones: np.ndarray,
                    hop_seconds: float, config, time_offset_seconds: float = 0.0) -> None:
    """Pitch-bend events that make each note follow the f0 curve around its snapped pitch.

    Args:
        instrument: the MIDI track holding ``table``'s notes.
        table: note table of one part, in the order the notes were written.
        semitones: tuned pitch per f0 frame (fractional MIDI numbers).
        hop_seconds: the f0 hop.
        config: the `configs/notes/midi.yaml` settings (bend range, minimum step).
        time_offset_seconds: added to every event time, as for the notes.
    """
    bend_range = config.pitch_bend_range_semitones
    # RPN 0 (pitch-bend sensitivity) = bend_range semitones. pretty_midi sorts same-time control changes
    # by number, which would put the data entry (CC 6) before the RPN select (CC 101/100) and the synth
    # would drop it, so the data entry goes 2 ms later
    for number, value, delay in ((101, 0, 0.0), (100, 0, 0.0), (6, bend_range, 0.002), (38, 0, 0.002)):
        instrument.control_changes.append(pretty_midi.ControlChange(number, value, time_offset_seconds + delay))
    min_step = config.pitch_bend_min_step_cents / 100.0
    start_frames, end_frames = note_table_to_frames(table, hop_seconds)
    for start_frame, end_frame, pitch in zip(start_frames, end_frames, table["pitch"].to_numpy()):
        last = None
        for frame in range(start_frame, end_frame):
            bend = float(np.clip(semitones[frame] - pitch, -bend_range, bend_range))
            if last is None or abs(bend - last) >= min_step:
                value = int(np.clip(round(bend / bend_range * 8191), -8192, 8191))
                instrument.pitch_bends.append(pretty_midi.PitchBend(value, time_offset_seconds + frame * hop_seconds))
                last = bend
    final_frame = end_frames[-1] if len(end_frames) else 0
    instrument.pitch_bends.append(pretty_midi.PitchBend(0, time_offset_seconds + final_frame * hop_seconds))
