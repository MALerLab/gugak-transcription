"""Command: pseudo-labels for unlabelled stems: a fine-tuned Onsets and Frames model transcribes them, and its
notes become training labels for the next model (self-training).

Reads a stem table (strand B's ``pick_stems`` table: ``stem_id``, ``song_id``, ``instrument``, ``stem_path``,
``split``), keeps ``run.splits``, transcribes each stem with the run's weights and decoding (per-instrument onset
thresholds from ``run.onset_thresholds_table``), and writes one MIDI per stem (one track, named after the
instrument) to <midi_dir>, plus <items_table>: one row per stem in the training-item format (answer key
``pseudo``), read by finetune_onsets_frames through ``run.pseudo_labels.table``.

Usage: python -m src.pipelines.pseudo_label_stems --config experiments/<exp>/configs/pseudo_label_stems.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.audio.io import load_mono_resampled_cached
from src.notes.midi import note_table_to_midi
from src.pipelines.common import cached_audio_path, load_run_config
from src.pipelines.score_onsets_frames import decoding_config
from src.transcribe.onsets_frames import OnsetsFramesTranscriber, decode_notes


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run, transcribe_config = config.run, config.transcribe.onsets_frames
    stems = pd.read_parquet(run.stems_table)
    stems = stems[stems["split"].isin(list(run.splits))]
    items = pd.DataFrame({"item_id": stems["stem_id"], "answer_key": "pseudo", "source_id": stems["song_id"],
                          "instrument": stems["instrument"], "split": stems["split"], "audio_path": stems["stem_path"],
                          "reference_midi_path": [str(Path(run.midi_dir) / f"{stem_id}.mid") for stem_id in stems["stem_id"]],
                          "reference_part": None}).reset_index(drop=True)
    print(items.groupby(["split", "instrument"]).size(), flush=True)
    print(f"weights: {transcribe_config.finetuned_weights_path}", flush=True)
    thresholds = pd.read_parquet(run.onset_thresholds_table)
    onset_thresholds = dict(zip(thresholds["instrument"], thresholds["onset_threshold"].astype(float)))
    transcriber = OnsetsFramesTranscriber(transcribe_config, run.device)
    gm_programs = dict(config.notes.midi.gm_programs)
    note_counts, seconds = [], []
    for item in tqdm(items.to_dict("records"), desc="pseudo-label", mininterval=30):
        audio = load_mono_resampled_cached(item["audio_path"], transcribe_config.sample_rate, cached_audio_path(config, item))
        outputs = transcriber.predict(audio, item["instrument"])
        notes = decode_notes(outputs["onset"], outputs["frame"], outputs["velocity"], item["instrument"],
                             decoding_config(transcribe_config, onset_thresholds, item["instrument"]))
        note_table_to_midi(notes, Path(item["reference_midi_path"]), gm_programs)
        note_counts.append(len(notes))
        seconds.append(len(audio) / transcribe_config.sample_rate)
    items["note_count"], items["duration_seconds"] = note_counts, seconds
    Path(run.items_table).parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(run.items_table)
    hours = items.groupby("instrument")["duration_seconds"].sum() / 3600
    print(f"{len(items)} stems, {hours.sum():.2f} h {hours.round(2).to_dict()}, "
          f"{items['note_count'].sum()} notes ({items['note_count'].sum() / items['duration_seconds'].sum():.2f} per second)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/pseudo_label_stems.yaml")
    main(parser.parse_args().config)
