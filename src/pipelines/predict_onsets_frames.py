"""Command: run Onsets and Frames (MAESTRO or fine-tuned weights) over the answer keys and keep its raw outputs.

Writes one ``.npz`` per item (onset, offset, frame, velocity: ``(frames, keys)`` float16) and
<predictions_dir>/items.parquet. Decoding to notes and scoring is a separate command
(src/pipelines/score_onsets_frames.py), so decoding settings can change without rerunning the model.

Usage: python -m src.pipelines.predict_onsets_frames --config experiments/<exp>/configs/predict_onsets_frames.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio.io import load_mono_resampled_cached
from src.pipelines.common import cached_audio_path, list_answer_key_items, load_run_config
from src.transcribe.onsets_frames import OnsetsFramesTranscriber


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run, transcribe_config = config.run, config.transcribe.onsets_frames
    items = list_answer_key_items(config, list(run.six_dragon.splits), with_solo=run.with_solo)
    if run.limit_items_per_answer_key:
        items = items.groupby("answer_key", group_keys=False).head(run.limit_items_per_answer_key)
    print(items.groupby(["answer_key", "split", "instrument"]).size(), flush=True)
    print(f"weights: {transcribe_config.finetuned_weights_path or transcribe_config.checkpoint_path}", flush=True)
    transcriber = OnsetsFramesTranscriber(transcribe_config, run.device)
    predictions_dir = Path(run.predictions_dir)
    rows = []
    for item in tqdm(items.to_dict("records"), desc="predict", mininterval=30):
        audio = load_mono_resampled_cached(item["audio_path"], transcribe_config.sample_rate, cached_audio_path(config, item))
        outputs = transcriber.predict(audio)
        out_path = predictions_dir / item["answer_key"] / f"{item['item_id']}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **{name: array.astype(np.float16) for name, array in outputs.items()})
        rows.append({**item, "prediction_path": str(out_path), "duration_seconds": len(audio) / transcribe_config.sample_rate})
    pd.DataFrame(rows).to_parquet(predictions_dir / "items.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/predict_onsets_frames.yaml")
    main(parser.parse_args().config)
