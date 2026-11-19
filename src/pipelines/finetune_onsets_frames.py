"""Command: fine-tune Onsets and Frames (MAESTRO weights) on six dragon stems and / or solo clips.

``run.train_datasets`` picks the data: six dragon (the frozen split's train songs, per-stem labels, grace
windows left out of the loss), solo (the train clips of ``run.solo_split.table``), pseudo (stems labelled by an
earlier model, ``run.pseudo_labels.table``), or several, mixed by drawing segments in proportion to audio time. Every ``validate_every_steps`` it transcribes the val items of
``run.validation_datasets`` and scores onset F1 with the grace-note windows left out (mean of instrument
means per dataset, then the mean over datasets). Keeps the best-scoring weights (with ``ema_decay`` set, the live
weights and their moving average are both scored, the better one kept), stops after ``patience`` validations
without a gain or at ``max_steps``.

Writes <weights_dir>/best.pt + last.pt (state dicts) and <tables_dir>/training_log.parquet,
validation_log.parquet.

Usage: python -m src.pipelines.finetune_onsets_frames --config experiments/<exp>/configs/finetune_onsets_frames.yaml
"""
from __future__ import annotations

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.audio.io import load_mono_resampled_cached
from src.evaluate.transcription import estimated_notes, masked_and_unmasked_scores
from src.pipelines.common import (cached_audio_path, item_grace_windows, item_reference_notes, list_training_items,
                                  load_run_config)
from src.transcribe.onsets_frames import (LogMel, conditioning_instruments, decode_notes, instrument_indices,
                                          load_pretrained_onsets_frames, predict_batched)
from src.transcribe.onsets_frames_training import SegmentSampler, stem_labels, training_step


def _load_audio_job(args) -> np.ndarray:
    audio_path, sample_rate, cache_path = args
    torch.set_num_threads(1)
    return load_mono_resampled_cached(audio_path, sample_rate, cache_path)


def load_items_audio(items: pd.DataFrame, config) -> list[np.ndarray]:
    """Every item's mono audio at the model's sample rate, resampled once and cached."""
    sample_rate = config.transcribe.onsets_frames.sample_rate
    jobs = [(item["audio_path"], sample_rate, cached_audio_path(config, item)) for item in items.to_dict("records")]
    with ProcessPoolExecutor(config.run.num_workers) as pool:
        return list(tqdm(pool.map(_load_audio_job, jobs), total=len(jobs), desc="audio", mininterval=30))


def _score_validation_item(args) -> dict:
    """One validation item's scores from its model outputs (runs in a worker process)."""
    item, onset, frame, velocity, transcribe_config, config = args
    notes = decode_notes(onset, frame, velocity, item["instrument"], transcribe_config)
    reference = item_reference_notes(item)
    reference_intervals = np.stack([reference["start"], reference["start"] + reference["duration"].clip(lower=1e-3)], 1)
    return {"item_id": item["item_id"], "answer_key": item["answer_key"], "instrument": item["instrument"],
            **masked_and_unmasked_scores(reference_intervals, reference["pitch"].to_numpy(float),
                                         *estimated_notes(notes), item_grace_windows(item, config),
                                         config.evaluate.transcription)}


@torch.no_grad()
def validate(model, log_mel, items: pd.DataFrame, audios: list[np.ndarray], config,
             pool: ProcessPoolExecutor | None = None) -> pd.DataFrame:
    """Transcribe and score every validation stem; one row per stem. The model runs here, the scoring in ``pool`` if given."""
    model.eval()
    transcribe_config, device = config.transcribe.onsets_frames, next(model.parameters()).device
    validation_onset_threshold = config.run.get("validation_onset_threshold")
    if validation_onset_threshold is not None:
        transcribe_config = OmegaConf.merge(transcribe_config, {"onset_threshold": float(validation_onset_threshold)})
    records = items.to_dict("records")
    batch_frames = config.run.get("validation_batch_frames")
    if batch_frames:  # padded batches: much faster, same outputs except the last few frames of each item
        outputs = predict_batched(model, log_mel, audios,
                                  instrument_indices(transcribe_config, list(items["instrument"]), "cpu"), batch_frames)
    else:
        outputs = []
        for item, audio in zip(records, audios):
            mel = log_mel(torch.from_numpy(np.clip(audio, -1, 1)).to(device)[None, :-1])
            onset, _, _, frame, velocity = model(mel, instrument_indices(transcribe_config, [item["instrument"]], device))
            outputs.append({"onset": onset[0].cpu().numpy(), "frame": frame[0].cpu().numpy(),
                            "velocity": velocity[0].cpu().numpy()})
    jobs = [(item, output["onset"], output["frame"], output["velocity"], transcribe_config, config)
            for item, output in zip(records, outputs)]
    rows = pool.map(_score_validation_item, jobs, chunksize=4) if pool is not None else map(_score_validation_item, jobs)
    return pd.DataFrame(list(rows))


def instrument_mean(scores: pd.DataFrame, column: str) -> float:
    """Mean over instruments of the per-instrument mean (every instrument counts the same)."""
    return float(scores.groupby("instrument")[column].mean().mean())


def selection_scores(scores: pd.DataFrame, column: str) -> dict[str, float]:
    """The selection score (mean over datasets of each one's instrument mean) and each dataset's own, keyed by name."""
    per_dataset = {f"{column}_{key}": instrument_mean(group, column) for key, group in scores.groupby("answer_key")}
    return {column: float(np.mean(list(per_dataset.values()))), **per_dataset}


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run, training = config.run, config.transcribe.onsets_frames_training
    transcribe_config = config.transcribe.onsets_frames
    torch.manual_seed(training.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(run.device)
    weights_dir, tables_dir = Path(run.weights_dir), Path(run.tables_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Items, audio, labels (exp007–exp009's run configs predate the dataset settings: six dragon only)
    train_datasets = list(run.get("train_datasets", ["six_dragon"]))
    validation_datasets = list(run.get("validation_datasets", ["six_dragon"]))
    train_items = list_training_items(config, list(run.train_splits), train_datasets)
    val_items = list_training_items(config, list(run.validation_splits), validation_datasets)
    if run.limit_items:
        train_items = train_items.groupby("answer_key", group_keys=False).head(run.limit_items).reset_index(drop=True)
        val_items = val_items.groupby("answer_key", group_keys=False).head(run.limit_items).reset_index(drop=True)
    print(f"train items {train_items['answer_key'].value_counts().to_dict()}, "
          f"val items {val_items['answer_key'].value_counts().to_dict()}", flush=True)
    train_audios = load_items_audio(train_items, config)
    val_audios = load_items_audio(val_items, config)
    train_labels = [stem_labels(item_reference_notes(item), item_grace_windows(item, config), len(audio),
                                transcribe_config, training)
                    for item, audio in zip(train_items.to_dict("records"), train_audios)]
    masked_share = 1 - np.concatenate([labels["mask"] for labels in train_labels]).mean()
    seconds = pd.Series([len(audio) for audio in train_audios]) / transcribe_config.sample_rate
    hours = seconds.groupby(train_items["answer_key"].to_numpy()).sum() / 3600
    print(f"train audio {hours.sum():.2f} h {hours.round(2).to_dict()}, frames masked {masked_share:.3f}", flush=True)
    known_instruments = conditioning_instruments(transcribe_config)
    sampler = SegmentSampler(train_audios, train_labels, training.segment_frames, transcribe_config.hop_length,
                             device, training.seed,
                             [known_instruments.index(name) for name in train_items["instrument"]] if known_instruments else None)
    print(f"items shorter than a segment, padded: {sampler.padded_count}", flush=True)

    # 2. Model, optimiser, the zero-shot val score as the starting point
    # Validation scoring runs in worker processes (spawned: this process holds a CUDA context)
    pool = ProcessPoolExecutor(run.num_workers, mp_context=multiprocessing.get_context("spawn"))
    model = load_pretrained_onsets_frames(transcribe_config).to(device)
    log_mel = LogMel(transcribe_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    # exp007's run config predates the decay settings: it ran at a constant learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, training.get("learning_rate_decay_steps", training.max_steps),
                                                training.get("learning_rate_decay_rate", 1.0))
    validation_rows, training_rows = [], []
    scores = validate(model, log_mel, val_items, val_audios, config, pool)
    selection = selection_scores(scores, run.selection_metric)
    best_score = selection[run.selection_metric]
    validation_rows.append({"step": 0, **selection, **selection_scores(scores, "onset_f1")})
    torch.save(model.state_dict(), weights_dir / "best.pt")
    best_step, best_weights, validations_without_gain = 0, "live", 0
    print(f"step 0 val {run.selection_metric} {best_score:.4f}", flush=True)

    # Optional moving average of the weights (exp019 on): validated next to the live weights, the better one is kept
    ema_decay = training.get("ema_decay")
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
                                               use_buffers=True) if ema_decay else None)

    # 3. Train, validate, keep the best
    started = time.time()
    for step in range(1, training.max_steps + 1):
        losses = training_step(model, log_mel, optimizer, sampler.sample(training.batch_size), training)
        scheduler.step()
        if ema is not None:
            ema.update_parameters(model)
        if step % training.log_every_steps == 0:
            training_rows.append({"step": step, "seconds": time.time() - started,
                                  "learning_rate": scheduler.get_last_lr()[0], **losses})
            print(f"step {step} loss {losses['loss']:.4f} ({time.time() - started:.0f} s)", flush=True)
        if step % training.validate_every_steps:
            continue
        candidates = {"live": model} if ema is None else {"live": model, "ema": ema.module}
        gained = False
        for weights, candidate in candidates.items():
            for module in candidate.modules():  # the average's copied LSTM weights aren't one contiguous block
                if isinstance(module, torch.nn.LSTM):
                    module.flatten_parameters()
            scores = validate(candidate, log_mel, val_items, val_audios, config, pool)
            selection = selection_scores(scores, run.selection_metric)
            score = selection[run.selection_metric]
            validation_rows.append({"step": step, "weights": weights, **selection, **selection_scores(scores, "onset_f1")})
            if score > best_score:
                best_score, best_step, best_weights, gained = score, step, weights, True
                torch.save(candidate.state_dict(), weights_dir / "best.pt")
            per_dataset = ", ".join(f"{name.removeprefix(run.selection_metric + '_')} {value:.4f}"
                                    for name, value in selection.items() if name != run.selection_metric)
            print(f"step {step} val{'' if weights == 'live' else ' ema'} {run.selection_metric} {score:.4f} [{per_dataset}] "
                  f"(best {best_score:.4f} at {best_step})", flush=True)
        validations_without_gain = 0 if gained else validations_without_gain + 1
        pd.DataFrame(training_rows).to_parquet(tables_dir / "training_log.parquet")
        pd.DataFrame(validation_rows).to_parquet(tables_dir / "validation_log.parquet")
        if validations_without_gain >= training.patience:
            print(f"no val gain for {training.patience} validations, stopping", flush=True)
            break
    torch.save(model.state_dict(), weights_dir / "last.pt")
    pd.DataFrame(training_rows).to_parquet(tables_dir / "training_log.parquet")
    pd.DataFrame(validation_rows).to_parquet(tables_dir / "validation_log.parquet")
    pool.shutdown()
    print(f"done: best val {run.selection_metric} {best_score:.4f} at step {best_step} ({best_weights} weights)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/finetune_onsets_frames.yaml")
    main(parser.parse_args().config)
