# exp007 — Onsets and Frames fine-tuned on the six dragon MIDI

Run 2026-11-15 on sym8, GPU 0, branch `main` at `b9789c5` plus uncommitted changes (start state in `git_head.txt`, `git_status.txt`; the uncommitted code is saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`). Autonomous session. Nothing committed.

> Outputs live in `artifacts/` (→ `storage/experiments/exp007_261115_onsets_frames_finetune/`): `weights/` (`best.pt`, `last.pt`), `tables/` (training + validation logs), `baseline/` and `finetuned/` (`predictions/`, `note_tables/`, `midi/`, `tables/scores.parquet`), `logs/`.

---

## 1. Verdict

**Fine-tuning works on six dragon: test onset F1 0.26 → 0.48, and the blips are gone. But the fine-tuned model writes too few notes, and on solo it collapses (0.27 → 0.09).** Both come from the same thing: the onset head got shy. It rarely clears the 0.5 threshold, so the model writes 0.4 notes per reference note where zero-shot wrote 2.5. exp008 tunes that threshold on val and fixes most of the six dragon gap, but not solo.

| onset F1, grace windows left out | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40 stems), zero-shot → fine-tuned | 0.35 → 0.51 | 0.31 → 0.53 | 0.10 → 0.28 | 0.25 → 0.38 | 0.33 → 0.69 | 0.24 → 0.47 | **0.26 → 0.48** |
| six dragon val (59 stems) | 0.34 → 0.56 | 0.28 → 0.51 | 0.10 → 0.36 | 0.27 → 0.31 | 0.31 → 0.68 | 0.25 → 0.36 | **0.26 → 0.46** |
| solo test (581 clips) | 0.22 → 0.08 | 0.15 → 0.08 | 0.21 → 0.14 | 0.28 → 0.07 | 0.53 → 0.15 | 0.21 → 0.00 | **0.27 → 0.09** |

---

## 2. What ran

- **Start**: exp004's Onsets and Frames (Jong Wook Kim's PyTorch reimplementation, MAESTRO weights, 500k steps).
- **Training data**: the six dragon MIDI (exp001's per-stem labels) on the train songs of the frozen split: 242 stems of 49 songs, 18.76 h. Solo is left out of training (2026-11-13 decision: it's the outside test).
- **Grace-note windows** (2026-11-13 decision): exp001 gave every grace note a made-up 80 ms and started its main note 80 ms later. Around each grace note, from 50 ms before to 180 ms after its start, no loss term counts and no note is scored, either way. The audio stays whole. That's 2.0 % of training frames; it leaves out 18 % of test reference notes (the grace notes, their main notes, and anything else starting in the window).
- **Training**: the repo's recipe where it applies (8 segments of 20.48 s, Adam, gradient clip 3, binary cross-entropy on the onset, offset and frame heads). Learning rate 1e-4 constant, seed 42, no augmentation, no velocity loss. Up to 20,000 steps, validating every 500.
- **Picking the weights**: val onset F1 (grace windows left out, mean of instrument means) at the recipe's 0.5 onset threshold. Best at step 18,500 (0.463). Still creeping up at the 20,000-step cap.
- **Scoring**: test once, with the best weights. The zero-shot model went through the exact same predict + score code (`baseline/`) so the two are graded identically.
- **Run time**: 83 min training on GPU 0 (≈ 4 steps/s including validation), then 5 min predicting 680 items.

---

## 3. Scores

> Means over items per instrument, then the mean of instrument means. Grace windows left out unless noted.

| | zero-shot | fine-tuned |
|---|---|---|
| six dragon test onset F1 | 0.264 | **0.477** |
| precision / recall | 0.19 / 0.47 | 0.84 / 0.35 |
| notes written per reference note | 2.48 | 0.41 |
| share of written notes under 100 ms | 54 % | 2.5 % |
| onset-only F1 (pitch ignored) | 0.34 | 0.48 |
| onset F1, every note scored (no windows) | 0.259 | 0.436 |
| solo onset F1 | 0.267 | 0.087 |
| solo notes per reference note | 2.65 | 0.08 |

- **The blips are gone.** The share of very short notes dropped from about half to 2.5 %. That was the main thing we wanted from fine-tuning, and it came for free from training on labels that hold one note through the wobble.
- **Precision jumped, recall dropped.** The model went from writing too much to writing too little. At the 99th percentile of frames, the onset head's peak is 0.55 on six dragon and 0.38 on solo (zero-shot: 0.95, 0.98), so a 0.5 threshold throws most of its onsets away → exp008.
- **Every instrument improves on six dragon.** 가야금 most (0.33 → 0.69), 해금 least, still the worst (0.28).
- **The zero-shot numbers match exp004 exactly** when every note is scored (0.259 test, 0.254 val, 0.267 solo), so the new predict + score code grades the same way.
- **No piece memorisation**: half the test songs (4 of 8) play a piece that's also in the train songs, as a different recording. With exp008's thresholds, those score *lower* (0.61) than the unseen pieces (0.69).

---

## 4. Checks

1. **Labels by hand** on the stem with the most grace notes (0125 대금, 157 grace windows): one onset cell per reference note, 191 of the 194 80-ms notes fall inside a window, and the reference scored against itself gives F1 1.0 with and without windows.
2. **Refactored transcriber = exp004**: `predict` + decode returns exactly exp004's stored note tables on 2 six dragon stems and 2 solo clips.
3. **Smoke run** (3 stems, 20 steps) before the full one; its outputs are in `artifacts/smoke/`.
4. **Zero-shot through the new code = exp004** on all 680 items (scores above).

---

## 5. Calls I made alone

1. **Learning rate 1e-4**, constant: the repo trained from scratch at 6e-4; fine-tuning usually starts lower. Not tuned.
2. **No velocity loss**: the six dragon MIDI's velocity is a constant 80.
3. **Window 50 ms before to 180 ms after each grace start** (80 ms grace + 100 ms margin for where the main note's attack really lands). Not tuned.
4. **Masked frames drop the loss for every key**, not just the grace and main note's keys. Simpler, and at 2 % of frames it costs little.
5. **Patience 6** (stop after 6 validations without a gain). Never triggered; the run hit the cap.
6. **No wandb**: logs are parquet tables in `tables/` plus the console log in `logs/`.
7. **Validation at threshold 0.5** picked the weights. In hindsight the threshold should be tuned too (exp008, exp009).

---

## 6. Files changed

- New: `src/notes/roll.py` (note table → label rolls, windows → frame mask), `src/transcribe/onsets_frames_training.py` (labels, segment sampler, masked loss, training step), `src/pipelines/finetune_onsets_frames.py`, `predict_onsets_frames.py`, `score_onsets_frames.py`; recipes `configs/transcribe/onsets_frames_training.yaml`, `configs/datasets/answer_keys.yaml`, `configs/pipelines/{finetune,predict,score}_onsets_frames.yaml`.
- Changed: `src/transcribe/onsets_frames.py` (`predict` split from decoding; fine-tuned weights via `finetuned_weights_path`), `src/datasets/answer_keys.py` (`six_dragon_grace_windows`), `src/evaluate/transcription.py` (scoring with windows left out), `src/pipelines/common.py` (shared item listing, reference notes, audio cache paths), `src/audio/io.py` (cached resampling), `configs/transcribe/onsets_frames.yaml`.
- Audio cache (mono 16 kHz, all six dragon stems + solo test clips): `storage/cache/audio_mono_16k/`.

---

## 7. Next

- **Tune the onset threshold on val** → exp008.
- **Train longer**: val was still rising at the cap → exp009.
- **Solo**: the fine-tuned model barely writes notes on 민속악 solo clips. Fixing that probably needs solo (or other 민속악) audio in training, which changes the training data → Jae's call.
