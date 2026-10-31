# exp004 — zero-shot Onsets and Frames on gugak audio, no fine-tuning

Run 2026-10-31 on sym8, GPU 0, branch `main` at `47e1f73` plus uncommitted changes (start state in `git_head.txt`, `git_status.txt`; the uncommitted code is also saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`). Nothing committed.

> Every output lives in `artifacts/` (→ `storage/experiments/exp004_261031_onsets_frames_zeroshot/`): `midi/` and `note_tables/` (one file per scored item), `tables/`, `logs/`.

---

## 1. Verdict

**Onsets and Frames roughly doubles exp003's Kong model on both answer keys:** onset F1 **0.26** vs 0.13 on six dragon, **0.27** vs 0.15 on solo. It wins on 94 of 99 six dragon stems and 492 of 581 solo clips.

| onset F1 (Kong → O&F) | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon (99 stems) | 0.26 → 0.33 | 0.05 → 0.28 | 0.00 → 0.10 | 0.05 → 0.26 | 0.20 → 0.32 | 0.19 → 0.25 | **0.13 → 0.26** |
| solo (581 clips) | 0.17 → 0.22 | 0.03 → 0.15 | 0.02 → 0.21 | 0.14 → 0.28 | 0.42 → 0.53 | 0.14 → 0.21 | **0.15 → 0.27** |

> The gain is mostly on the sustained instruments (피리, 해금, 아쟁), which Kong barely heard. O&F does hear their note starts, but it writes 2–3 notes per reference note, mostly very short blips. So recall went up a lot (0.23 → 0.43 on six dragon) and precision only a little (0.16 → 0.19).

Still a floor, not a usable transcriber: strand B's f0 pipeline (exp002) scores 0.38 on the same solo clips. For fine-tuning, O&F is now the stronger starting point of the two piano models.

---

## 2. What ran

- **Model**: Onsets and Frames (Hawthorne et al. 2018) in Jong Wook Kim's PyTorch reimplementation (github.com/jongwook/onsets-and-frames, MIT, commit `783ca08`). Probably the "torch version" the prof meant (not checked with him).
  - It has an **onset head** ("a note starts here") and a **frame head** ("this key is held"). A note only exists if the onset head starts it, then it lasts as long as either head stays on.
  - Weights: `model-500000.pt`, trained on MAESTRO piano for 500k steps, shared by the author in the repo's issues #17 / #23 (no official release). 106,038,101 bytes, md5 `5cdfbfc3cd1d1298a2891640baf6fa68`. On disk: `storage/weights/onsets_frames_maestro/`.
  - Settings: the repo's `transcribe.py` defaults, untouched (onset threshold 0.5, frame threshold 0.5).
- **Answer keys**: exactly exp003's. Six dragon = exp001's per-stem aligned MIDI on 99 val + test stems (9.0 h). Solo = exp002's 581 test clips (2.1 h).
- **Metric**: exp003's, unchanged. Onset F1 = onset within ±50 ms and pitch within ±50 cents.
- **Run time**: 2 min 38 s for all 680 items on one GPU (Kong took 23 min).

---

## 3. Scores

> Means over items per instrument; **all** = mean of the six instrument means. Kong numbers are from exp003. "Notes per reference note" > 1 = the model writes extra notes.

| answer key | instrument | items | precision Kong → O&F | recall Kong → O&F | **onset F1** Kong → O&F | onset-only F1 Kong → O&F | notes per reference note Kong → O&F |
|---|---|---|---|---|---|---|---|
| six dragon | 대금 daegeum | 20 | 0.44 → 0.26 | 0.20 → 0.45 | **0.26 → 0.33** | 0.33 → 0.45 | 0.48 → 1.70 |
| | 피리 piri | 20 | 0.19 → 0.22 | 0.03 → 0.40 | **0.05 → 0.28** | 0.11 → 0.39 | 0.18 → 1.76 |
| | 해금 haegeum | 20 | 0.00 → 0.10 | 0.00 → 0.11 | **0.00 → 0.10** | 0.06 → 0.30 | 0.11 → 1.05 |
| | 아쟁 ajaeng | 8 | 0.08 → 0.18 | 0.04 → 0.49 | **0.05 → 0.26** | 0.10 → 0.29 | 0.72 → 2.75 |
| | 가야금 gayageum | 16 | 0.12 → 0.21 | 0.70 → 0.69 | **0.20 → 0.32** | 0.22 → 0.35 | 6.21 → 3.52 |
| | 거문고 geomungo | 15 | 0.13 → 0.17 | 0.42 → 0.44 | **0.19 → 0.25** | 0.26 → 0.32 | 3.47 → 2.47 |
| | **all** | 99 | 0.16 → 0.19 | 0.23 → 0.43 | **0.13 → 0.26** | 0.18 → 0.35 | 1.38 → 1.96 |
| solo | 대금 daegeum | 130 | 0.19 → 0.15 | 0.18 → 0.41 | **0.17 → 0.22** | 0.34 → 0.39 | 1.11 → 2.91 |
| | 피리 piri | 134 | 0.09 → 0.11 | 0.02 → 0.29 | **0.03 → 0.15** | 0.17 → 0.39 | 0.22 → 3.00 |
| | 해금 haegeum | 27 | 0.03 → 0.16 | 0.02 → 0.35 | **0.02 → 0.21** | 0.30 → 0.41 | 0.51 → 2.48 |
| | 아쟁 ajaeng | 75 | 0.12 → 0.21 | 0.17 → 0.48 | **0.14 → 0.28** | 0.24 → 0.41 | 1.30 → 2.93 |
| | 가야금 gayageum | 179 | 0.38 → 0.45 | 0.56 → 0.67 | **0.42 → 0.53** | 0.51 → 0.61 | 1.66 → 1.51 |
| | 거문고 geomungo | 36 | 0.09 → 0.16 | 0.28 → 0.33 | **0.14 → 0.21** | 0.39 → 0.49 | 3.26 → 2.18 |
| | **all** | 581 | 0.15 → 0.21 | 0.21 → 0.42 | **0.15 → 0.27** | 0.33 → 0.45 | 1.21 → 2.40 |

- **Lenient F1** (onsets within ±100 ms): 0.32 six dragon, 0.31 solo. **Onset + offset F1**: 0.09 and 0.06. Per item: `tables/scores.parquet`; per instrument: `tables/summary_by_instrument.parquet`.
- **Overtones** (`tables/onset_matched_pitch_gaps.parquet`, from exp003's script): on six dragon, 22 % of the in-time written notes sit on harmonics 2–5 of the reference note (Kong 37 %). 가야금 is still the worst case (48 %).
- **The extra notes are short blips**: the median O&F note lasts 50–100 ms on most instruments, and few of them are one held note chopped into same-pitch pieces (3–10 %, except six dragon 아쟁 at 24 %). Probably vibrato and 시김새 read as quick new notes (not checked).

---

## 4. Checks

1. **Weights all load, with strict key matching**: 126 tensors, 26.5 M values, every model key present, none extra, all finite.
2. **Front end matches the repo's**: my log-mel against the repo's recipe rebuilt with numpy + librosa's filter: max difference 3e-5 (log scale), mel filters 7e-7.
3. **Positive control**: a rendered piano piece (56 notes, arpeggios + chords, fluidsynth with MuseScore General) → onset F1 0.97, onset + offset F1 0.88. So the model, the front end and the note decoding work.
4. **Kong still gives exactly exp003's notes** through the new model picker (one solo clip, one six dragon stem: identical note tables). exp003's committed run config still runs.
5. **Smoke test first** (3 items per answer key, `configs/score_piano_zero_shot_smoke.yaml`); its outputs were overwritten by the full run.
6. **No item came back empty** (Kong had 39).

---

## 5. Calls I made alone

1. **The checkpoint is never unpickled as code.** It's a whole pickled model from an old PyTorch (≈ 1.0, pre-1.6 file format), with the model classes' source code saved inside. Normal loading would run code from a Google Drive download, and the session's safety check blocked that. So `read_pickled_weights` reads the file with a restricted unpickler: it imports nothing and rebuilds the weight tensors only. Then the weights go into my own copy of the architecture.
2. **The architecture is copied into `src/`** (MIT, credited in the module docstring), not run from the clone. The clone in `externals/onsets-and-frames` (untracked) is now only a reference.
3. **The front end is rebuilt on `torch.stft` + `torchaudio`.** The repo's `mel.py` calls `librosa.filters.mel` positionally, which the current librosa no longer accepts, and the lab avoids librosa anyway. Check 2 shows it's the same computation.
4. **The whole stem goes in at once.** The repo does this too: its LSTMs run in 512-frame chunks carrying their state. Kong cut audio into 10 s segments.
5. **Audio not normalised**, the same as exp003: mono, resampled to 16 kHz, clipped to [-1, 1] (the repo asserts that range).
6. **The command is generalised, not copied**: `score_piano_zero_shot` now takes `run.transcriber` (`piano_transcription` | `onsets_frames`) and `src/transcribe/__init__.py` picks the model, like `run.f0_tracker` does for f0. Run configs without the key (exp003's) default to Kong.
7. **Experiment number** exp004, as reserved.
8. **No wandb, no seed**: inference only, nothing random.

---

## 6. Files changed

**Tracked files modified**
- `src/transcribe/__init__.py`: `make_transcriber` (picks the model from `run.transcriber`).
- `src/pipelines/score_piano_zero_shot.py`: uses `make_transcriber`; docstring.
- `configs/pipelines/score_piano_zero_shot.yaml`: `uses` adds `transcribe/onsets_frames`; new `run.transcriber`.

**New**
- `src/transcribe/onsets_frames.py` + `configs/transcribe/onsets_frames.yaml`: restricted checkpoint reader, the model, the front end, note decoding.
- `experiments/exp004_261031_onsets_frames_zeroshot/`: `REPORT.md`, `git_head.txt`, `git_status.txt`, `configs/score_piano_zero_shot.yaml`, `configs/score_piano_zero_shot_smoke.yaml`; `artifacts` is a git-ignored symlink.

**Untracked by design**
- `externals/onsets-and-frames/` (reference clone, ignored by `/externals/`). The summary tables come from exp003's untracked `scripts/exp003/summarize_scores.py`, run on this folder.

---

## 7. Suggested next steps

1. **Fine-tune from O&F, not Kong**: it starts from twice the score and it's a tenth of the inference time.
2. **Listen to one 피리 or 해금 render** next to its stem (`artifacts/midi/`): are the blips vibrato and 시김새?
3. **Wait for exp006** (label timing audit) before trusting any ±50 ms onset number on six dragon.
