# exp003 — zero-shot baseline: a piano transcription model on gugak audio, no fine-tuning

Run 2026-10-28 on sym8, GPU 0, branch `main` at `1cea75b` plus uncommitted changes (start state in `git_head.txt`, `git_status.txt`). Nothing committed.

> Every output lives in `artifacts/` (→ `storage/experiments/exp003_261028_zeroshot/`): `midi/` and `note_tables/` (one file per scored item), `tables/`, `logs/`.

---

## 1. Verdict

**The baseline exists, and it is low, as expected.** The piano model gets mean onset F1 **0.13** on the six dragon labels and **0.15** on the solo clips (onset within 50 ms, pitch within 50 cents, mean over instruments).

> The model hears plucked instruments best and sustained ones barely at all. On 가야금 and 거문고 it catches most note starts but also writes each note's overtones as extra notes. On 피리, 해금 and 아쟁 it mostly stays silent, because nothing in them sounds like a piano's hammer attack.

So these numbers mostly measure the gap between piano timbre and gugak timbre, not the quality of our labels. They are still the bar any fine-tuned model has to clear.

---

## 2. What ran

- **Model**: Kong et al. 2021, "High-resolution piano transcription with pedals by regressing onset and offset times", trained on MAESTRO piano only.
  - Code: the authors' `piano_transcription_inference` package, version 0.0.6 (latest on PyPI), added with `uv add`.
  - Weights: `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth` from Zenodo record 4034264 (CC BY 4.0), 171,966,578 bytes, md5 `22b961b77c1878239fec963362097045`, which matches Zenodo's checksum.
  - On disk: `storage/weights/kong2021_piano_transcription/` (→ `~/storage/gugak-transcription/weights/`).
  - Settings: the package defaults, untouched (onset threshold 0.3, offset 0.3, frame 0.1, 10 s segments). Notes only; the pedal output is dropped.
- **Answer key 1, six dragon**: exp001's per-stem aligned MIDI (sounding pitch applied) against the matching single stems.
  - exp001's usable songs in val + test: 20 songs (12 val, 8 test), 99 stems, 9.0 h of stem audio, 28,249 reference notes.
- **Answer key 2, solo**: exp002's solo clips, test partition, each against its own MIDI.
  - 588 test clips minus exp002's 7 broken references = 581 clips, 2.1 h, 14,517 reference notes.
- **Metric**: `mir_eval` note F-measure with `configs/evaluate/transcription.yaml`.
  - **onset F1**: onset within ±50 ms and pitch within ±50 cents (the main number).
  - **lenient F1**: the same with ±100 ms onsets.
  - **onset + offset F1**: also needs the offset within 20% of the note length.
  - **onset-only F1**: pitch ignored. Did the model hear that a note starts at all? (added, see section 5)
- **Run time**: 23 min 25 s for all 680 items on one GPU, about 1.3 GB of GPU memory.

---

## 3. Scores

> Precision, recall and F1 are means over items per instrument. The **all** row is the mean of the six instrument means (exp002's convention), except the counts, which are totals. "Notes written per reference note" > 1 means the model writes extra notes.

### Six dragon labels (val + test, 99 stems)

| instrument | stems | hours | reference notes | notes written per reference note | onset precision | onset recall | **onset F1** | lenient F1 | onset + offset F1 | onset-only F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 대금 daegeum | 20 | 1.72 | 9,633 | 0.48 | 0.45 | 0.20 | **0.26** | 0.32 | 0.10 | 0.33 |
| 가야금 gayageum | 16 | 1.51 | 3,312 | 6.21 | 0.12 | 0.70 | **0.20** | 0.23 | 0.12 | 0.22 |
| 거문고 geomungo | 15 | 1.42 | 3,202 | 3.47 | 0.13 | 0.42 | **0.19** | 0.25 | 0.07 | 0.26 |
| 아쟁 ajaeng | 8 | 0.92 | 1,534 | 0.72 | 0.08 | 0.04 | **0.05** | 0.09 | 0.02 | 0.10 |
| 피리 piri | 20 | 1.72 | 5,654 | 0.18 | 0.19 | 0.03 | **0.05** | 0.07 | 0.01 | 0.11 |
| 해금 haegeum | 20 | 1.72 | 4,914 | 0.11 | 0.00 | 0.00 | **0.00** | 0.00 | 0.00 | 0.06 |
| **all** | 99 | 9.02 | 28,249 | 1.38 | 0.16 | 0.23 | **0.13** | 0.16 | 0.05 | 0.18 |

### Solo dataset (test partition, 581 clips)

| instrument | clips | hours | reference notes | notes written per reference note | onset precision | onset recall | **onset F1** | lenient F1 | onset + offset F1 | onset-only F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 가야금 gayageum | 179 | 0.52 | 4,998 | 1.66 | 0.38 | 0.56 | **0.42** | 0.44 | 0.11 | 0.51 |
| 대금 daegeum | 130 | 0.73 | 4,212 | 1.11 | 0.19 | 0.18 | **0.17** | 0.20 | 0.05 | 0.34 |
| 거문고 geomungo | 36 | 0.10 | 632 | 3.26 | 0.09 | 0.28 | **0.14** | 0.17 | 0.04 | 0.40 |
| 아쟁 ajaeng | 75 | 0.23 | 1,263 | 1.30 | 0.12 | 0.17 | **0.14** | 0.14 | 0.01 | 0.24 |
| 피리 piri | 134 | 0.44 | 2,993 | 0.22 | 0.09 | 0.02 | **0.03** | 0.04 | 0.01 | 0.17 |
| 해금 haegeum | 27 | 0.05 | 419 | 0.51 | 0.03 | 0.02 | **0.02** | 0.03 | 0.00 | 0.30 |
| **all** | 581 | 2.06 | 14,517 | 1.21 | 0.15 | 0.21 | **0.15** | 0.17 | 0.04 | 0.33 |

- **Against exp002 on the same 581 clips**: strand B's f0 pipeline scored onset F1 0.38 in its strict version. That's the fair comparison, since this run doesn't forgive a whole-clip one-semitone shift either. So the zero-shot piano model sits well below a hand-tuned pitch tracker.
- Tables: `tables/scores.parquet` (one row per item, with split and paths), `tables/summary_by_instrument.parquet` (the two tables above).

### Why the notes come out wrong

For every written note whose onset falls within 50 ms of a reference onset, I took its pitch gap to that reference note (`tables/onset_matched_pitch_gaps.parquet`).

| answer key | same pitch | +12 (harmonic 2) | +19 (harmonic 3) | +24 (harmonic 4) | +28 (harmonic 5) | other |
|---|---|---|---|---|---|---|
| six dragon | 26% | 10% | 10% | 9% | 8% | 38% |
| solo | 37% | 11% | 4% | 5% | 2% | 41% |

- **Overtones written as notes**: about 37% of the timely notes on six dragon sit exactly on harmonics 2–5 of the reference note. A piano model has learned piano overtone patterns; a 가야금's overtones don't fit them, so it writes them as extra notes.
  - Six dragon 가야금: 6.2 notes written per reference note, and only 17% of the timely ones at the right pitch.
- **해금 is heard only as overtones**: on the six dragon stems the model writes 541 notes for 4,914 reference notes, at median MIDI 91 (G6) while the reference median is 65 (F4), about +26 semitones. Not one lands on the right pitch, hence F1 0.000. The solo 해금 clips sit higher (reference median 72) and do slightly better.
- **Sustained winds and bowed strings barely trigger it**: 피리 gets 0.18–0.22 notes per reference note on both answer keys. Note changes without a fresh attack are invisible to a model trained on hammer strikes.
- **대금 is the exception among winds**: the highest six dragon F1 (0.26), with precision 0.45. When it does write a note, it's often right. It still misses most notes (recall 0.20).

---

## 4. What broke, and checks

1. **The package loads weights with `strict=False`**, so a checkpoint that doesn't fit the model would load silently with random layers.
   - The wrapper first loads the checkpoint with strict key matching (`check_checkpoint_keys`). It passed.
2. **Positive control**: I rendered two reference parts with a piano sound (fluidsynth, MuseScore General, GM 0) and ran them through the same wrapper and scoring.
   - 0010 가야금: onset F1 1.00. 0063 대금 (군악, fast): 0.998.
   - So the wrapper, the 16 kHz resampling and the scoring are right; the low gugak numbers come from the model.
3. **Onset timing checked on the smoke test**: the best whole-item lag between written and reference onsets was 0.02–0.05 s, so nothing is shifted.
4. **The package prints one line per 10 s segment.** The wrapper silences it to keep the log readable.
5. **pandas round trip**: an empty `reference_part` came back as NaN instead of None after parquet, which made a diagnostic of mine select no reference notes. I fixed the command before the full run (any non-string now means "all tracks"). In the full run no item has zero reference notes.
6. **39 items got no notes at all** and score 0: 14 solo 아쟁, 14 solo 피리, 8 solo 해금 and 3 six dragon 해금. They count in the means.
7. **Not mine**: `.gitignore` and `setup_worktree.sh` were modified at 15:03 during the run (a `todo.md` link and ignore line), presumably by another session. I left them untouched and kept them out of the commit set.
8. **Notion says "Onsets and Frames"**: Plans and Decisions (2026-10-26) name a zero-shot Onsets and Frames run. Your brief picked Kong et al. 2021 instead, and this run follows the brief. Notion isn't updated (no writes without your go-ahead).

---

## 5. Calls I made alone

1. **Weights folder name**: `kong2021_piano_transcription`. Zenodo 4034264 is the one checkpoint the package and the authors point to, so there was no version to choose.
2. **Thresholds left at the package defaults**: tuning them on our data would no longer be zero-shot.
3. **Notes only**: the pedal output is dropped. Note offsets are the model's own, not extended by its pedal.
4. **Audio in**: mono by channel average, resampled 44.1 → 16 kHz with torchaudio (`load_mono_resampled`), not with librosa as the package's example does.
5. **librosa**: the package needs `librosa` and `torchlibrosa`. `torchlibrosa` uses librosa to build the mel filters when the model is created. librosa was already in the env (pulled in by synctoolbox and torchcrepe), so `uv add` only added `piano-transcription-inference` and `torchlibrosa`. Our code never calls librosa.
6. **Six dragon items**: exp001's `usable` flag (`alignment_quality_by_song.parquet`) and the frozen split's val + test → 20 songs. Stems = the songs' `stems_with_part` in exp001's `match_table.parquet` → 99 stems.
7. **Whole stems scored, no cropping**: in all 20 songs the labels run from at most 3.2 s after the stem's start (0071) to its end, so there is almost no unlabelled audio.
8. **Broken solo references**: exp002's report counts 7 on the 588 test clips but stores no list, so I rebuilt it from exp002's test scores (`test_stability_clips.parquet`): a whole-clip pitch gap of more than one semitone between reference and tracked pitch → 7 clips, matching the report.
9. **Strict pitch only**: no forgiving of whole-clip ±1 semitone shifts (exp002's "aligned" score). Its shift was measured on the CREPE pitch, which this model doesn't have.
10. **Onset-only F1 added** (`onset_only_scores`, in `src/evaluate/transcription.py`): beyond the brief. It separates "didn't hear a note start" from "heard it at the wrong pitch".
11. **Averaging**: mean over items within an instrument, then the mean of instruments for "all", as in exp002. Val and test are pooled; per-split numbers are in `scores.parquet`.
12. **Written MIDI** uses the listening setup's GM stand-ins (`configs/notes/midi.yaml`) and the model's velocities, so renders sound like the other strands'.
13. **No seed**: inference has no random step, so there is nothing to seed.
14. **Command name** `score_piano_zero_shot`, mirroring `score_solo`. Answer-key listing in `src/datasets/answer_keys.py`; it reads exp001/exp002 tables whose paths are set in the run config, not in the recipe.
15. **Smoke test first** (2 items per answer key, its own run config `score_piano_zero_shot_smoke.yaml` and `tables/smoke_scores.parquet`). Its MIDI and note tables were overwritten by the full run.
16. **No wandb**: inference and scoring only, nothing to track over time.

---

## 6. Files changed

**Tracked files modified**
- `pyproject.toml`, `uv.lock`: `piano-transcription-inference` (+ `torchlibrosa`), via `uv add`.
- `src/evaluate/transcription.py`: `reference_notes` takes an optional `part` (one MIDI track); new `onset_only_scores`. Existing callers unchanged.

**New**
- `src/transcribe/piano_transcription.py` + `configs/transcribe/piano_transcription.yaml`: the model wrapper (strict load check, thresholds from config, audio → note table).
- `src/datasets/answer_keys.py`: (audio, reference MIDI) pairs for both answer keys.
- `src/pipelines/score_piano_zero_shot.py` + `configs/pipelines/score_piano_zero_shot.yaml`: the command.
- `experiments/exp003_261028_zeroshot/`: `REPORT.md`, `git_head.txt`, `git_status.txt`, `configs/score_piano_zero_shot.yaml`, `configs/score_piano_zero_shot_smoke.yaml`; `artifacts` is a git-ignored symlink.

**Untracked by design**
- `scripts/exp003/summarize_scores.py`: builds the summary and pitch-gap tables above.

**Proposed commit set** (nothing staged):
1. `[deps] add piano_transcription_inference`: `pyproject.toml`, `uv.lock`.
2. `[feat] add zero-shot piano transcription scoring`: the three new `src/` modules, `src/evaluate/transcription.py`, both new recipes.
3. `[exp] add exp003 zero-shot report and run configs`: the experiment folder (the `artifacts` link stays ignored).
- Leave out `.gitignore` and `setup_worktree.sh` (another session's change, section 4).

---

## 7. Suggested next steps

1. **Listen to a few renders** next to their stems (`artifacts/midi/`), especially a 가야금 stem to hear the overtone notes. I can't listen.
2. **Fine-tune on exp001's train split** (49 songs, per-stem labels) and score with this same command and answer keys. That's the real test of the labels; this baseline is the floor.
3. **Use per-item scores as a label-quality probe** (the "MIDI is not ground truth" checks): for 가야금 and 거문고, where the model reliably hears plucks, items with unusually low onset-only recall are candidate bad labels. Cheap; not done.
4. **Maybe a second zero-shot baseline from an instrument-agnostic model**, e.g. Spotify's basic-pitch (from memory, not checked). It would show how much of the gap is piano bias rather than gugak difficulty.
5. **Housekeeping**, needs your go-ahead: update Notion (Plans: zero-shot done; Decisions: Kong 2021 instead of Onsets and Frames), and add a `PROJECT.md` line that `storage/` now also holds `weights/`.

---

## 8. Discussion (after the run, 2026-10-28)

> Ideas from going over the results, not decisions. Nothing here is built.

### The model family

- **Onsets and Frames** (Hawthorne et al. 2018, from memory): audio → piano roll (time frames × 88 keys, one "is this key sounding" probability per cell) → MIDI by simple rules.
  - Two heads: the **onset** head ("a note starts here") and the **frame** head ("this key is held"). A note exists only if the onset head starts it, and the frame head only keeps it going. That gating removes most ghost notes.
  - A small velocity head gives one number per note: how hard it was struck.
- **Kong et al. 2021** (what exp003 ran): the same family and the same piano data (MAESTRO). It regresses the exact onset and offset time inside a frame instead of classifying the frame, and adds the sustain pedal.
- **Why 피리 and 해금 nearly vanish** (inferred, not checked line by line): notes are only written where something looks like a piano attack. A wind or bowed note change without a fresh attack never triggers an onset, even when the pitch is clearly there.

### Rewiring: dynamics into the frame head

- **Kong's current wiring** (checked in the package's `models.py`): velocity → onset head; onset + offset → frame head. Velocity does not reach the frame head. Each wire is `.detach()`ed, so the receiving head reads the signal but doesn't train the sender.
- **Idea (Jae)**: also feed a dynamics signal into the frame head, so it can learn that a held note may swell or hold steady instead of decaying like a struck piano string. Possibly with an instrument flag (decaying vs sustained, e.g. `constant_decay: false`).
- **Catch**: the existing velocity head knows attack strength only (its loss is masked to onsets, from memory), and our labels carry no dynamics (six dragon velocity is a constant 80). Wired in as-is, it would tell the frame head "it started loud", not "it's swelling now".
- **Proposed target**: each training stem's own loudness curve (energy per frame), which costs no annotation. A dynamics head trained on it, then wired into the frame head.
  - **Worry (Jae)**: bleed and artifacts in the curve. The curve is a training target only: at inference the head predicts loudness from the audio, so separation artifacts don't reach it directly. Bleed in the training stems does (exp002 heard bleed in 대풍류 stems) → smooth the curve and put a dB floor under it.
  - The instrument flag may be unnecessary, since the audio already says "flute". It could still help with this little data.
- **Order**: a plain fine-tune first (every head starts from the piano weights, so no learned features are thrown away), then the rewiring as an ablation on top of it, to show whether the extra wire earns its keep.

### The full pipeline shape (a working guess, not decided)

- mix → the separation model → one stem per instrument → transcription model, one stem at a time → MIDI per instrument → merged.
- Main gap: training on clean stems, running on separated stems (bleed, separation artifacts). First test as in exp002's next steps: the same songs on real vs separated stems.

### 시김새

- One of the biggest levers, for the model and for the dataset as both improve.
- A piano roll only holds whole semitones, so bends and vibrato can't live in it (strand B's notes-vs-bends question).
- The solo dataset already carries timestamped 시김새 tags → a possible 시김새 head later.
