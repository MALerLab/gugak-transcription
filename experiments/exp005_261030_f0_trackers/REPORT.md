# exp005 — f0 trackers on single stems: CREPE vs stock PESTO vs custom PESTO

Run 2026-10-30 on sym8, GPU 0, branch `main` at `a32167d` plus uncommitted changes (start state in `git_head.txt`, `git_status.txt`; the uncommitted code is also saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`).

> Every output lives in `artifacts/` (→ `storage/experiments/exp005_261030_f0_trackers/`): `f0/<answer key>/<tracker>/` (one `.npz` per item), `tables/<answer key>/`, `tables/diagnostics/`, `logs/`.

> **Two caveats on every number below.**
> 1. **The answer keys aren't ground truth.** Neither the six dragon MIDI nor the solo MIDI is verified note by note. Label errors hit all three trackers the same way, so the **ranking** is fairly safe; the **absolute** numbers are not. A flat MIDI note also counts vibrato and 시김새 as errors for every tracker.
> 2. **Custom PESTO's training audio isn't checked against our test songs.** Its training list (`nia_gugak_abs.csv`) is on another machine. If it holds our six dragon val/test stems or solo test clips, custom PESTO has heard them before (no labels, since PESTO is self-supervised, but still a head start). To check with 성욱님; rerun the affected answer key if they overlap.

---

## 1. Verdict

**Stock PESTO is the best all-rounder. Custom PESTO is not a drop-in replacement for CREPE as it stands.**

- Stock PESTO has the highest raw pitch accuracy on both answer keys: **0.84** on six dragon (CREPE 0.81, custom PESTO 0.78) and **0.62** on solo (CREPE 0.60, custom PESTO 0.54).
- **Custom PESTO can't go above 880 Hz (A5).** On the frames we score it never outputs more than 880 Hz, so notes above roughly C5 mostly come out an octave low. That sinks 대금 (raw pitch 0.34 on six dragon vs CREPE 0.79). Switching its Viterbi smoothing off doesn't help, so the ceiling sits in the network, presumably in its training data's pitch range (inferred, not checked).
- **Below the ceiling, custom PESTO is good on bowed strings**: best on six dragon 해금 (0.85 vs CREPE 0.54) and solo 아쟁 (0.69 vs 0.64).
- **CREPE has one big weak spot, 해금**: 0.54 on six dragon, where both PESTOs reach 0.82–0.85.

So for the prof's decision (custom PESTO replaces CREPE, 2026-10-26): the gugak training helps where it reaches, but the 880 Hz ceiling has to go first. Until then, stock PESTO beats CREPE on our audio and costs a fraction of the GPU time.

---

## 2. What ran

- **Three trackers**, each with its own recipe in `configs/features/f0/`, at a 10 ms hop:
  - **CREPE** (Kim et al. 2018): a CNN trained on labelled f0, through `torchcrepe`. Full model, Viterbi decoding, 16 kHz input, seed 42 for its dither. Strand B's tracker since exp002, settings unchanged.
  - **Stock PESTO** (Riou et al. 2023): self-supervised. It learns pitch without labels by requiring that a pitch-shifted input moves its output by exactly that shift. `pesto-pitch` 2.0.1, its default weights `mir-1k_g7` (trained on MIR-1K, singing voice), its default decoding (peak-weighted mean, no smoothing over time), 44.1 kHz input.
  - **Custom PESTO**: MALer Lab's PESTO retrained on gugak audio (`MALerLab/Custom_PESTO`, commit `7be9812`, checkpoint `nia_gugak.ckpt`, md5 `41d3a7fa…`). Run through the lab's own loader and its CREPE-style Viterbi (the repo's defaults), from an untracked clone in `externals/`, 44.1 kHz input.
- **Answer key 1, six dragon**: exp001's per-stem aligned MIDI against the matching single stems, val + test, the same 99 stems (20 songs, 9.0 h) as exp003.
- **Answer key 2, solo**: exp002's solo test clips against their own MIDI, minus the 7 broken references, the same 581 clips (2.1 h) as exp003.
- Each answer key ran as its own run (`configs/score_f0_six_dragon.yaml`, `configs/score_f0_solo.yaml`), so either can be rerun alone.
- **Run time** on one GPU: CREPE 9.4 min, stock PESTO 0.7 min, custom PESTO 6.9 min on six dragon; 2.2 / 0.1 / 1.7 min on solo.

### How it's scored

- The reference notes become one reference pitch per 10 ms frame. Only frames in the **middle 20–80 % of a note** at least 0.15 s long, with **no other note sounding**, are scored. That keeps the alignment's timing error at note edges and ringing overlaps out. Scored audio: 5.2 h of six dragon, 0.95 h of solo.
- **Raw pitch accuracy** (the main number): share of scored frames where the tracked pitch is within 50 cents (half a semitone) of the reference. From `mir_eval`.
- **Raw chroma accuracy**: the same, but a wrong octave still counts as right. A big gap between the two = octave errors.
- **Best-shift raw pitch accuracy**: the reference moved by −1, 0 or +1 semitone, whichever suits that tracker best on that item. For the solo MIDI's whole-clip semitone gaps.
- **Voicing is ignored.** Every tracker gives a pitch on every frame and they mark silence differently, so this compares pitch only.
- Means: per item, then per instrument, then "all" = the mean of the six instrument means (exp002/exp003's convention).

---

## 3. Scores

### Six dragon (val + test, 99 stems)

| instrument | stems | scored h | raw pitch: CREPE | stock PESTO | custom PESTO | raw chroma: CREPE | stock PESTO | custom PESTO |
|---|---|---|---|---|---|---|---|---|
| 대금 daegeum | 20 | 0.97 | 0.79 | 0.76 | **0.34** | 0.80 | 0.79 | 0.79 |
| 피리 piri | 20 | 0.96 | 0.83 | 0.83 | 0.83 | 0.84 | 0.84 | 0.84 |
| 해금 haegeum | 20 | 1.00 | **0.54** | 0.82 | 0.85 | 0.68 | 0.85 | 0.85 |
| 아쟁 ajaeng | 8 | 0.54 | 0.97 | 0.96 | 0.97 | 0.98 | 0.98 | 0.97 |
| 가야금 gayageum | 16 | 0.90 | 0.92 | 0.87 | 0.91 | 0.94 | 0.91 | 0.95 |
| 거문고 geomungo | 15 | 0.83 | 0.82 | 0.80 | 0.74 | 0.86 | 0.87 | 0.89 |
| **all** | 99 | 5.20 | 0.81 | **0.84** | 0.78 | 0.85 | 0.87 | 0.88 |

Best-shift accuracy is identical to raw pitch here: exp001 already applied each part's sounding-pitch shift.

### Solo (test partition, 581 clips)

| instrument | clips | scored h | raw pitch: CREPE | stock PESTO | custom PESTO | raw chroma: CREPE | stock PESTO | custom PESTO | best shift: CREPE | stock PESTO | custom PESTO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 대금 daegeum | 130 | 0.28 | 0.47 | 0.52 | **0.15** | 0.47 | 0.55 | 0.51 | 0.50 | 0.55 | 0.17 |
| 피리 piri | 134 | 0.20 | 0.57 | 0.60 | 0.60 | 0.57 | 0.60 | 0.60 | 0.61 | 0.64 | 0.63 |
| 해금 haegeum | 27 | 0.03 | 0.71 | 0.72 | 0.70 | 0.72 | 0.72 | 0.72 | 0.71 | 0.72 | 0.71 |
| 아쟁 ajaeng | 75 | 0.13 | 0.64 | 0.63 | 0.69 | 0.65 | 0.67 | 0.69 | 0.65 | 0.65 | 0.70 |
| 가야금 gayageum | 179 | 0.25 | 0.68 | 0.69 | 0.68 | 0.74 | 0.77 | 0.78 | 0.68 | 0.69 | 0.69 |
| 거문고 geomungo | 36 | 0.06 | 0.52 | 0.56 | 0.40 | 0.57 | 0.65 | 0.65 | 0.52 | 0.56 | 0.40 |
| **all** | 581 | 0.95 | 0.60 | **0.62** | 0.54 | 0.62 | 0.66 | 0.66 | 0.61 | 0.64 | 0.55 |

- Solo numbers sit about 0.2 below six dragon for every tracker. The biggest error type there is **±1 semitone, on 17–26 % of frames for every tracker** (six dragon: 4–6 %). Since every tracker shows it, it's most likely the solo MIDI or the recordings' tuning against the A440 grid, not the trackers (inferred, not checked). Fixing a whole clip by one semitone only helps a little, so it isn't a whole-clip transposition.
- Tables: `tables/<answer key>/scores.parquet` (one row per item × tracker), `summary_by_instrument.parquet`, `paired_wins.parquet`, `error_types.parquet`.

### Item by item (raw pitch accuracy, custom PESTO vs the others)

| | six dragon: custom beats CREPE | custom beats stock PESTO | solo: custom beats CREPE | custom beats stock PESTO |
|---|---|---|---|---|
| 대금 | 0 of 20 | 0 of 20 | 2 % | 0 % |
| 해금 | 20 of 20 | 20 of 20 | 52 % | 52 % |
| 아쟁 | 2 of 8 | 6 of 8 | 73 % | 79 % |
| 피리 | 15 of 20 | 13 of 20 | 75 % | 26 % |
| all | 48 % | 56 % | 49 % | 36 % |

---

## 4. Why custom PESTO drops: an 880 Hz ceiling

Share of scored frames where the tracker is exactly **an octave low**, by the reference note's height (`tables/diagnostics/`):

| reference note | six dragon frames | CREPE | stock PESTO | custom PESTO |
|---|---|---|---|---|
| below C4 (MIDI < 60) | 847,698 | 0.02 | 0.01 | 0.05 |
| C4–F4 | 416,854 | 0.00 | 0.00 | 0.03 |
| F#4–B4 | 352,206 | 0.00 | 0.01 | 0.06 |
| C5–F5 | 174,527 | 0.01 | 0.02 | **0.53** |
| F#5–B5 | 66,681 | 0.01 | 0.09 | **0.82** |
| C6 and up | 13,847 | 0.00 | 0.00 | **0.85** |

- Custom PESTO's highest output on any scored frame is **863 Hz** (six dragon) and **880 Hz** (solo). CREPE and stock PESTO go past 2,000 Hz. The references reach 1,109 Hz (six dragon) and 1,568 Hz (solo) at their 99.9th percentile.
- With its Viterbi switched off (the network's own peak-weighted pitch), custom PESTO scores the same on the six dragon 대금 stems (mean 0.35 vs 0.34) and still never passes 880 Hz. So the ceiling is in the network.
- Most of 대금 sits above C5 (57 % of six dragon 대금 frames), hence the collapse. Six dragon 거문고 and 가야금 also lose some frames an octave low.
- **Below C5 only**, the three are close: six dragon 0.80 / 0.84 / 0.82 (CREPE / stock / custom), solo 0.55 / 0.59 / 0.57. Custom PESTO still wins on 해금 and solo 아쟁, and stays behind on 거문고.
- Likely cause: the training audio's pitch range. The repo's data configs also list 판소리 vocals, which rarely go that high (inferred from the repo, not checked).

---

## 5. Checks

1. **The f0 split changed nothing for the old trackers**: YIN and CREPE give bit-identical `time`, `f0_hz` and `confidence` on a solo clip before and after the refactor.
2. **Custom PESTO's weights all load.** Its loader logs a `RuntimeError` about two missing keys on each start, then remaps the key names and loads again. I checked the remapped load: no missing keys, no unexpected keys, every tensor equal to the checkpoint's. The log line is harmless.
3. **Scoring checked by hand** on made-up notes: overlap frames dropped, short notes dropped, an octave error counted wrong for raw pitch and right for raw chroma, a one-semitone-sharp track scored 0 raw and 1.0 at best shift +1. All as worked out by hand.
4. **Pinning**: custom PESTO refuses to load unless the clone is at commit `7be9812` and the checkpoint's md5 matches the recipe.
5. **Smoke test first** (3 items per answer key, `configs/score_f0_*_smoke.yaml`). Its f0 files were reused by the full run (same code and settings); its tables were overwritten.
6. **Frame timing**: I moved the reference by −50 to +50 ms on 30 items per answer key (sampled with seed 42).
   - Solo: every tracker peaks within 30 ms of zero, and the curves are flat there (±0.001). No tracker is shifted.
   - Six dragon: all three trackers score slightly higher the earlier the reference is moved (about +0.005 at −50 ms, still rising). Since all three move together it's the reference, not a tracker: **exp001's per-stem aligned notes may run late by 50 ms or more**. It doesn't change this comparison; not chased further.
7. **The ±1 semitone share on solo** (section 3) is the same for all three trackers, so it doesn't change the ranking. Its cause isn't checked.

---

## 6. Calls I made alone

1. **One recipe per tracker**: `src/features/f0.py` became the package `src/features/f0/` (`yin`, `crepe`, `pesto`, `custom_pesto`, + `make_f0_tracker`), with `configs/features/f0/<tracker>.yaml`. The tracker a command runs is now `run.f0_tracker` in its recipe (it was `features.f0.f0_tracker`).
   - Consequence: **exp001's and exp002's committed run configs no longer run on this code**; they still run at `a32167d` and earlier.
2. **Each tracker as its authors ship it**: CREPE with our exp002 settings, stock PESTO with its package defaults (no smoothing), custom PESTO with its repo's defaults (Viterbi). The decoding differs, so the comparison is "tracker as shipped", not "network vs network". The Viterbi-off check (section 4) covers custom PESTO's decoder.
3. **Stock PESTO weights**: `mir-1k_g7`, the package's default. The other bundled set, `mir-1k`, wasn't run.
4. **Custom PESTO's code runs from `externals/Custom_PESTO`**, untracked (`/externals/` in `.gitignore`), imported under the name `custom_pesto_utils` so its top-level `utils` package can't clash. `setup_worktree.sh` links `externals/` into new worktrees.
5. **Custom PESTO's confidence** = the peak output activation per frame, which is what its Viterbi uses. The model has no voicing head. Not used in the scoring.
6. **Input rates**: 16 kHz for CREPE (fixed by the model), the files' own 44.1 kHz for both PESTOs (as the custom repo does).
7. **Scored frames**: middle 20–80 % of notes at least 0.15 s long, as exp001's pitch check, plus "no other note sounding".
8. **Test sets** = exp003's, for comparability. Choosing a tracker on test data is a mild selection effect; with three untuned off-the-shelf trackers I judged it negligible.
9. **Shift range** for best-shift accuracy: ±1 semitone only, matching exp003's broken-reference rule (bigger gaps = broken, already dropped).
10. **No wandb**: inference and scoring only.
11. **Experiment number**: exp005, since exp004 is kept for your Onsets and Frames run.

---

## 7. Files changed

**Moved / split**
- `src/features/f0.py` → `src/features/f0/{__init__,yin,crepe,pesto,custom_pesto}.py`.
- `configs/features/f0.yaml` → `configs/features/f0/{yin,crepe,pesto,custom_pesto}.yaml`.

**Modified**
- `src/pipelines/{common,track_f0,score_solo,transcribe_stems,align_songs}.py`, `src/evaluate/alignment.py`, `src/transcribe/note_ify.py` (a docstring): the new f0 layout.
- `configs/pipelines/{track_f0,score_solo,transcribe_stems,align_songs}.yaml`, `configs/evaluate/alignment.yaml`: the same.
- `src/launch/make_run_config.py`: recipes may sit deeper than `<package>/<module>` (`features/f0/crepe`).
- `pyproject.toml`, `uv.lock`: `pesto-pitch`, `hmmlearn` (custom PESTO's Viterbi), via `uv add`.
- `.gitignore`: `/externals/`. `setup_worktree.sh`: links `externals/`.

**New**
- `src/evaluate/f0.py` + `configs/evaluate/f0.yaml`: reference pitch per frame, raw pitch / chroma / best-shift accuracy.
- `src/pipelines/score_f0.py` + `configs/pipelines/score_f0.yaml`: the command.
- `experiments/exp005_261030_f0_trackers/`: this report, git state, the four run configs.

**Untracked by design**
- `externals/Custom_PESTO/` (the lab's private code).
- `scripts/exp005/summarize.py`, `scripts/exp005/octave_check.py`: the tables in sections 3–5.

---

## 8. Suggested next steps

1. **Ask 성욱님** (Slack DM, when he's back): (a) the training audio list, to check overlap with our test songs; (b) whether the 880 Hz ceiling is known, and whether a retrain with higher-pitched audio (대금, 소금) is possible.
2. **Pick a tracker for now**: stock PESTO, or a per-instrument choice (custom PESTO for 해금/아쟁, stock PESTO elsewhere). Prof's call, given his custom PESTO decision.
3. **Rerun strand B's note cutting with the chosen tracker**. Its settings were tuned on CREPE's dev-set output (todo: promote the tuning scripts), so they need retuning first.
4. **Look into the solo ±1 semitone share**: per-clip tuning offset against A440 vs wrong MIDI notes. Cheap, and it's part of the "how good is the solo MIDI" study.
5. **Look into the six dragon timing lag** (section 5, check 6): if exp001's aligned notes run late by 50 ms or more, that matters for any onset-based training.
6. **Optional**: pYIN as a no-model baseline (prof mtg 2026-10-26: pYIN replaces YIN). It plugs in as one more module + recipe.
