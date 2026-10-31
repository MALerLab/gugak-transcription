# exp006 — label timing audit: do exp001's six dragon labels run late against their stems?

Run 2026-10-31 on sym8, GPU 1, branch `main` at `47e1f73` plus uncommitted changes (start state in `git_head.txt`, `git_status.txt`; the uncommitted code is also saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`). Nothing committed.

> Every output lives in `artifacts/` (→ `storage/experiments/exp006_261031_label_timing/`): `f0/` (stock PESTO, one `.npz` per audio file), `tables/`, `logs/`.

---

## 1. Verdict

**The per-stem labels are on time, and steadily so. Nothing to fix before fine-tuning.** Their best fit sits at 0 to −10 ms (−10 ms = the labels run 10 ms late, one f0 frame), and it holds along the whole song. No stem is off by more than 50 ms.

**The per-song labels wobble.** Their median is also close to zero, but the fit wanders along the song, and a quarter of their 30 s windows are off by more than 50 ms. That backs the prof's per-stem decision (2026-10-26) with a number.

| how far the labels sit from the audio (best shift, ms) | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 |
|---|---|---|---|---|---|---|
| **per-stem**, median over stems | 0 | −10 | −10 | −10 | 0 | 0 |
| per-stem, middle 80 % of stems | −10…0 | −20…0 | −30…0 | −20…−4 | −10…0 | −10…0 |
| per-song, median over stems | +10 | 0 | −10 | **+40** | −10 | −20 |
| per-song, middle 80 % of stems | −10…+32 | −50…+20 | −62…+10 | −30…+70 | −20…0 | −30…0 |

> Negative = the labels are late, positive = early. 대금 to 아쟁 are judged by pitch, 가야금 and 거문고 by onsets (section 2). With onsets scored at ±50 ms, a 10 ms bias costs next to nothing.

**exp005's hint was a false alarm.** It scored only the middles of notes, which barely react to timing: across −100…+100 ms its accuracy moves between 0.829 and 0.853, so "still rising at −50 ms" was +0.003. Scored on whole notes, the same 99 stems peak exactly at 0 ms (check 3).

---

## 2. What ran

The idea: slide each stem's labels earlier and later, 10 ms at a time from −300 to +300 ms, and see where they fit the audio best. Same trick as the master↔stem lag in the separation repo, but with notes instead of waveforms.

- **Labels**: exp001's aligned MIDI for every usable song in **every split** (train 242 stems, val 59, test 40; 69 songs, 341 stems). Both label sets ran on the same stems:
  - **per-stem**: one warping path per stem, from that stem and its own part (the labels we'd fine-tune on).
  - **per-song**: one warping path from the partial sum-stem, used for every part.
- **Pitch judge** (every instrument): stock PESTO's f0 curve on the stem (exp005's best all-rounder, 10 ms frames). Score at each shift = share of in-note frames where the tracked pitch is within 50 cents of the label's pitch. **Whole notes** count, since the timing evidence sits where the pitch changes, which is exactly what exp005 left out on purpose.
- **Onset judge** (가야금, 거문고): spectral-flux note starts detected in the stem (exp001's detector and settings), matched to the label note starts within ±25 ms. Score = onset F-measure at each shift.
  - Plucked notes ring on at the same pitch, so the pitch judge can't time them: on 가야금 / 거문고 its curve is nearly flat and creeps towards +300 ms. **The table above uses the onset judge for plucked, the pitch judge for the rest.** Both are in the tables.
- **Steady or wobbly**: the best shift again, per 30 s window. A window counts with ≥ 5 s of scored audio and ≥ 10 label pitch changes (sustained instruments only; plucked stems rarely have enough single-note frames). Spread = middle half of the window shifts; trend = straight-line slope over the song.
- **Control**: the solo answer key (581 test clips, their own MIDI, timed independently of exp001) through the same judges. A judge that's early or late by itself would show up there too.
- **Run time**: 4 min 20 s on one GPU (f0 for 914 audio files 2 min 16 s, then scoring 2 min).

---

## 3. Results

### Per-stem labels (341 stems)

| instrument | stems | judge | median best shift | stems off by > 30 ms | > 50 ms | score at 0 → at best | window spread (median) | windows used |
|---|---|---|---|---|---|---|---|---|
| 대금 | 69 | pitch | 0 ms | 0 | 0 | 0.719 → 0.719 | 10 ms | 653 |
| 피리 | 69 | pitch | −10 ms | 0 | 0 | 0.765 → 0.765 | 10 ms | 602 |
| 해금 | 69 | pitch | −10 ms | 4 | 0 | 0.752 → 0.753 | 20 ms | 591 |
| 아쟁 | 25 | pitch | −10 ms | 0 | 0 | 0.914 → 0.914 | 15 ms | 164 |
| 가야금 | 57 | onsets | 0 ms | 0 | 0 | 0.533 → 0.543 | – | – |
| 거문고 | 52 | onsets | 0 ms | 0 | 0 | 0.356 → 0.363 | – | – |

- **Shifting buys nothing**: the score at the best shift equals the score at 0 to within 0.01.
- **No drift**: the slope of the window shifts is 0 to ±2 ms per minute.
- **Windows** (sustained instruments): 6 % of the 2,010 usable windows are off by more than 50 ms, 0.7 % by more than 100 ms.

### Per-song labels (same stems)

| instrument | median best shift | stems off by > 50 ms | score at 0 (per-song vs per-stem) | window spread (median) |
|---|---|---|---|---|
| 대금 | +10 ms | 0 | 0.664 vs 0.719 | 28 ms |
| 피리 | 0 ms | 8 | 0.725 vs 0.765 | 30 ms |
| 해금 | −10 ms | 10 | 0.766 vs 0.752 | 30 ms |
| 아쟁 | **+40 ms** | 6 | 0.890 vs 0.914 | 25 ms |
| 가야금 | −10 ms | 0 | 0.436 vs 0.533 (onsets) | – |
| 거문고 | −20 ms | 0 | 0.353 vs 0.356 (onsets) | – |

- **Windows** (sustained instruments): 25 % of the 2,011 usable windows are off by more than 50 ms, 5 % by more than 100 ms (per-stem: 6 % and 0.7 %).
- **This answers exp001's open question** ("steady offset or random wobble?"): the per-song path differs from each stem's own timing by **wobble**, not a steady offset. One exception: 아쟁 sits steadily about 40 ms early.
- Per-song fits worse on most instruments, but it's not a fair fight on pitch: per-stem alignment was fitted to that same stem's chroma. The onset judge doesn't share that bias, and it also prefers per-stem on 가야금 (0.53 vs 0.44).

### Solo control (581 clips)

- **Pitch judge** (sustained): median best shift 0 to +20 ms (피리 +20). The pitch judge isn't biased.
- **Onset judge** (plucked): −20 ms (가야금) and −25 ms (거문고), and the onset F-measure jumps when shifted (가야금 0.30 → 0.67 at ±25 ms). The same detector gives 0 ms on six dragon, so **the solo dataset's plucked MIDI starts ≈ 20–25 ms after the attack** (not chased, see next steps).

---

## 4. Checks

1. **Planted errors come back out** (made-up notes with a perfect f0 curve, 3 min): labels 60 ms late → −60 ms, window spread 0 · on time → 0 · drift 0→120 ms over 3 min → slope −37 ms/min (planted −40) · ±80 ms wobble → spread 65 ms, median ≈ 0 · onsets 40 ms late → −40 ms.
2. **No real peak gets flagged**: `peak_at_range_edge` marks items whose best shift lands at ±300 ms. Under the judges used in the tables: 0 per-stem stems, 2 per-song 피리 stems.
3. **exp005's rule vs whole notes**, same 99 val + test per-stem stems, pooled: note middles 0.829–0.853 across −100…+100 ms (flat, peak −40 ms by +0.003) · whole notes 0.748 → **0.779 at 0 ms** → 0.748.
4. **Smoke test first** (4 items per label set, `configs/audit_label_timing_smoke.yaml`). It's where the plucked pitch-judge problem showed up; its f0 files were reused by the full run.

---

## 5. Calls I made alone

1. **Whole notes for the pitch judge**, not exp005's note middles (check 3 is why).
2. **Judge per instrument**: onsets for 가야금 / 거문고, pitch for the rest, decided after the smoke test showed the plucked pitch curves are flat.
3. **Every split**, not just val + test: the train labels are the ones a fine-tune would learn from.
4. **Search range ±300 ms in 10 ms steps** (= the f0 hop); **30 s windows** with ≥ 5 s scored and ≥ 10 pitch changes; **onset tolerance ±25 ms**, tight so the F-measure peaks sharply.
5. **Stock PESTO** as the pitch judge (exp005's best all-rounder); custom PESTO's 880 Hz ceiling would have hurt 대금.
6. **Onsets at 22,050 Hz**, exp001's alignment-check rate.
7. **Experiment number** exp006 (exp005 taken). **No wandb, no seed**: nothing random.

---

## 6. Files changed

**New**
- `src/evaluate/label_timing.py` + `configs/evaluate/label_timing.yaml`: shift scans for both judges, window best shifts, drift summary.
- `src/pipelines/audit_label_timing.py` + `configs/pipelines/audit_label_timing.yaml`: the command.
- `experiments/exp006_261031_label_timing/`: `REPORT.md`, `git_head.txt`, `git_status.txt`, `configs/audit_label_timing.yaml`, `configs/audit_label_timing_smoke.yaml`; `artifacts` is a git-ignored symlink.

**Tables** (`artifacts/tables/`): `items.parquet` (best shift per stem × label set × judge), `curves.parquet` (score at every shift), `windows.parquet`, `summary_by_instrument.parquet`.

---

## 7. Suggested next steps

1. **Fine-tune on the per-stem labels as they are**: no timing fix needed.
2. **Drop per-song labels for good** in anything onset-based (already the plan).
3. **Solo plucked MIDI ≈ 20–25 ms late**: add to the "how good is the solo MIDI" study. Not urgent at ±50 ms, but it matters for any tighter onset metric.
