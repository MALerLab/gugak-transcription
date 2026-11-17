# exp008 — per-instrument onset thresholds for Onsets and Frames, picked on val

Run 2026-11-15 on sym8, CPU only, branch `main` at `b9789c5` plus uncommitted changes (`git_head.txt`, `git_status.txt`). Autonomous session, straight after exp007. Nothing committed. No model was run: it re-decodes exp007's stored model outputs.

> Outputs live in `artifacts/` (→ `storage/experiments/exp008_261115_onset_thresholds/`): `baseline/` and `finetuned/`, each with `tables/` (`threshold_sweep.parquet`, `onset_thresholds.parquet`, `scores.parquet`), `note_tables/`, `midi/`; `logs/`.

---

## 1. Verdict

**With its onset threshold tuned on val, the fine-tuned model reaches onset F1 0.66 on six dragon test, against 0.30 for zero-shot tuned the same way.** The tuned threshold is 0.2 for every instrument (exp007 decoded at 0.5). The fine-tuned model now writes 0.87 notes per reference note, at precision 0.70 and recall 0.62.

**Solo stays below zero-shot (0.23 vs 0.25).** Part of that is a pitch convention, not missed notes: on solo 거문고, 96 % of the fine-tuned model's in-time notes sit exactly one semitone below the solo MIDI (section 3).

| onset F1, grace windows left out, tuned thresholds | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40 stems), zero-shot → fine-tuned | 0.37 → 0.68 | 0.32 → 0.67 | 0.11 → 0.51 | 0.26 → 0.67 | 0.42 → 0.77 | 0.31 → 0.65 | **0.30 → 0.66** |
| six dragon val (59 stems) | 0.37 → 0.69 | 0.28 → 0.67 | 0.11 → 0.57 | 0.29 → 0.54 | 0.40 → 0.77 | 0.32 → 0.61 | **0.29 → 0.64** |
| solo test (581 clips) | 0.21 → 0.18 | 0.15 → 0.22 | 0.19 → 0.39 | 0.28 → 0.25 | 0.49 → 0.37 | 0.20 → 0.00 | **0.25 → 0.23** |

---

## 2. What ran

- **Sweep**: the six dragon val stems' stored outputs (exp007, `baseline/` and `finetuned/predictions/`) decoded at onset thresholds 0.1–0.9 in steps of 0.1. Per instrument, the best mean onset F1 (grace windows left out) wins. The frame threshold stays at 0.5: it only sets how long a note lasts, never where one starts.
- **Picked** (`tables/onset_thresholds.parquet`):
  - fine-tuned: **0.2** for all six instruments
  - zero-shot: 대금 0.7, 피리 0.6, 해금 0.2, 아쟁 0.6, 가야금 0.9, 거문고 0.9 (higher: it over-writes, so it wants a stricter gate)
- **Scored** test and solo with those thresholds; solo uses each instrument's six dragon threshold. Test was never looked at while picking.
- **Run time**: about 1 min per model.

---

## 3. Scores

| six dragon test | zero-shot, 0.5 | zero-shot, tuned | fine-tuned, 0.5 (exp007) | **fine-tuned, tuned** |
|---|---|---|---|---|
| onset F1 | 0.264 | 0.296 | 0.477 | **0.657** |
| precision / recall | 0.19 / 0.47 | 0.24 / 0.41 | 0.84 / 0.35 | 0.70 / 0.62 |
| notes per reference note | 2.48 | 1.89 | 0.41 | 0.87 |
| share of notes under 100 ms | 54 % | 52 % | 2.5 % | 5.1 % |
| onset F1, every note scored (no windows) | 0.259 | 0.289 | 0.436 | 0.613 |

- **The val curve is flat around 0.2** for the fine-tuned model (0.1–0.3 within ≈ 0.05 per instrument), so the pick isn't fragile.
- **Short notes stay rare** at the lower threshold (5 %), so the blips don't come back.
- **Every instrument gains 0.26 or more over tuned zero-shot** (가야금 0.42 → 0.77, 대금 0.37 → 0.68, the rest double or better). 해금 is still the weakest (0.51).
- **Solo**: the fine-tuned model still under-writes there (0.36 notes per reference note, recall 0.17). It gains on 해금 (0.19 → 0.39) and 피리, and loses on 가야금, 대금, 아쟁 and 거문고.
- **Solo pitch convention**: of the solo notes it places in time (onset within ±50 ms of a reference note), the fine-tuned model writes 거문고 one semitone *below* the solo MIDI 96 % of the time. Zero-shot already leans that way on the same clips (48 % at −1, 29 % on the dot). On six dragon 거문고, the fine-tuned model is on the dot 98 % of the time. So the six dragon labels and the solo MIDI disagree by a semitone on 거문고, and the model learned the six dragon side. That fits exp002's finding that many solo recordings sit ≈ 40 cents sharp, right on the edge of the ±50-cent pitch tolerance (not chased further). 대금 shows a milder version (63 % on the dot, 18 % +1, 14 % −1).

---

## 4. Checks

1. **The 0.5 row of the sweep reproduces exp007's val scores** (fine-tuned 0.463, zero-shot 0.258 mean of instrument means).
2. **Test never touched while picking**: the command reads only the `val` rows of the items table.

---

## 5. Calls I made alone

1. **One onset threshold per instrument**, a grid of 0.1 steps. A finer grid or per-instrument frame thresholds weren't tried: the curve around 0.2 is flat.
2. **Solo reuses the six dragon thresholds.** Tuning on solo's dev clips would mix the outside test into the recipe.
3. **The semitone check** used a ±50 ms onset match with pitch ignored; it's a diagnostic, not a score.

---

## 6. Files changed

- New: `src/pipelines/tune_onsets_frames_thresholds.py`, `configs/pipelines/tune_onsets_frames_thresholds.yaml`.
- Changed: `src/pipelines/score_onsets_frames.py` + its recipe (`run.onset_thresholds_table`: per-instrument onset thresholds).

---

## 7. Next

- **exp009**: train longer with learning-rate decay, validating at threshold 0.2 so the weights are picked the way they'll be used.
- **Solo**: both the under-writing and the 거문고 semitone gap point at a 정악 vs 민속악 difference (tempo, note density, tuning). That needs 민속악 in training or a look at how the solo MIDI names pitches. Jae's call.
