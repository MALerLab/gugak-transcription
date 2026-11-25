# exp014 — peak onset decoding: notes start at the onset head's peak, not where it first crosses the threshold

Run 2026-11-19 on sym8, CPU only (no training: re-decodes stored model outputs), branch `improv` at `f19dac2` plus the
uncommitted decoding change (`git_head.txt`, `git_status.txt`; code in `artifacts/logs/code_at_launch.diff`). Autonomous session.

> Outputs live in `artifacts/` (→ `storage/experiments/exp014_261119_peak_onset_decoding/`): exp013 rescored at the top level
> (`tables/`, `note_tables/`, `midi/`), exp009 and exp012 rescored in `exp009/` and `exp012/`, `logs/`.

---

## 1. Verdict

**Starting each note at the onset head's peak (refined between frames) instead of its first frame above the threshold
gains +0.02–0.03 onset F1 on six dragon test, for free.** exp013 goes 0.630 → 0.656, exp009 0.655 → 0.676 (new six dragon
best). Solo moves less: exp013 +0.005, exp012 −0.007.

| onset F1, grace windows left out, thresholds retuned on val | six dragon test (40) | solo test, dropped clips out (445) |
|---|---|---|
| exp009 (six dragon only), edge → peak | 0.655 → **0.676** | 0.244 → 0.279 |
| exp012 (solo only), edge → peak | 0.489 → 0.497 | **0.628** → 0.621 |
| exp013 (both), edge → peak | 0.630 → 0.656 | 0.613 → 0.618 |

---

## 2. Why

- With the tuned onset threshold (0.15–0.2), the first frame above it sits early in the head's bump. On exp013's val
  notes matched within 100 ms, written onsets came **≈ 10 ms early** (median; six dragon −10 ms, solo −6 ms).
- The head's frames are 32 ms apart, coarse next to the 50 ms tolerance. A parabola through the peak frame and its two
  neighbours puts the note between frames.
- Val sweep on exp013's outputs (mean of six dragon val and solo val, each a mean of instrument means): edge 0.694 →
  peak 0.705 → **peak + interpolation 0.710**, best at onset threshold 0.15. A constant time shift on top of it doesn't help.

## 3. What ran

- `onset_time` in `configs/transcribe/onsets_frames.yaml`: `edge` (the repo's, what a missing key means), `peak`,
  `peak_interp` (new default). Only the note's start moves; its end and pitch don't.
- Threshold grid widened to 0.05–0.9 (added 0.05, 0.15, 0.25). Thresholds retuned per run on its own val sets: exp013 on
  both, exp009 on six dragon val, exp012 on solo val. New picks: 0.1–0.25 (exp013: 0.1–0.2).
- Test touched once per run, after the thresholds were picked.

## 4. Checks

1. **Peak decoding by hand** on a made-up onset bump (0.3, 0.9, 0.5): edge starts at frame 3, peak at 4, peak + interpolation at 4.1.
2. **Tried and dropped (val only)**: a note also starting where the frame head switches on without an onset (for legato
   note changes): recall +0.03, precision −0.04, F1 down in every setting. An exp012 + exp013 average of outputs: solo val
   +0.016, six dragon val −0.09.

## 5. Next

- Every run from exp015 on decodes with `peak_interp` and validates at onset threshold 0.15.
- Solo gains little and exp012 even loses on 해금 (0.78 → 0.73, threshold 0.1 picked on solo val). Solo val has no 민속악,
  so its picks don't carry well to the 민속악 solo test (exp012 §5).
