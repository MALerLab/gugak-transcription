# exp022 — wider triangle onset targets: each note start spread over ±2 frames

Run 2026-11-19 on sym8, GPU 1, branch `improv` at `6603484` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp021 and exp023.

> Outputs live in `artifacts/` (→ `storage/experiments/exp022_261119_wide_triangle_onset_targets/`), same layout as exp020's.

---

## 1. Verdict

**The wider triangle is the best single change of the day on six dragon: test 0.660 → 0.668 against the control
exp020 (paired 95 % interval +0.002…+0.016).** Solo test 0.623 → 0.630 (−0.002…+0.016), the same as exp018's narrow
triangle. Best tuned val of all runs (0.725).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp020 → exp022 | 0.67 → 0.68 | 0.67 → 0.68 | 0.51 → 0.52 | 0.70 → 0.71 | 0.77 → 0.78 | 0.64 → 0.65 | **0.660 → 0.668** |
| solo test, dropped clips out (445), exp020 → exp022 | 0.46 → 0.48 | 0.64 → 0.65 | 0.72 → 0.73 | 0.55 → 0.56 | 0.74 → 0.71 | 0.63 → 0.65 | **0.623 → 0.630** |

## 2. What ran

- **exp018's run** with the triangle twice as wide: target 1 − |frame − exact start| / 2, so a start reaches up to 4
  frames (`onset_label_half_width_frames: 2.0`); e.g. a start on frame 10 gives 0.5 / 1.0 / 0.5 on frames 9–11.
- **Run time**: 1 h 12 min training (stopped by patience at 29,000, best at 24,000, live weights), then 4 min predict,
  tune, score.

## 3. Reading

- The tuned onset thresholds go up (0.2–0.3, from 0.1–0.2): the wider targets make the onset head more confident.
- Six dragon gains on every instrument, solo on all but 가야금. Fits the six dragon labels' timing spread (±30–50 ms
  around the audio, exp014's error check): a target as wide as the label's own uncertainty stops punishing the model
  for near misses the label can't settle.

## 4. Next

- exp024: this wide triangle + instrument conditioning (exp019).
- A half-width of 3 frames is the next point on the curve, not run.
