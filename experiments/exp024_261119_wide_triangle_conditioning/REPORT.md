# exp024 — wide triangle onset targets + instrument conditioning

Run 2026-11-19 on sym8, GPU 1, branch `improv` at `6603484` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp025.

> Outputs live in `artifacts/` (→ `storage/experiments/exp024_261119_wide_triangle_conditioning/`), same layout as exp020's.

---

## 1. Verdict

**Same place as exp021–exp023: six dragon test 0.666, solo test 0.632.** The wide triangle (exp022) and conditioning
(exp019) don't stack past either one: every combination of today's training changes lands at ≈ 0.667 / 0.630.
Best tuned val of all runs (0.728), but val no longer separates these runs from each other.

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | six dragon test (40) | solo test, dropped clips out (445) | val (tuned) |
|---|---|---|---|
| exp020 control | 0.660 | 0.623 | 0.717 |
| exp021 triangle + conditioning (seed 42 / exp023 seed 43) | 0.668 / 0.668 | 0.630 / 0.634 | 0.723 / 0.725 |
| exp022 wide triangle | 0.668 | 0.630 | 0.725 |
| exp024 wide triangle + conditioning | 0.666 | 0.632 | 0.728 |

Against exp020: six dragon +0.007 (paired 95 % interval +0.001…+0.012), solo +0.009 (−0.002…+0.020).

## 2. What ran

- **exp021's run** with exp022's wide triangle (half-width 2 frames) instead of the narrow one.
- **Run time**: 1 h 9 min training (stopped by patience at 28,000, best at 23,000, moving-average weights), then 4 min
  predict, tune, score.
