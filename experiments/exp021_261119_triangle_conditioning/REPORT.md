# exp021 — triangle onset targets + instrument conditioning together

Run 2026-11-19 on sym8, GPU 0, branch `improv` at `6603484` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp022 and exp023 (exp021 again with seed 43).

> Outputs live in `artifacts/` (→ `storage/experiments/exp021_261119_triangle_conditioning/`), same layout as exp020's.

---

## 1. Verdict

**Best mean of both test sets so far: six dragon test 0.668, solo test 0.630** (exp013 this morning: 0.630, 0.613).
Against the control exp020 it gains +0.008 on each, but both paired 95 % intervals just touch zero
(six dragon −0.000…+0.019, solo −0.002…+0.018): the two changes add up, not much past what each does alone.

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp020 → exp021 | 0.67 → 0.67 | 0.67 → 0.67 | 0.51 → 0.52 | 0.70 → 0.73 | 0.77 → 0.77 | 0.64 → 0.65 | **0.660 → 0.668** |
| solo test, dropped clips out (445), exp020 → exp021 | 0.46 → 0.46 | 0.64 → 0.64 | 0.72 → 0.71 | 0.55 → 0.57 | 0.74 → 0.75 | 0.63 → 0.65 | **0.623 → 0.630** |

## 2. What ran

- **exp020's run** with exp018's triangle onset targets (half-width 1 frame) and exp019's instrument conditioning.
- **Run time**: 1 h 27 min training (stopped by patience at 35,000, best at 30,000, live weights), then 4 min predict,
  tune, score.

## 3. Next

- exp023 repeats this run with seed 43 (how much a seed moves it); exp024 swaps in exp022's wider triangle.
