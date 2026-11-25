# exp018 — triangle onset targets: each note start split between its two neighbouring frames

Run 2026-11-19 on sym8, GPU 1, branch `improv` at `b92f76d` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp019 and exp020 (its control).

> Outputs live in `artifacts/` (→ `storage/experiments/exp018_261119_triangle_onset_targets/`), same layout as exp020's.

---

## 1. Verdict

**Triangle targets help solo a little and leave six dragon where it was.** Against the control exp020: solo test
0.623 → 0.630 (paired 95 % interval +0.001…+0.013), six dragon test 0.660 → 0.661 (−0.003…+0.004). Best val of all
runs so far (0.719).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp020 → exp018 | 0.67 → 0.67 | 0.67 → 0.67 | 0.51 → 0.51 | 0.70 → 0.70 | 0.77 → 0.77 | 0.64 → 0.64 | **0.660 → 0.661** |
| solo test, dropped clips out (445), exp020 → exp018 | 0.46 → 0.47 | 0.64 → 0.64 | 0.72 → 0.72 | 0.55 → 0.56 | 0.74 → 0.72 | 0.63 → 0.66 | **0.623 → 0.630** |

## 2. What ran

- **exp020's run** with one change: the onset target is no longer a 0/1 at the nearest frame but
  1 − |frame − exact start| (in frames) on the two frames around the start, e.g. 0.75 / 0.25 for a start a quarter of the
  way between them (`onset_label_shape: triangle`). The peak decoding from exp014 can then read the start back between frames.
- **Run time**: 1 h 13 min training (stopped by patience at 30,000, best at 25,000, moving-average weights), then 4 min
  predict, tune, score.

## 3. Checks

1. **Targets by hand**: a start at frame 3.25 gives 0.75 at frame 3 and 0.25 at frame 4; one exactly on frame 10 gives 1.0.
2. Test touched once, after weights and thresholds were picked on val.

## 4. Next

- exp021: triangle targets + instrument conditioning (exp019) together. exp022: a wider triangle (half-width 2 frames).
