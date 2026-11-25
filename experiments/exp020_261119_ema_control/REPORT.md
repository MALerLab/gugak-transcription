# exp020 — control: exp013's recipe with today's validation set-up and a moving average of the weights

Run 2026-11-19 on sym8, GPU 2, branch `improv` at `b92f76d` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp018 and exp019.

> Outputs live in `artifacts/` (→ `storage/experiments/exp020_261119_ema_control/`): `weights/` (`best.pt` = the better
> of `best_live.pt` and `best_ema.pt`), `predictions/`, `note_tables/`, `midi/`, `tables/`, `logs/`, `code/`.

---

## 1. Verdict

**The control for exp018 and exp019.** Same data, seed and training as exp013; only the validation set-up differs
(peak decoding at onset threshold 0.15, every 1,000 steps, padded batches) plus a moving average of the weights
validated next to them. It lands at six dragon test 0.660 and solo test 0.623, slightly above exp013 rescored with peak
decoding (exp014: 0.656, 0.618).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp014 → exp020 | 0.66 → 0.67 | 0.67 → 0.67 | 0.51 → 0.51 | 0.70 → 0.70 | 0.77 → 0.77 | 0.63 → 0.64 | **0.656 → 0.660** |
| solo test, dropped clips out (445), exp014 → exp020 | 0.46 → 0.46 | 0.62 → 0.64 | 0.71 → 0.72 | 0.55 → 0.55 | 0.73 → 0.74 | 0.64 → 0.63 | **0.618 → 0.623** |

## 2. Run noise

- exp014 → exp020 is the same training up to GPU non-determinism, picked at a different step (33,000 vs 23,500).
  The gap, +0.004 six dragon (paired 95 % interval +0.001…+0.007) and +0.005 solo (−0.002…+0.011), is the size of
  change one rerun makes. Changes smaller than ≈ 0.005 today are not real.
- **The moving average does nothing**: best val 0.7145 live vs 0.7140 averaged, here and within ±0.001 in exp017–exp019.

## 3. What ran

- exp013's recipe; validation as above; moving average decay 0.999 (`ema_decay`).
- **Run time**: 1 h 35 min training (stopped by patience at 38,000, best at 33,000, live weights), then 4 min predict,
  tune, score.
