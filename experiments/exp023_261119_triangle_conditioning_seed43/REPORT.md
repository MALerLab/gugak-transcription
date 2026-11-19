# exp023 — exp021 again with seed 43: how much a seed moves the combined recipe

Run 2026-11-19 on sym8, GPU 2, branch `improv` at `6603484` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp021 and exp022.

> Outputs live in `artifacts/` (→ `storage/experiments/exp023_261119_triangle_conditioning_seed43/`), same layout as exp020's.

---

## 1. Verdict

**The seed barely matters: six dragon test 0.668 again (exp021: 0.668), solo test 0.634 (exp021: 0.630).** With two
seeds agreeing, the combined recipe's gain over the control exp020 holds: here +0.008 six dragon (paired 95 %
interval +0.003…+0.014) and +0.012 solo (+0.004…+0.021).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40): exp020 / exp021 / exp023 | 0.67 / 0.67 / 0.68 | 0.67 / 0.67 / 0.67 | 0.51 / 0.52 / 0.52 | 0.70 / 0.73 / 0.71 | 0.77 / 0.77 / 0.78 | 0.64 / 0.65 / 0.65 | **0.660 / 0.668 / 0.668** |
| solo test, dropped clips out (445): exp020 / exp021 / exp023 | 0.46 / 0.46 / 0.48 | 0.64 / 0.64 / 0.64 | 0.72 / 0.71 / 0.72 | 0.55 / 0.57 / 0.55 | 0.74 / 0.75 / 0.75 | 0.63 / 0.65 / 0.66 | **0.623 / 0.630 / 0.634** |

## 2. What ran

- **exp021's run** with seed 43: different training segments in a different order (the conditioning starts at zero
  either way).
- **Run time**: 1 h 41 min training (ran to the 40,000-step cap, best at 39,000, live weights), then 4 min predict,
  tune, score.

## 3. Reading

- Per instrument the two seeds differ by up to 0.02 (아쟁, 대금), so single-instrument changes under ≈ 0.02 are noise;
  the six-instrument means move under 0.005.
- Still improving at the cap (best at 39,000 of 40,000), unlike seed 42 (stopped at 35,000).
