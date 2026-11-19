# exp025 — wider still: triangle onset targets over ±3 frames

Run 2026-11-19 on sym8, GPU 0, branch `improv` at `6603484` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp024. The last run of the session.

> Outputs live in `artifacts/` (→ `storage/experiments/exp025_261119_wider_triangle_onset_targets/`), same layout as exp020's.

---

## 1. Verdict

**Best test scores of the day, by a hair: six dragon 0.671, solo 0.632** (mean 0.652; exp021–exp024 0.649–0.651).
Against the control exp020: six dragon +0.011 (paired 95 % interval +0.005…+0.017), solo +0.010 (−0.001…+0.020).
Val disagrees (tuned 0.722, below exp022's 0.725), so read 1, 2 and 3 frames as one plateau, not a trend.

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | half-width (frames) | six dragon test (40) | solo test, dropped clips out (445) | val (tuned) |
|---|---|---|---|---|
| exp020 control (0/1 target) | — | 0.660 | 0.623 | 0.717 |
| exp018 | 1 | 0.661 | 0.630 | 0.719 |
| exp022 | 2 | 0.668 | 0.630 | 0.725 |
| exp025 | 3 | **0.671** | **0.632** | 0.722 |

| per instrument, exp020 → exp025 | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 |
|---|---|---|---|---|---|---|
| six dragon test | 0.67 → 0.68 | 0.67 → 0.68 | 0.51 → 0.52 | 0.70 → 0.73 | 0.77 → 0.77 | 0.64 → 0.65 |
| solo test | 0.46 → 0.48 | 0.64 → 0.64 | 0.72 → 0.70 | 0.55 → 0.56 | 0.74 → 0.72 | 0.63 → 0.68 |

## 2. What ran

- **exp022's run** with half-width 3 frames (`onset_label_half_width_frames: 3.0`): a start on a frame gives
  0.33 / 0.67 / 1.0 / 0.67 / 0.33 over 5 frames.
- **Run time**: 1 h 40 min training (ran to the 40,000-step cap, best at 39,000, live weights), then 4 min predict,
  tune, score. It learns slower than the narrower targets and was still creeping up at the cap.
- Tuned onset thresholds rise again (0.2–0.3); val during training decoded at a fixed 0.15, which undersells wide targets.

## 3. Next

- If a run continues from here: wide targets (2–3 frames) with the validation threshold raised to ≈ 0.25, and a higher
  step cap for half-width 3. Expect ≤ 0.01; the bigger levers are a 민속악 val set and six dragon label timing.
