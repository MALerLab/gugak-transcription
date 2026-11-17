# exp009 — Onsets and Frames fine-tuned longer, with learning-rate decay and weights picked at onset threshold 0.2

Run 2026-11-15 on sym8, GPU 0, branch `main` at `b9789c5` plus uncommitted changes (`git_head.txt`, `git_status.txt`; the uncommitted code is saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`). Autonomous session, straight after exp008. Nothing committed.

> Outputs live in `artifacts/` (→ `storage/experiments/exp009_261115_onsets_frames_longer/`): `weights/` (`best.pt`, `last.pt`), `predictions/`, `note_tables/`, `midi/`, `tables/` (training + validation logs, `threshold_sweep.parquet`, `onset_thresholds.parquet`, `scores.parquet`), `logs/`.

---

## 1. Verdict

**Training longer gains nothing: six dragon test onset F1 0.655, against exp008's 0.657.** Val plateaued at ≈ 0.64 by step 20,000 and the run stopped itself at step 36,000 (best 0.646 at step 31,000). exp007's 20,000 steps were already enough; it only looked unfinished because exp007 validated at the wrong threshold (0.5). Solo is unchanged too (0.231 vs 0.234).

| onset F1, grace windows left out, tuned thresholds | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40 stems), exp008 → exp009 | 0.68 → 0.68 | 0.67 → 0.68 | 0.51 → 0.50 | 0.67 → 0.65 | 0.77 → 0.78 | 0.65 → 0.65 | **0.657 → 0.655** |
| six dragon val (59 stems) | 0.69 → 0.70 | 0.67 → 0.67 | 0.57 → 0.56 | 0.54 → 0.58 | 0.77 → 0.77 | 0.61 → 0.60 | **0.642 → 0.646** |
| solo test (581 clips) | 0.18 → 0.17 | 0.22 → 0.21 | 0.39 → 0.42 | 0.25 → 0.24 | 0.37 → 0.35 | 0.00 → 0.00 | **0.234 → 0.231** |

---

## 2. What ran

- **Same as exp007** (start from exp004's MAESTRO weights, six dragon MIDI on the 242 train stems, grace windows left out of the loss, seed 42), except:
  - up to **40,000 steps** (exp007: 20,000)
  - **learning rate 1e-4, halved every 10,000 steps** (exp007: constant)
  - **val decoded at onset threshold 0.2** to pick the weights (exp007: 0.5), the threshold exp008 found
  - **patience 10** validations (exp007: 6)
- **Then**: predict val, test and solo with `best.pt`, tune per-instrument onset thresholds on val (exp008's command), score test and solo once.
- **Run time**: 131 min training (stopped by patience at step 36,000), then a few minutes for predict, tune and score.

---

## 3. Scores

| six dragon test | exp008 (exp007's weights, tuned) | **exp009** |
|---|---|---|
| onset F1 | 0.657 | **0.655** |
| precision / recall | 0.70 / 0.62 | 0.70 / 0.62 |
| notes per reference note | 0.87 | 0.87 |
| share of notes under 100 ms | 5.1 % | 5.4 % |
| onset F1, every note scored (no windows) | 0.613 | 0.612 |

- **Val curve (threshold 0.2)**: 0.21 at step 0, 0.58 at 2,000, 0.62 at 8,000, 0.64 at 20,000, then flat within ±0.003 to the end. Training loss kept falling, val didn't follow.
- **Tuned threshold is 0.2 again** for every instrument, and the val curve around it has the same shape as exp008's.
- **Per instrument, every change is within ±0.04** on val and ±0.015 on test: noise level, no pattern.

---

## 4. Checks

1. **Chain ran end to end** (`logs/chain_exit.txt` = 0): train → predict → tune → score, each reading only its run config in `configs/`.
2. **Test never touched while picking** the weights or the thresholds.

---

## 5. Calls I made alone

1. **Learning rate halved every 10,000 steps**: a simple decay so the tail of training takes small steps. Not tuned.
2. **Patience 10**: stopping after 5,000 steps with no val gain.
3. **Stopped the follow-ups here.** The agreed rule was to stop once a follow-up gains under 0.01 onset F1; exp009 gains −0.002.
4. **No scale-quantisation follow-up.** It was meant to clean up blips; on six dragon only ≈ 5 % of written notes are under 100 ms, so there's little left to clean.

---

## 6. Files changed

- None beyond exp007's and exp008's: exp009 only sets the decay and threshold settings those commands already read.

---

## 7. Next

- **Six dragon is at a plateau for this recipe** (≈ 0.65 test). Further gains need a change in data or model, not more steps: Jae's call.
- **Solo** still needs 민속악 in training, or a look at the solo MIDI's pitch naming on 거문고 (exp008, section 3).
