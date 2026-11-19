# exp013 — Onsets and Frames fine-tuned on solo clips + six dragon stems

Run 2026-11-18 on sym8, GPU 1, branch `main` at `340f852` plus uncommitted changes (`git_head.txt`, `git_status.txt`; the uncommitted code is saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`, the launch scripts as `launch_scripts.tgz`). Autonomous session, in parallel with exp012. Nothing committed.

> Outputs live in `artifacts/` (→ `storage/experiments/exp013_261118_solo_six_dragon_finetune/`): `weights/` (`best.pt`, `last.pt`), `predictions/`, `note_tables/`, `midi/`, `tables/` (training + validation logs, `threshold_sweep.parquet`, `onset_thresholds.parquet`, `scores.parquet`), `logs/`, `smoke/`. The solo split table is exp012's (`exp012_261118_solo_finetune/artifacts/tables/solo_split.parquet`).

---

## 1. Verdict

**Mixing mostly works: one model gets close to the best of both.** Solo test onset F1 0.244 → 0.613 (exp012 alone: 0.628), and six dragon test only drops 0.655 → 0.630 (exp012 alone: 0.489). Solo 거문고 is fixed here too (0.00 → 0.64, right key on all 36 clips). The cost: −0.025 on six dragon test, spread over every instrument.

| onset F1, grace windows left out, tuned thresholds | exp009 (six dragon) | exp012 (solo) | exp013 (both) |
|---|---|---|---|
| six dragon test (40 stems) | **0.655** | 0.489 | 0.630 |
| solo test, dropped clips out (445 clips) | 0.244 | **0.628** | 0.613 |

---

## 2. What ran

- **Same as exp009** (start from exp004's MAESTRO weights, lr 1e-4 halved every 10,000 steps, up to 40,000 steps, patience 10, val decoded at onset threshold 0.2, seed 42), except:
  - **trained on solo + six dragon**: the 4,048 solo train clips (17.9 h) + exp009's 242 six dragon train stems (18.8 h), drawn by audio time, so about half of each
  - **weights picked on the mean of six dragon val (59 stems) and solo val (522 clips)**, each one a mean of instrument means
  - **per-instrument onset thresholds tuned on the same mean**
- **Solo split**: exp012's (§2 of its report). Same train / val clips, same dropped clips left out.
- **Scored on**: six dragon test (40 stems) and solo test minus the dropped clips (445 clips, exp011's set), today's onset F1 and exp011's tuning-gated onset F1.
- **Run time**: 2 h 17 min training (stopped by patience at step 28,500, best at step 23,500), then 3 min for predict, tune and score.

---

## 3. Scores

Onset F1 per instrument, grace windows left out, mean of instrument means in **all**. exp009's rows = exp011's rescored table (exp009's predictions). Tuning-gated is identical to today's on six dragon test for all three runs, so only today's is shown there.

| test set | run | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|---|
| six dragon test (40) | exp009 | 0.68 | 0.68 | 0.50 | 0.65 | 0.78 | 0.65 | **0.655** |
| | exp012 | 0.56 | 0.49 | 0.44 | 0.32 | 0.67 | 0.45 | **0.489** |
| | exp013 | 0.64 | 0.65 | 0.47 | 0.63 | 0.76 | 0.62 | **0.630** |
| solo test, dropped out (445), today | exp009 | 0.18 | 0.23 | 0.45 | 0.24 | 0.36 | 0.00 | **0.244** |
| | exp012 | 0.46 | 0.62 | 0.78 | 0.56 | 0.75 | 0.59 | **0.628** |
| | exp013 | 0.46 | 0.60 | 0.73 | 0.53 | 0.73 | 0.64 | **0.613** |
| solo test, dropped out (445), tuning-gated | exp009 | 0.18 | 0.23 | 0.45 | 0.24 | 0.36 | 0.06 | **0.254** |
| | exp012 | 0.46 | 0.62 | 0.78 | 0.56 | 0.75 | 0.61 | **0.630** |
| | exp013 | 0.46 | 0.60 | 0.73 | 0.53 | 0.73 | 0.64 | **0.613** |

- **Six dragon: exp013 loses a little everywhere vs exp009**, −0.01 to −0.04 per instrument (biggest on 대금 −0.04 and 피리 −0.03). Precision 0.70 → 0.69, recall 0.62 → 0.59, notes per reference note 0.77 → 0.74. So it writes slightly fewer notes, not wrong ones. exp012's over-writing (1.10 notes per reference note) is gone.
- **Solo: exp013 sits just under exp012** (−0.015), except 거문고, where it's better (0.64 vs 0.59). Precision 0.68, recall 0.58, notes per reference note 0.84.
- **The tuning gate moves nothing for exp013** (0 on both test sets). With 거문고 on the right key there's nothing left for it to catch.
- **Val curve (mean of six dragon val and solo val, threshold 0.2)**: 0.29 at step 0, 0.61 at 1,000, 0.68 at 10,000, 0.694 at 23,500. Still creeping up after each lr halving, unlike exp012, which was flat from 11,000. Tuned threshold is 0.2 for every instrument again.

---

## 4. 가야금 and 거문고 on six dragon test

The solo plucked MIDI may start 20–30 ms late, so these two could have learned late onsets from it. They didn't lose more than the others:

| six dragon test | 가야금 | 거문고 | other four, mean |
|---|---|---|---|
| exp009 | 0.776 | 0.646 | 0.627 |
| exp013 | 0.764 | 0.623 | 0.598 |
| change | −0.012 | −0.023 | −0.028 |

- So no sign of a late-onset problem at the 50 ms onset tolerance. A finer check (onset error histograms) wasn't run.

---

## 5. Solo 거문고

**Fixed, and better than exp012.** exp013 writes every solo 거문고 clip on the right key:

| solo 거문고 test (36 clips, every clip) | F1 at −1 | as written | F1 at +1 | best shift = as written |
|---|---|---|---|---|
| exp009 | 0.00 | 0.00 | 0.18 | 0 of 36 |
| exp012 | 0.02 | 0.56 | 0.16 | 31 of 36 |
| exp013 | 0.02 | **0.63** | 0.09 | **36 of 36** |

- Adding solo 거문고 audio fixed it in both runs, so the one-low error came from learning 거문고 on six dragon audio only. The cause inside the model is still not checked.
- six dragon 거문고 stays on the right key too (0.62 on six dragon test).

---

## 6. Solo val ≠ solo test: different genres

Same caveat as exp012 (§5 of its report): solo test is 100 % 민속악, solo train only 122 of 4,048 clips (0.33 h) of it, solo val none. exp013's solo val is 0.770 vs solo test 0.613. So the weights and thresholds were again picked partly on genres the solo test doesn't contain.

---

## 7. Checks

1. **Smoke run first** (300 steps, then predict + tune + score on 6 items per answer key × split; `artifacts/smoke/`): chain ran end to end with both datasets in train and val.
2. **Chain ran end to end** (`logs/chain_exit.txt` = 0): train → predict → tune → score, each reading only its run config in `configs/`.
3. **Test never touched while picking** the weights or the thresholds (six dragon val + solo val only).
4. **Both datasets in training, by audio time**: the log lists 4,048 solo + 242 six dragon items, 17.9 h + 18.8 h. No six dragon stem is padded (all 2,817 padded items are solo clips).
5. **Same test sets as exp009 / exp012**: 40 six dragon test stems, 445 solo test clips; exp009's numbers read from exp011's table.
6. **nvidia-smi before launch**: GPUs 0 and 1 free; exp012 on 0, exp013 on 1. All three free again after the runs.

### Skipped

- **wandb**: there's no `.env` in the main checkout, and nothing in the training chain logs to wandb anyway (exp007–exp009 didn't either). Logs are in `artifacts/logs/` and `tables/`.
- **Onset timing check for 가야금 / 거문고** (see §4): only F1 at the 50 ms tolerance was compared, not onset error histograms.

---

## 8. Calls I made alone

1. **Mixed by audio time, not 50 / 50 by item.** The brief said "mixed by audio time"; the two datasets happen to be about the same size (17.9 h vs 18.8 h), so it lands near half and half either way.
2. **Selection = plain mean of the two datasets' scores**, each one already a mean of instrument means. Same rule for the threshold sweep.
3. Everything else as exp012 (§7 of its report): per-instrument val pick, short clips padded, dropped clips kept in the split table, solo test from exp002's table.

---

## 9. Files changed

Same as exp012 (§8 of its report). This run only adds its run configs in `configs/`.

---

## 10. Next

- **exp013's weights are the new default candidate** over exp009: −0.025 on six dragon test buys +0.37 on solo test and a fixed solo 거문고. Jae's call whether that trade is worth it.
- **Close the six dragon gap**: e.g. weight six dragon more than half, or fine-tune exp013's weights a little on six dragon only. Not run.
- **민속악 is barely in solo training** (0.33 h), as in exp012. If 민속악 is the target, the solo split needs a rethink; the test set was fixed on purpose, so Jae's call.
- **Solo 거문고 one note low**: closed by both runs. The scale-quantisation idea loses its main use case.
