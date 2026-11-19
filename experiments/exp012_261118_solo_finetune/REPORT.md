# exp012 — Onsets and Frames fine-tuned on solo clips only

Run 2026-11-18 on sym8, GPU 0, branch `main` at `340f852` plus uncommitted changes (`git_head.txt`, `git_status.txt`; the uncommitted code is saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`, the launch scripts as `launch_scripts.tgz`). Autonomous session, in parallel with exp013. Nothing committed.

> Outputs live in `artifacts/` (→ `storage/experiments/exp012_261118_solo_finetune/`): `tables/solo_split.parquet` (the solo split, shared with exp013), `weights/` (`best.pt`, `last.pt`), `predictions/`, `note_tables/`, `midi/`, `tables/` (training + validation logs, `threshold_sweep.parquet`, `onset_thresholds.parquet`, `scores.parquet`), `logs/`, `smoke/`.

---

## 1. Verdict

**Solo audio teaches a lot on its own: solo test onset F1 0.244 → 0.628, and solo 거문고 is fixed (0.00 → 0.59, notes now on the right key).** It costs six dragon: six dragon test drops 0.655 → 0.489. So each dataset mostly teaches its own audio; exp013 (both mixed) is the run to read for the main bet.

| onset F1, grace windows left out, tuned thresholds | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40 stems), exp009 → exp012 | 0.68 → 0.56 | 0.68 → 0.49 | 0.50 → 0.44 | 0.65 → 0.32 | 0.78 → 0.67 | 0.65 → 0.45 | **0.655 → 0.489** |
| solo test, dropped clips out (445 clips), exp009 → exp012 | 0.18 → 0.46 | 0.23 → 0.62 | 0.45 → 0.78 | 0.24 → 0.56 | 0.36 → 0.75 | 0.00 → 0.59 | **0.244 → 0.628** |

---

## 2. What ran

- **Same as exp009** (start from exp004's MAESTRO weights, lr 1e-4 halved every 10,000 steps, up to 40,000 steps, patience 10, val decoded at onset threshold 0.2, seed 42), except:
  - **trained on solo clips only**: the 4,048 train clips of the new solo split (17.9 h), no six dragon audio
  - **weights picked on solo val** (522 clips, 2.5 h)
  - **per-instrument onset thresholds tuned on solo val** too
- **Solo split** (`split_solo_clips`, table in `artifacts/tables/solo_split.parquet`), from the 5,557 solo clips of the six instruments (every genre; the same list as exp010's `items.parquet`):
  - test = exp002's 588 test clips, unchanged
  - every other clip split by piece, seed 42: train 4,048 clips (17.9 h, 355 pieces), val 522 clips (2.5 h, 53 pieces)
  - exp011's dropped clips (exp010's audit) stay in the table with a `drop_reason`; training and val skip them (399 non-test clips)
  - no piece is in two splits; no non-test clip shares a piece with test
- **Scored on**: six dragon test (40 stems) and solo test minus the dropped clips (445 clips, the same set as exp011's baseline), today's onset F1 and exp011's tuning-gated onset F1.
- **Run time**: 46 min training (stopped by patience at step 16,000, best at step 11,000), then 3 min for predict, tune and score.

---

## 3. Scores

Onset F1 per instrument, grace windows left out, mean of instrument means in **all**. exp009's row = exp011's rescored table (exp009's predictions).

| test set | run | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|---|
| six dragon test (40), today | exp009 | 0.68 | 0.68 | 0.50 | 0.65 | 0.78 | 0.65 | **0.655** |
| | exp012 | 0.56 | 0.49 | 0.44 | 0.32 | 0.67 | 0.45 | **0.489** |
| six dragon test (40), tuning-gated | exp009 | 0.68 | 0.68 | 0.50 | 0.65 | 0.78 | 0.65 | **0.655** |
| | exp012 | 0.56 | 0.49 | 0.44 | 0.32 | 0.67 | 0.45 | **0.489** |
| solo test, dropped out (445), today | exp009 | 0.18 | 0.23 | 0.45 | 0.24 | 0.36 | 0.00 | **0.244** |
| | exp012 | 0.46 | 0.62 | 0.78 | 0.56 | 0.75 | 0.59 | **0.628** |
| solo test, dropped out (445), tuning-gated | exp009 | 0.18 | 0.23 | 0.45 | 0.24 | 0.36 | 0.06 | **0.254** |
| | exp012 | 0.46 | 0.62 | 0.78 | 0.56 | 0.75 | 0.61 | **0.630** |

- **Solo gains everywhere**, +0.28 to +0.59 per instrument. exp009 barely wrote notes on solo audio (0.34 notes per reference note, recall 0.18); exp012 writes 0.89, precision 0.67 / recall 0.60.
- **Six dragon loses everywhere**, worst on 아쟁 (−0.33) and 피리 (−0.19). exp012 over-writes on six dragon: 1.10 notes per reference note (exp009 0.77), precision 0.70 → 0.45.
- **The tuning gate still barely moves anything** (+0.002 on solo, 0 on six dragon). Without exp009's one-low 거문고 there's almost nothing left for it to catch.
- **Val curve (solo val, threshold 0.2)**: 0.38 at step 0, 0.66 at 500, 0.74 at 2,000, 0.786 at 11,000, then flat. Tuned threshold is 0.2 for every instrument again.

---

## 4. Solo 거문고

**Fixed.** exp009 wrote every solo 거문고 note one semitone low (0.00 as written, 0.18 shifted up). exp012 writes them on the right key:

| solo 거문고 test (36 clips, every clip) | F1 at −1 | as written | F1 at +1 |
|---|---|---|---|
| exp009 | 0.00 | 0.00 | 0.18 |
| exp012 | 0.02 | **0.56** | 0.16 |

- "As written" is the best shift on 31 of 36 clips (exp009: none).
- So the one-low error came from learning 거문고 on six dragon audio only, not from the solo audio or MIDI. Cause inside the model still not checked.

---

## 5. Solo val ≠ solo test: different genres

Solo val scores 0.786, solo test 0.628. The gap is mostly genre, not overfitting:
- **Solo test is 100 % 민속악** (exp002's pick: 민속 풍류, 산조, 시나위 = genre codes F01–F03).
- **Solo train is 98 % other genres**: only 122 of 4,048 train clips (0.33 h) are F01–F03. exp002's pick had already taken every F01–F03 clip, and 588 of its 738 went to test.
- **Solo val has no 민속악 at all.** The weights and thresholds were picked on genres the test doesn't contain.
- The brief expected ≈ 18 h "of 민속악 and other genres"; in practice the 민속악 share of training is tiny.

---

## 6. Checks

1. **Smoke run first** (300 steps, then predict + tune + score on 6 items per answer key × split; `artifacts/smoke/`): chain ran end to end, val rose 0.38 → 0.62.
2. **Chain ran end to end** (`logs/chain_exit.txt` = 0): train → predict → tune → score, each reading only its run config in `configs/`.
3. **Test never touched while picking** the weights or the thresholds (solo val only).
4. **Solo split invariants**: test = exp002's 588 clip ids exactly; 0 pieces in two splits; train 4,048 clips / 17.9 h, as the brief expected (≈ 4,100 / ≈ 18 h).
5. **Solo test set = exp011's baseline set**: 581 clips minus 136 dropped = 445, and exp009's rescored numbers (0.244 / 0.254) are read from exp011's table.
6. **exp009's run configs still work on the new code**: the six dragon train list from exp009's finetune config is identical (242 stems), and its predict config lists the same 40 / 59 / 581 items.

### Skipped

- **wandb**: there's no `.env` in the main checkout, and nothing in the training chain logs to wandb anyway (exp007–exp009 didn't either). Logs are in `artifacts/logs/` and `tables/`.

---

## 7. Calls I made alone

1. **Val pieces picked per instrument.** A plain 10 % of pieces gave 아쟁 only 4 val clips, too few for a mean of instrument means. Instead: pieces in a seeded order, a piece joins val if one of its instruments is under 10 % of its kept clips and none goes past 15 %. Result: 10–13 % per instrument, still split by piece.
2. **Short clips padded, not dropped.** The training segment is 20.5 s (exp009's), and 2,817 of 4,048 solo train clips are shorter. Those are padded with silence to one segment and the padding is left out of the loss. Each clip is drawn with weight = its real length, so sampling stays in proportion to audio time. (The model's batch norm still sees the padded silence.)
3. **Dropped non-test clips stay in the split table** with their `drop_reason`, so the table lists every clip; readers skip them.
4. **Solo test items come from exp002's table as before** (581 minus 7 broken references, then minus the dropped clips at scoring), not from the new split table, so the solo test set is exactly exp011's 445.
5. **Six dragon val and test were still predicted and scored**, to show the six dragon cost.

---

## 8. Files changed

- `src/datasets/solo.py`: `split_solo_clips` (the solo training split).
- `src/datasets/answer_keys.py`: `solo_split_pairs` (solo clips of given splits, dropped ones skipped).
- `src/pipelines/split_solo_clips.py` (new command) + `configs/pipelines/split_solo_clips.yaml`.
- `src/pipelines/common.py`: `list_training_items` (six dragon and / or solo); `list_answer_key_items` can add solo clips from a split table.
- `src/pipelines/finetune_onsets_frames.py`: `run.train_datasets`, `run.validation_datasets`, `run.solo_split.table`; selection = mean over datasets of each one's instrument mean; per-dataset val columns in the log. Defaults = six dragon only, so exp007–exp009's run configs run unchanged.
- `src/transcribe/onsets_frames_training.py`: `SegmentSampler` pads items shorter than a segment instead of dropping them (no six dragon stem is that short, so six dragon sampling is unchanged).
- `src/pipelines/tune_onsets_frames_thresholds.py`: `run.tuning_datasets` (default six dragon).
- `src/pipelines/predict_onsets_frames.py`: smoke limit per answer key × split.
- Configs: `configs/datasets/solo.yaml` (`training_split`), `configs/pipelines/{finetune_onsets_frames,predict_onsets_frames,tune_onsets_frames_thresholds}.yaml`.
- Untracked: `scripts/exp012/` (`make_configs.sh`, `chain.sh`, `compare.py`).

---

## 9. Next

- **exp013 (solo + six dragon) answers the mixing question**: solo test 0.613, six dragon test 0.630, so mixing keeps most of both. Read its report.
- **민속악 is barely in solo training** (0.33 h). If 민속악 is the target, the split needs a rethink: e.g. move some of exp002's 민속악 test pieces into train, or build a 민속악 val set. Jae's call, since the test set was fixed on purpose.
- **Solo 거문고 is fixed by solo audio**: the todo line on "solo 거문고 one note low" can close, and the scale-quantisation idea loses its main use case.
