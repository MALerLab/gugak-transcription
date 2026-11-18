# exp011 — tuning-gated onset F1: exp009's predictions rescored with a one-semitone tuning allowance

Run 2026-11-18 on sym8, CPU only (no GPU, no training), branch `main` at `e649bce` plus uncommitted changes (`git_head.txt`, `git_status.txt`; code saved as `artifacts/logs/code_at_launch.diff`). Autonomous session. Nothing committed.

> Outputs live in `artifacts/` (→ `storage/experiments/exp011_261118_tuning_gated_scoring/`): `tables/scores.parquet` (one row per item: today's scores, tuning offset, gate result, gated scores, `drop_reason`), `tables/solo_shift_sweep.parquet` (every solo test clip scored at shifts −1 / 0 / +1), `note_tables/`, `midi/`, `logs/`, `smoke/`.

---

## 1. Verdict

**The tuning gate barely moves anything, and it doesn't fix solo 거문고, the case it was built for.** Six dragon test stays at 0.655 (no stem passes the gate). Solo test goes 0.231 → 0.243, the solo baseline (unreliable MIDI dropped) 0.244 → 0.254, and nearly all of that is 거문고 going 0.00 → 0.04–0.06.

**Solo 거문고 isn't a tuning problem.** All 36 solo 거문고 test clips score 0.00 as written and 0.18 on average with every note moved one semitone up, whatever their offset: clips only 15 cents flat fail just like the 28-cent ones. The model writes 거문고 one semitone low on solo audio across the board, and the gate only opens for the 9 clips past 25 cents.

| headline onset F1 (grace windows left out, exp009's thresholds) | today | tuning-gated |
|---|---|---|
| six dragon test (40 stems) | 0.655 | 0.655 |
| solo test (581 clips) | 0.231 | 0.243 |
| solo test, unreliable MIDI dropped (445 clips): the new solo baseline | 0.244 | 0.254 |

---

## 2. What ran

- **Predictions**: exp009's stored Onsets and Frames outputs (`predictions/`), decoded again through `score_onsets_frames` with exp009's per-instrument onset thresholds (0.2 for all), answer keys and grace windows. Run config: `configs/score_onsets_frames.yaml`; the only differences from exp009's are the gate, the solo marks and the output paths.
- **Tuning offset per recording**: `reference_pitch_offset` (exp010) on stock PESTO f0 over the note middles; offset = `semitone_gap` × 100 + `residual_cents`, positive = audio sharp of the reference. f0 reused from cache: six dragon from exp006 (`f0/six_dragon/`), solo from exp010 (`f0/`). Both used the same tracker settings (`mir-1k_g7`, `alwa`, 44.1 kHz, 10 ms hop) and `uv.lock` hasn't changed since 2026-10-30, so nothing was tracked again.
- **Gate**: |offset| > 25 cents (`configs/evaluate/transcription.yaml`, `tuning_gate_cents`) → also score with every note moved one semitone back toward the reference (sharp audio → −1, flat → +1), keep the better `onset_f1_masked`, record which. Never more than one semitone.
- **Solo dropped piles** (`src/datasets/solo.py`, `unreliable_solo_clips`; settings in `configs/datasets/solo.yaml`): from exp010's audit, any non-zero `semitone_gap` or `agreement_after_gap` < 0.5. On the 581 test clips this drops 136: low agreement 87, one semitone off 47, octave or more 2. No test clip has a NaN gap.
- **Run time**: under a minute to score 680 items, then about a minute for the analysis script.

---

## 3. Scores

Onset F1 per instrument, today → tuning-gated (mean of instrument means in **all**).

| test set | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40) | 0.68 → 0.68 | 0.68 → 0.68 | 0.50 → 0.50 | 0.65 → 0.65 | 0.78 → 0.78 | 0.65 → 0.65 | **0.655 → 0.655** |
| solo test (581) | 0.17 → 0.17 | 0.21 → 0.22 | 0.42 → 0.43 | 0.24 → 0.24 | 0.35 → 0.35 | 0.00 → 0.04 | **0.231 → 0.243** |
| solo test, unreliable dropped (445) | 0.18 → 0.18 | 0.23 → 0.23 | 0.45 → 0.45 | 0.24 → 0.24 | 0.36 → 0.36 | 0.00 → 0.06 | **0.244 → 0.254** |

- **Dropping the unreliable MIDI lifts the solo baseline by 0.013** before any gate (0.231 → 0.244). The drop hits 대금 hardest: 65 of 130 test clips go.
- **On the solo baseline, the gate changes only 거문고** (+0.056) and 피리 (+0.002). Without 거문고 the baseline doesn't move.

---

## 4. The gate, per instrument

Recordings past 25 cents, and how many of those scored better shifted.

| test set | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test: passed / better | 0 / 0 of 8 | 0 / 0 of 8 | 0 / 0 of 8 | 0 / 0 of 3 | 0 / 0 of 7 | 0 / 0 of 6 | **0 / 0 of 40** |
| solo test | 27 / 7 of 130 | 35 / 13 of 134 | 4 / 1 of 27 | 19 / 1 of 75 | 4 / 1 of 179 | 9 / 4 of 36 | **98 / 27 of 581** |
| solo test, unreliable dropped | 2 / 0 of 65 | 19 / 3 of 108 | 2 / 0 of 24 | 11 / 0 of 57 | 1 / 0 of 166 | 6 / 3 of 25 | **41 / 6 of 445** |

- **Most gated solo clips are in the dropped piles**: 57 of the 98, incl. all 47 one-semitone-off clips. Those are clips where the MIDI names the far key, and the gate just takes the model's side. Once they're dropped, 6 of 41 gated clips gain.
- **Median offset per instrument (solo test, cents)**: 거문고 −23, 아쟁 −10, 피리 −10, 대금 +4, 가야금 +2, 해금 +10.
- **Six dragon audio sits +10 cents** (median, test; range +2 to +23), a bit sharper than the ≈ +5 the brief expected, but no stem crosses 25. Every six dragon stem has `semitone_gap` 0.

---

## 5. Solo 거문고

Every solo test clip was also scored at −1, 0 and +1 regardless of the gate (`scripts/exp011/analyse_gate.py` → `tables/solo_shift_sweep.parquet`).

| solo 거문고, by offset (cents) | clips | F1 at −1 | as written | F1 at +1 |
|---|---|---|---|---|
| −50 to −25 (gate opens) | 8 | 0.00 | 0.00 | 0.19 |
| −25 to −15 | 24 | 0.00 | 0.00 | 0.17 |
| −15 to 0 | 3 | 0.00 | 0.00 | 0.20 |

- +1 is the best shift on 24 of 36 clips, and as written wins on none. The other 12 are 0 at every shift (ties go to −1 in the table's `best` column).
- So the model writes solo 거문고 one semitone low even when the audio sits only 15 cents flat, where plain rounding would land on the right key. On six dragon 거문고 (audio +6 cents) the same model scores 0.65 as written.
- Even shifted, 0.18 is well below what solo 거문고 got before fine-tuning (exp004 zero-shot 0.21) or from the f0 pipeline (exp002 0.48).
- Cause not checked. Guesses: the fine-tune learned 거문고 pitch from six dragon audio only, all of it sharp, so its key boundaries shifted. Or the low 거문고 register sits where the mel spectrogram is coarsest.
- No other instrument shows this: elsewhere "as written" is the best shift on most clips (e.g. 가야금 173 of 179, 해금 26 of 27).

---

## 6. Checks

1. **Today's onset F1 reproduces exp009 exactly**: six dragon test 0.6545, solo 0.2313, and every per-item score column matches exp009's `scores.parquet` with a max difference of 0.0.
2. **Smoke run first** (2 items per answer key × split × instrument, 36 items; `artifacts/smoke/`): same per-item match. The solo offsets match exp010's `semitone_gap` and `residual_cents` exactly.
3. **Shift direction**: on the 98 gated solo clips, the gate's shift averages F1 0.078 and the opposite shift 0.006, so the direction is right.
4. **The gate isn't too loose on six dragon**: 0 of 40 test stems and 0 of 59 val stems pass, so nothing can move there.
5. **Old run configs still work**: exp009's own `score_onsets_frames.yaml` (outputs pointed at a scratch folder) runs on the new code and gives 0.6545 / 0.6463 / 0.2313 again.

---

## 7. Calls I made alone

1. **Shift direction = back toward the reference's keys** (sharp audio → notes moved down). The brief says "toward the offset's direction". Taken literally that's the other way, which only ever makes scores worse (check 3), so the rule would never fire.
2. **Picked on `onset_f1_masked`**, the headline score. A tie keeps today's. The gated precision, recall and unmasked F1 come from the same pick (`*_tuning_gated` columns).
3. **Offset measured on every reference note**, grace windows included. Grace notes are 80 ms, under the f0 check's 0.15 s minimum, so they never count anyway.
4. **The filter only marks, it doesn't remove**: the scores table keeps all 581 solo clips with a `drop_reason` column (None = kept), so both solo sets come from one run. `drop_unreliable_solo_clips` does the actual dropping for exp012 / exp013's training.
5. **Labelled the dropped piles** (`semitone_off`, `octave_or_more_off`, `other_gap`, `low_agreement`). A gap wins over low agreement when a clip has both.
6. **Re-decoded instead of reading exp009's note tables**: that runs the same command end to end and is what proves the reproduction. Cost: duplicate note tables + MIDI on storage (small).
7. **Scored val too** (the command scores every item). Val isn't reported apart from the six dragon gate check.

---

## 8. Files changed

- `src/evaluate/f0.py`: `recording_tuning_offset`.
- `src/evaluate/transcription.py`: `tuning_gate_shift`, `tuning_gated_scores`.
- `src/datasets/solo.py`: `unreliable_solo_clips`, `drop_unreliable_solo_clips`.
- `src/pipelines/score_onsets_frames.py`: optional gate (`run.tuning_gate_f0_dirs`), solo drop marks (`run.mark_unreliable_solo_clips`), smoke limit (`run.limit_items_per_instrument`), summary per test set. All off by default, so exp009's run config still runs unchanged.
- Configs: `configs/evaluate/transcription.yaml` (`tuning_gate_cents`), `configs/datasets/solo.yaml` (`unreliable_midi`), `configs/pipelines/score_onsets_frames.yaml` (`uses` + run settings).
- Untracked: `scripts/exp011/analyse_gate.py`.

---

## 9. Next

- **Solo 거문고 needs its own look**: why does the fine-tuned model write it one semitone low? Compare exp004's zero-shot output on the same clips (was it right before fine-tuning?). Listen to one clip.
- **New solo baseline for exp012 / exp013**: 0.244 today / 0.254 tuning-gated on the 445 kept test clips.
- **Keep the gate or drop it?** It mostly rewards clips that are dropped anyway. On the kept clips it's a 거문고-only patch. Jae's call.
