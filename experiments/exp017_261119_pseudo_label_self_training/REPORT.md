# exp017 — self-training: exp013 transcribes unlabelled 민속악 stems, a new model trains on them too

Run 2026-11-19 on sym8, GPU 2, branch `improv`: pseudo-labels from the code committed in `714ac91`, training chain at `e2d71e7`
(`git_head.txt`, `git_status.txt`). Autonomous session.

> Outputs live in `artifacts/` (→ `storage/experiments/exp017_261119_pseudo_label_self_training/`): `pseudo_midi/` (one
> MIDI per pseudo-labelled stem), `tables/pseudo_labels.parquet`, then the usual `weights/` (`best.pt` = the better of
> `best_live.pt` and `best_ema.pt`), `predictions/`, `note_tables/`, `midi/`, `tables/`, `logs/`, `code/`.

---

## 1. Verdict

**Pseudo-labelled 민속악 stems don't help the 민속악 solo test (0.618 → 0.613) and cost six dragon (0.656 → 0.645).**
The six dragon drop is outside run noise (paired 95 % interval −0.016…−0.006), the solo one isn't (−0.013…+0.003).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp013 → exp017 | 0.66 → 0.64 | 0.67 → 0.66 | 0.51 → 0.50 | 0.70 → 0.67 | 0.77 → 0.76 | 0.63 → 0.63 | **0.656 → 0.645** |
| solo test, dropped clips out (445), exp013 → exp017 | 0.46 → 0.47 | 0.62 → 0.63 | 0.71 → 0.71 | 0.55 → 0.54 | 0.73 → 0.73 | 0.64 → 0.61 | **0.618 → 0.613** |

exp013's row = exp014 (exp013's weights, peak decoding, thresholds retuned).

## 2. What ran

- **Pseudo-labels** (`pseudo_label_stems`): exp013's weights with exp014's decoding transcribed strand B's 214 train-split
  stems of the separation dataset (산조 + 대풍류, the six instruments, 13.2 h; exp002's `picked_stems.parquet`). 78,085
  notes, 1.1–2.5 per second by instrument, close to the solo test's 민속악 reference (1.4–2.7). Val and test songs untouched.
- **Training**: exp013's recipe on six dragon + solo + the pseudo-labelled stems, mixed by audio time (pseudo 27 %,
  six dragon 38 %, solo 36 %; exp013: six dragon 51 %), hard labels, no augmentation. Validated on six dragon val + solo
  val with peak decoding at 0.15, in padded batches; a moving average of the weights (decay 0.999) validated next to them.
- **Run time**: 1 h 37 min training (stopped by patience at 36,000, best at 31,000, moving-average weights), then
  4 min predict, tune, score.

## 3. Reading

- The teacher's own misses on 민속악 (recall ≈ 0.6 on the solo test) become "no note here" labels, so the student learns
  the teacher's blind spots rather than past them.
- Six dragon's share of training drops from 51 % to 38 %; exp012 → exp013 already showed six dragon test tracks that share.
- The moving average changes nothing here: best val 0.7027 averaged vs 0.7025 live.

## 4. Checks

1. **Combined smoke run** (6 items per dataset, 200 steps) with pseudo data, conditioning, augmentation and the moving
   average all on: chain ran, item counts and hours as expected.
2. **Padded-batch validation = one-at-a-time validation**: exp013's best weights score val 0.6931 batched vs 0.6938 one at
   a time (six dragon identical to 4 decimals; solo −0.0014 from the last frames of short clips), 15 s instead of 77 s.
3. Test touched once, after weights and thresholds were picked on val.

## 5. Next

- Not worth repeating as is. A version that might work: keep only the teacher's confident notes and leave uncertain
  stretches out of the loss (like the grace windows), and keep six dragon at half of training.
