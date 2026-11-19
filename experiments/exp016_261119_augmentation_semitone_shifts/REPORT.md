# exp016 — exp015's augmentation plus whole-semitone shifts

Run 2026-11-19 on sym8, GPU 1, branch `improv` at `56045e1` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp015.

> Outputs live in `artifacts/` (→ `storage/experiments/exp016_261119_augmentation_semitone_shifts/`), same layout as exp015's.

---

## 1. Verdict

**Semitone shifts cost six dragon: test 0.656 → 0.637 (paired 95 % interval −0.030…−0.010).** Solo test gains a
little, 0.618 → 0.625 (−0.003…+0.017, not outside run noise), almost all of it on 대금 (0.46 → 0.51).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp013 → exp016 | 0.66 → 0.65 | 0.67 → 0.64 | 0.51 → 0.49 | 0.70 → 0.65 | 0.77 → 0.76 | 0.63 → 0.62 | **0.656 → 0.637** |
| solo test, dropped clips out (445), exp013 → exp016 | 0.46 → 0.51 | 0.62 → 0.64 | 0.71 → 0.72 | 0.55 → 0.54 | 0.73 → 0.72 | 0.64 → 0.62 | **0.618 → 0.625** |

exp013's row = exp014 (exp013's weights, peak decoding, thresholds retuned).

## 2. What ran

- **exp015's run** with one addition: half the segments also shifted by ±1 or ±2 semitones, labels moved by the same
  number of keys.
- **Run time**: 1 h 48 min training (stopped by patience at 32,000, best at 27,000), then 4 min predict, tune, score.

## 3. Why six dragon drops (guess, not checked)

- Six dragon pieces sit in a few fixed modes, and its train and test songs share them. Shifting segments by semitones
  teaches the model to expect notes off those modes, which it can no longer use as a prior.
- 민속악 solo test is in other modes than most training audio, so there the lost prior costs nothing and 대금 gains.

## 4. Next

- Semitone shifts stay off. If 민속악 matters more than 정악 later, a shift only on solo segments is the variant to try.
