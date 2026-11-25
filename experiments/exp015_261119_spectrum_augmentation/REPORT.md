# exp015 — exp013's fine-tune with spectrum augmentation (tuning, gain, tilt)

Run 2026-11-19 on sym8, GPU 0, branch `improv` at `56045e1` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp016.

> Outputs live in `artifacts/` (→ `storage/experiments/exp015_261119_spectrum_augmentation/`): `weights/`, `predictions/`,
> `note_tables/`, `midi/`, `tables/` (training + validation logs, `threshold_sweep.parquet`, `onset_thresholds.parquet`,
> `scores.parquet`), `logs/`, `code/` (the `src/` snapshot the chain ran), `smoke/`.

---

## 1. Verdict

**Augmentation doesn't help: six dragon test 0.656 → 0.653, solo test 0.618 → 0.609.** Neither change is outside run
noise (paired 95 % intervals −0.008…+0.002 and −0.021…+0.001). Val agrees: 0.711 vs exp013's 0.713.

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp013 → exp015 | 0.66 → 0.66 | 0.67 → 0.66 | 0.51 → 0.51 | 0.70 → 0.68 | 0.77 → 0.77 | 0.63 → 0.63 | **0.656 → 0.653** |
| solo test, dropped clips out (445), exp013 → exp015 | 0.46 → 0.44 | 0.62 → 0.64 | 0.71 → 0.72 | 0.55 → 0.55 | 0.73 → 0.74 | 0.64 → 0.56 | **0.618 → 0.609** |

exp013's row = exp014 (exp013's weights, peak decoding, thresholds retuned).

## 2. What ran

- **exp013's recipe** (MAESTRO start, solo + six dragon mixed by audio time, lr 1e-4 halved every 10,000 steps, seed 42),
  validated with peak decoding at onset threshold 0.15 every 1,000 steps (patience 5).
- **Augmentation per segment, on the magnitude spectrum** (`augmentation` in the training recipe): pitch moved by a
  uniform ±30 cents with labels unchanged · gain ±6 dB · spectral tilt ±3 dB per octave around 1 kHz.
- **Run time**: 2 h 0 min training (stopped by patience at 35,000, best at 30,000), then 4 min predict, tune, score.

## 3. Checks

1. **Augmentation by hand**: a 440 Hz tone moved by +100 / −1200 cents peaks at 469 / 219 Hz (bin width 7.8 Hz); label
   shift and drop at the key range edge correct.
2. **Augmentation off reproduces exp013**: exp013's smoke config on the new code gives val 0.4393 / 0.4753 / 0.5114 at
   steps 100–300 vs exp013's 0.4386 / 0.4707 / 0.5116 (GPU non-determinism only).
3. Test touched once, after weights and thresholds were picked on val.

## 4. Next

- Solo 거문고 drops most (0.64 → 0.56) and gets a lower threshold (0.1 vs 0.2). Threshold picks on val wobble by
  instrument; that alone moves a solo instrument score by ±0.05.
- Augmentation stays off from exp017 on.
