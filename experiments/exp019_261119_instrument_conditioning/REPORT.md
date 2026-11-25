# exp019 — instrument conditioning: the model is told which instrument the stem is

Run 2026-11-19 on sym8, GPU 0, branch `improv` at `b92f76d` (`git_head.txt`, `git_status.txt`). Autonomous session,
in parallel with exp018 and exp020 (its control).

> Outputs live in `artifacts/` (→ `storage/experiments/exp019_261119_instrument_conditioning/`), same layout as exp020's.

---

## 1. Verdict

**Conditioning gives the best six dragon test so far (0.664) but neither gain over the control is outside run
noise**: six dragon 0.660 → 0.664 (paired 95 % interval −0.002…+0.012), solo 0.623 → 0.628 (−0.004…+0.014).

| onset F1, grace windows left out, peak decoding, thresholds tuned on val | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 | **all** |
|---|---|---|---|---|---|---|---|
| six dragon test (40), exp020 → exp019 | 0.67 → 0.67 | 0.67 → 0.67 | 0.51 → 0.52 | 0.70 → 0.71 | 0.77 → 0.77 | 0.64 → 0.65 | **0.660 → 0.664** |
| solo test, dropped clips out (445), exp020 → exp019 | 0.46 → 0.47 | 0.64 → 0.64 | 0.72 → 0.71 | 0.55 → 0.55 | 0.74 → 0.74 | 0.63 → 0.66 | **0.623 → 0.628** |

## 2. What ran

- **exp020's run** with the instrument as an input (`conditioning_instruments`): per instrument, a scale and shift on
  each note head's conv features and a per-key bias on the onset and frame outputs. 28,704 new weights, all zero at the
  start, so training begins from the MAESTRO model's exact outputs. The instrument comes from the stem's or clip's label.
- **Run time**: 1 h 32 min training (stopped by patience at 37,000, best at 32,000, live weights), then 4 min predict,
  tune, score.

## 3. What it learned

- The per-key onset bias follows each instrument's range: 거문고 up on the lowest keys (MIDI 32–34) and down on the top
  ones (82–89), 해금 the other way round (up 75–84, down 44–51). Biases stay small (−0.44…+0.05 in logits).

## 4. Checks

1. **Zero start = pretrained model**: conditioned and unconditioned MAESTRO models give identical outputs (max difference 0.0).
2. **One training step moves the conditioning** and leaves the unconditioned path working.
3. Test touched once, after weights and thresholds were picked on val.

## 5. Next

- exp021 combines it with exp018's triangle targets; exp023 repeats exp021 with seed 43.
- A conditioned model needs the instrument at inference: fine for the separation dataset's labelled stems; for the
  separation model's output, the stem's slot gives it.
