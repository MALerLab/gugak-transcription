# exp010 — solo MIDI audit: does each solo clip's MIDI match its own audio?

Run 2026-11-17 on sym8, GPU 0, branch `main` at `5b8c641` plus uncommitted changes (`git_head.txt`, `git_status.txt`; code saved as `artifacts/logs/code_at_launch.diff` + `untracked_code_at_launch.tgz`). Read-only probe: nothing trained, nothing in the dataset changed. Nothing committed.

> Outputs live in `artifacts/` (→ `storage/experiments/exp010_261117_solo_midi_audit/`): `f0/` (stock PESTO track per clip), `tables/` (`clips.parquet`, `items.parquet`, `summary_by_instrument.parquet`), `logs/`, `smoke/`.

---

## 1. Verdict

**The solo MIDI is mostly sound: 94 % of clips (5,248 of 5,557) sit on the right pitch, and timing is on the dot for the sustained instruments.** The problems split into four kinds, each a separate keep / fix / drop call:

| kind | clips | what it looks like |
|---|---|---|
| one semitone off, whole clip | 227 (4.1 %) | the recording sits between two piano keys; the MIDI named the key on the far side. Shape right, label one key out |
| octave or more off | 64 (1.2 %) | mostly 거문고 (34) and 가야금 (18); many gaps are 19 or 28 semitones = the 3rd / 5th harmonic, so the tracker hearing an overtone, not the MIDI being wrong (not listened to) |
| on pitch, but low agreement | 244 (4.4 %) | gap 0, yet written and heard pitch agree < 50 % of the note middles: wrong notes, or tracker trouble |
| plucked MIDI > 50 ms off | 112 (5.1 % of plucked) | 거문고 78, 가야금 34; nearly all late |

Plus 18 odd ones: 14 clips with no note long enough to check (all < 0.15 s or overlapping), 4 거문고 clips with 5–10 semitone gaps. 20 clips' MIDI runs more than 0.5 s past the end of the audio.

---

## 2. What ran

- **Clips**: every solo clip (AI Hub 71470) of the six instruments, every genre: 대금 816, 피리 916, 해금 1,291, 아쟁 353, 가야금 1,356, 거문고 825 = **5,557**. The other 4,388 clips (vocals, percussion, 소금, 단소, 생황, 태평소 …) are out of scope.
- **Tracker**: stock PESTO (`mir-1k_g7`), 10 ms hop, 44.1 kHz.
- **Pitch check** (`src/evaluate/f0.py`, `reference_pitch_offset`): written vs tracked pitch over the middle 20–80 % of each note ≥ 0.15 s, frames with exactly one note sounding.
  - `semitone_gap`: most common whole-semitone gap, tracked − written
  - `residual_cents`: mean of what's left once the gap is out, over frames within half a semitone of it (a mean: PESTO's output clusters on its 1/3-semitone bins, so a median lands on 0 or ±33)
  - `agreement_after_gap`: share of note-middle frames within ±50 cents once the gap is out
- **Timing check** (`src/evaluate/label_timing.py`, exp006's method): the clip's notes, pitch gap taken out, shifted −300…+300 ms in 10 ms steps.
  - pitch judge (every instrument): share of in-note frames on the written pitch
  - onset judge (가야금, 거문고): MIDI note starts vs spectral-flux onsets, ±25 ms
  - negative best shift = the MIDI runs late; "confident" = more than 50 ms off, not at the range edge, and at least 0.05 better than unshifted
- **Run time**: 72 s tracking, 86 s measuring.

---

## 3. Pitch

| instrument | clips | gap 0 | ±1 | octave + | other | median agreement (gap out) |
|---|---|---|---|---|---|---|
| 대금 | 816 | 760 | 53 | 3 | 0 | 0.89 |
| 피리 | 916 | 877 | 38 | 1 | 0 | 0.82 |
| 해금 | 1,291 | 1,236 | 50 | 3 | 2 | 0.80 |
| 아쟁 | 353 | 329 | 17 | 5 | 2 | 0.85 |
| 가야금 | 1,356 | 1,281 | 54 | 18 | 3 | 0.83 |
| 거문고 | 825 | 765 | 15 | 34 | 11 | 0.73 |
| **all** | **5,557** | **5,248** | **227** | **64** | **18** | |

- **±1 = a tuning clash, not wrong notes.** 225 of the 227 have the residual on the opposite side of the gap (e.g. gap +1, residual −28 cents: the recording sits 72 cents above the written key, 28 below the next). Median |residual| 28 cents; 78 % are over 20 cents off (gap-0 clips: 26 %). So the recording's tuning sits between two keys and the MIDI named the far one. +1 (MIDI one key low) is 162 clips, −1 is 65.
- **Agreement after taking the ±1 out is lower** (median 0.57 vs 0.82 for gap 0): the notes sit near the edge of the ±50-cent tolerance even once moved.
- **Octave gaps**: +19 (9 거문고, 3 가야금) and +28 are overtone numbers (3rd, 5th harmonic), the tracker locking onto a low string's overtone. −12 / −24 could be either side. Median agreement 0.47 even with the gap out.
- **Low agreement with gap 0**: 244 clips, 거문고 most (67), then 대금 (52) and 가야금 (47).
- **Not the same as exp002's broken list**: exp002 (CREPE, whole notes) called 7 test clips broken (−12, −10, −7, +12). Here 3 of the 가야금 ones come out at gap 0 with agreement 0.52–0.74, so at least some of exp002's "broken" were tracker errors. exp003–exp009 left those 7 out of the solo answer key.

---

## 4. Timing

| best shift, ms (10th / median / 90th pct) | 대금 | 피리 | 해금 | 아쟁 | 가야금 | 거문고 |
|---|---|---|---|---|---|---|
| pitch judge | −10 / +10 / +20 | 0 / +10 / +30 | −10 / 0 / +30 | −10 / +20 / +120 | −20 / 0 / +30 | −50 / 0 / +40 |
| onset judge | | | | | −40 / −20 / −10 | −50 / −30 / −10 |

- **Sustained instruments are on time**: medians within ±20 ms. The pitch judge flags 459 clips as > 50 ms off, but almost all with no real peak (median gain 0.01); only 33 clearly are (거문고 11, 아쟁 8; 28 of them `CR..` or `FS..` genre codes; 27 with the MIDI early, median +120 ms).
- **Plucked MIDI runs ≈ 20–30 ms late** across the board (exp006 saw the same on the test clips). 112 clips are confidently more than 50 ms off, 104 of them late.
- The pitch judge can't time plucked notes (a ringing string fits any shift): on 가야금 and 거문고 the onset judge is the one to read.

---

## 5. Checks

1. **Smoke run** (3 clips per instrument) before the full run; outputs in `artifacts/smoke/`.
2. **Hand check against exp002's known gaps** (10 test clips): the ±1 clips came out ±1 (3 of 4; AM_F01_06079 came out 0 with agreement 0.44), the +12 대금 and −12 거문고 came out ±12. The 가야금 −10 / −7 / −12 clips came out 0 (section 3).
3. **Clip list**: 5,557 rows, one per clip; instrument codes match the separation manifest (SP01 가야금, SP02 거문고, SR01 해금, SR02 아쟁, WN01 대금, WR01 피리).

---

## 6. Calls I made alone

1. **Note middles for pitch, whole notes for timing**: pitch is judged where the note is steady, so a timing error can't pass for a pitch error.
2. **Pitch gap taken out before timing**: a semitone-off clip would otherwise have no hits at any shift.
3. **Mean, not median, for `residual_cents`** (PESTO's 1/3-semitone bins).
4. **No category thresholds in the command**: it writes raw numbers; the kinds in section 1 are read off afterwards (agreement < 0.5, > 50 ms with gain ≥ 0.05). Jae decides per kind.

---

## 7. Files changed

- New: `src/pipelines/audit_solo_midi.py`, `configs/pipelines/audit_solo_midi.yaml`.
- Changed: `src/evaluate/f0.py` (`reference_pitch_offset`), `src/datasets/solo.py` + `configs/datasets/solo.yaml` (`genre_codes: null` = every genre).

---

## 8. Next

- Per kind, keep / fix / drop: Jae's call. Cheapest fix on the table: move the ±1 clips by their gap.
- Listen to a few octave-gap 거문고 clips and low-agreement clips before deciding.
- Genre codes beyond F01–F03 are still unread.
