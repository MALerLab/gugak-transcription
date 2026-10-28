> Copied as-is from `experiments/exp001_261022_strand_a_pilot/REPORT.md` on branch `strand-a` (uncommitted there, base `915582e`) on 2026-10-23. Relative links (listening clips, tables) point into that experiment folder.

# exp001 — strand A pilot: MusicXML → aligned MIDI labels for 정악 stems

Run 2026-10-22 on sym8, CPU only, branch `strand-a` at `915582e` (start state in `git_head.txt`, `git_status.txt`). Nothing committed.

---

## 1. Verdict

**Viable.** 69 songs (5.47 h of ensemble audio, about 81k notes) get MIDI labels whose notes land on the pitch the stem is actually playing, 65–98% of the time depending on instrument, against 15–22% by chance.

The weak spot is slow 관악/평조 상령산·중령산, where I couldn't place each 장 inside the score reliably. Those, plus 보허자 and one 취타 piece, make 18 excluded songs (1.29 h) for now.

---

## 2. Numbers

### Match (stage 1)

> Each recording is one 장 (or a few), while each MusicXML is a whole movement. So "matching" means finding the right piece and also *where in it* the recording sits.

- **정악 songs with audio**: 151 songs, 11.09 h.
- **Songs with a title candidate**: 99.
  - The other 52 (3.35 h) are repertoire the six dragon dataset doesn't have: 가사, 시조, 종묘제례악, 황하청, 낙양춘, 본령, 유황곡, 만, 정동방곡, 경모궁제례악.
- **Identity confirmed by audio**: 87 songs, 6.76 h, covering 54 of the 91 MusicXML pieces.
  - The rule: the title's piece must be the cheapest of all 91 pieces in a chroma-based search.
- **Rejected by audio**: 12 songs (see memo 8).
- **Usable after alignment**: 69 songs, 5.47 h, 49 pieces.
  - A song counts as usable when its span doesn't overlap a neighbouring 장 and its pitch hit rate beats chance by at least 0.3.
  - 26 of them are 가곡; the rest are 영산회상 family, 여민락, 수제천, 취타 and 도드리 pieces.
- **Usable data by split** (the separation repo's frozen split, not re-split):

  | split | songs | hours | notes |
  |---|---|---|---|
  | train | 49 | 3.75 | 55,954 |
  | val | 12 | 0.96 | 14,044 |
  | test | 8 | 0.76 | 11,196 |

- **Stem-hours, usable set**: 27.8 in total.
  - 대금, 피리, 해금: 5.47 h each.
  - 가야금: 4.84 h.
  - 거문고: 4.39 h.
  - 아쟁: 2.15 h.
- Tables: `match_table.parquet` (one row per 정악 song, with confidence, located span and part↔stem overlap), `match_costs.parquet` (every song × every piece).

### Pitch: is the MusicXML at sounding pitch?

Measured from each stem's f0 after alignment, not assumed.

- **대금, 피리, 해금, 가야금, 거문고**: written = sounding, octave included.
- **아쟁**: sounds one octave **below** written, in every piece with an 아쟁 stem (16 pieces, plus 보허자 where it adds to the −3).
- **보허자**: written 3 semitones **above** sounding.
  - Its 대금 part comes out +9 (that is −3 + an octave).
- **현악취타 취타 / 길타령**: written 5 semitones above sounding (대금 +7).
- **Tuning**: 0 to +19 cents sharp of A440 (대금 sharpest, median +19).
- The MIDI files apply these shifts, so they are at sounding pitch.

### Alignment quality (usable set, median over songs)

> The check: for every aligned note, take the stem's f0 in the middle of the note. A **hit** means it's within half a semitone of the note's pitch. "Chance" is the same notes shifted by ±3 and ±7 s; "unaligned" is the right section of score at a constant average tempo.

| stem | chance | unaligned | aligned per song | aligned per stem |
|---|---|---|---|---|
| 아쟁 | 0.22 | 0.83 | 0.98 | 0.99 |
| 대금 | 0.18 | 0.45 | 0.71 | 0.82 |
| 가야금 | 0.21 | 0.69 | 0.89 | 0.89 |
| 거문고 | 0.20 | 0.62 | 0.81 | 0.83 |
| 해금 | 0.15 | 0.44 | 0.65 | 0.65 |
| 피리 | 0.18 | 0.49 | 0.80 | 0.86 |

- **Onset agreement** (detected onsets within ±100 ms, F-measure) only means something for plucked stems:
  - 가야금: chance 0.09, unaligned 0.21, aligned 0.69 (per song) / 0.75 (per stem).
  - 거문고: chance 0.07, unaligned 0.18, aligned 0.68 / 0.63.
  - For winds and bowed strings the onset detector barely sees legato note changes: 0.17–0.42 aligned vs 0.14–0.28 chance.
- **Two honest caveats on the pitch hit rate**:
  - It's partly circular. The alignment itself maximises chroma agreement, so a high hit rate says "the notes sit on matching pitch", not "onsets are within X ms".
  - It can't tell two span hypotheses apart in repetitive slow movements (memo 6). The 장 tiling check below is the independent sanity check.
- **Tiling check**: consecutive 장 of a movement should cover the score in order without overlap.
  - 현악영산회상 상령산 1–4장 land at quarters 9–90, 90–210, 210–330, 330–510.
  - 여민락 1–3장 tile exactly at 0/960/1920/2880.
  - 수제천 1–3장 tile at 0/181/351/526.
- Per-song tables: `alignment_quality_by_song.parquet` (one row per song); `alignment_quality.parquet` (one row per song × stem × method).

---

## 3. Listening list

Every clip is 30 s and loudness-matched. **real** = the actual stem (or the mix of the matched stems), **render** = the aligned MIDI played by General MIDI stand-ins, **layered** = both together. The aligned `.mid` sits next to each clip's WAVs.

1. **Ensemble, 현악영산회상 상령산 1장** (0009, from 46 s, all 5 parts): [real](listening/01_ensemble_hyeonak_sangnyeongsan/01_ensemble_hyeonak_sangnyeongsan_real.wav) · [render](listening/01_ensemble_hyeonak_sangnyeongsan/01_ensemble_hyeonak_sangnyeongsan_render.wav) · [layered](listening/01_ensemble_hyeonak_sangnyeongsan/01_ensemble_hyeonak_sangnyeongsan_layered.wav)
   - **Unaligned version, same 30 s**: [real](listening/01_ensemble_hyeonak_sangnyeongsan/01_ensemble_hyeonak_sangnyeongsan_unaligned_real.wav) · [render](listening/01_ensemble_hyeonak_sangnyeongsan/01_ensemble_hyeonak_sangnyeongsan_unaligned_render.wav) · [layered](listening/01_ensemble_hyeonak_sangnyeongsan/01_ensemble_hyeonak_sangnyeongsan_unaligned_layered.wav)
   - Listen for: in the aligned layer, the orchestra changes notes when the ensemble does; in the unaligned one it drifts within seconds.
   - Pitch hit 0.74 aligned vs 0.30 unaligned.
2. **가야금, 가곡 남창우조 우롱** (0069, from 152 s): [real](listening/02_gayageum_gagok_urong/02_gayageum_gagok_urong_real.wav) · [render](listening/02_gayageum_gagok_urong/02_gayageum_gagok_urong_render.wav) · [layered](listening/02_gayageum_gagok_urong/02_gayageum_gagok_urong_layered.wav)
   - Listen for: harp plucks on top of the 가야금 plucks.
   - Pitch hit 0.90, onset F 0.74.
3. **피리, 수제천 1장** (0140, from 35 s): [real](listening/03_piri_sujecheon/03_piri_sujecheon_real.wav) · [render](listening/03_piri_sujecheon/03_piri_sujecheon_render.wav) · [layered](listening/03_piri_sujecheon/03_piri_sujecheon_layered.wav)
   - Listen for: 수제천 is very slow and free, so the timing is hard. The oboe should hold the long 피리 notes; the 시김새 bends aren't in the score, so expect flat oboe notes where the 피리 slides.
   - Pitch hit 0.68 vs 0.13 unaligned.
4. **아쟁, 여민락 6장** (0130, from 143 s): [real](listening/04_ajaeng_yeominlak/04_ajaeng_yeominlak_real.wav) · [render](listening/04_ajaeng_yeominlak/04_ajaeng_yeominlak_render.wav) · [layered](listening/04_ajaeng_yeominlak/04_ajaeng_yeominlak_layered.wav)
   - Listen for: the cello in the same octave as the 아쟁. This checks the "아쟁 sounds an octave below written" fix.
   - Pitch hit 0.96.
5. **대금, 영산회상 군악** (0063, from 18 s, a fast piece at 3.5 quarters/s): [real](listening/05_daegeum_gunak_fast/05_daegeum_gunak_fast_real.wav) · [render](listening/05_daegeum_gunak_fast/05_daegeum_gunak_fast_render.wav) · [layered](listening/05_daegeum_gunak_fast/05_daegeum_gunak_fast_layered.wav)
   - Listen for: whether the flute keeps up with quick note changes.
   - Pitch hit 0.79.
6. **거문고, 천년만세 계면가락도드리** (0046, from 41 s): [real](listening/06_geomungo_gyemyeon_garakdodeuri/06_geomungo_gyemyeon_garakdodeuri_real.wav) · [render](listening/06_geomungo_gyemyeon_garakdodeuri/06_geomungo_gyemyeon_garakdodeuri_render.wav) · [layered](listening/06_geomungo_gyemyeon_garakdodeuri/06_geomungo_gyemyeon_garakdodeuri_layered.wav)
   - Listen for: pizzicato strings landing on the 거문고 strokes.
   - Onset F 0.84, pitch hit 0.74.
7. **해금, 보허자 1~2장** (0123, from 264 s): [real](listening/07_haegeum_bohoeja_transposed/07_haegeum_bohoeja_transposed_real.wav) · [render](listening/07_haegeum_bohoeja_transposed/07_haegeum_bohoeja_transposed_render.wav) · [layered](listening/07_haegeum_bohoeja_transposed/07_haegeum_bohoeja_transposed_layered.wav)
   - Listen for: violin in tune with the 해금. The score is written 3 semitones high, so if it sounds a minor third off, the transposition fix failed.
   - Not in the usable set: its span overlaps 3장's (memo 6).
   - Pitch hit 0.71.
8. **Worst case: 관악영산회상 중령산 2장** (0014, from 19 s, all 4 parts): [real](listening/08_worst_gwanak_jungnyeongsan/08_worst_gwanak_jungnyeongsan_real.wav) · [render](listening/08_worst_gwanak_jungnyeongsan/08_worst_gwanak_jungnyeongsan_render.wav) · [layered](listening/08_worst_gwanak_jungnyeongsan/08_worst_gwanak_jungnyeongsan_layered.wav)
   - Listen for: stretches where the orchestra plays a different phrase than the ensemble. This is the slow-movement placement problem.
   - Pitch hit 0.33 vs 0.16 chance; excluded from the usable set.

**GM stand-ins**, one per gugak instrument (`configs/labels/convert.yaml`):

| gugak | GM instrument (program, 0-based) | why |
|---|---|---|
| 대금 | Flute (73) | transverse bamboo flute |
| 피리 | Oboe (68) | double reed, nasal |
| 해금 | Violin (40) | bowed, mid register |
| 아쟁 | Cello (42) | bowed, low |
| 가야금 | Orchestral Harp (46) | plucked zither |
| 거문고 | Pizzicato Strings (45) | plucked, low, percussive attack |

- Soundfont: GeneralUser GS, downloaded to `~/storage/gugak-transcription/soundfonts/` (the system TimGM6mb sounds poor).

---

## 4. Memos

1. **synctoolbox pins `pandas<3`; this repo pins `pandas>=3.0.6`.**
   - I added `override-dependencies = ["pandas>=3.0.6"]` to `pyproject.toml`; synctoolbox only uses pandas in one CSV helper.
   - synctoolbox pulls in `librosa` as a dependency. I never call librosa directly; audio I/O stays `soundfile`.
2. **partitura's `note_array()` crashes when grace notes are included** (a partitura bug).
   - I read `part.notes_tied` directly instead; ties are merged and grace notes kept.
3. **libfmp's subsequence DTW counts a 2-frame step as one frame**, which biases paths toward skipping audio.
   - I wrote a small numba version that weights each step by the audio frames it covers (`src/labels/locate.py`).
4. **Recordings are per 장, MusicXML per movement.**
   - I locate each recording inside the score with subsequence DTW over 1 Hz chroma (CENS), at 9 tempi from 0.25 to 4 quarters/s.
   - The same cost doubles as the match check.
5. **My first acceptance rule rejected correct matches.**
   - It required beating the best *unrelated* piece by 0.03. Related movements (가곡 forms, 타령/군악) sit close, so 43 songs failed even though their title's piece was the best of all 91.
   - New rule: identity = the cheapest of all 91 pieces; margins only set high/medium/low; alignment quality has the final say.
6. **Slow 관악/평조 상령산·중령산: 장 placed at double time.** Placed one at a time, consecutive 장 claimed overlapping score. For example, 관악 중령산 1장 claimed quarters 0–240 and 2장 claimed 4–242.
   - Alternative tried: place all 장 of a movement jointly, with one path through the concatenated recordings (`src/labels/relocate.py`).
   - For 관악 the joint spans are stable (1 Hz vs 10 Hz features agree within about 25 quarters), but the pitch hit stayed low (per song ≈ 0.3, chance ≈ 0.2). So the songs are excluded.
   - For 평조 the joint spans were unstable (boundaries moved about 30 quarters), so I kept the one-at-a-time spans and excluded the songs that overlap.
   - Not solved. Likely causes: 장 reuse melodic material, so a 장 also matches its neighbours' score; and the performed version may differ from the score (repeats). Ideas in section 6.
   - 보허자 1~2장 + 3장 also overlap (their spans add up to more score than exists, which hints at a performed repeat). Excluded too.
7. **현악취타 길타령 (0149)**: identity confirmed, but pitch hit only 0.40 vs 0.16 chance. Excluded.
8. **Rejected by the audio check, 12 songs**:
   - **현악 세령산 1~2장, 하현도드리 3~4장, 염불도드리 2~4장, 타령 3~4장** (0032, 0050, 0056, 0060): each is 3–4× longer than the same sections played by the other suites. They probably contain more than the title says. Not checked by ear.
   - **관악 가락덜이 1·2·3장** (0034–0036): each recording is about 311–327 s, but the MusicXML has only 10 measures.
   - **평조 상현도드리 ×2, 관악 타령 ×2, 동동** (0041, 0042, 0057, 0058, 0139): no explanation found; maybe a different version than the score.
   - I checked the equal durations (0034, 0035, 0057 are all 311.14 s): different files, uncorrelated audio, so it's a coincidence and not duplicate uploads.
9. **Onset detection is weak on legato instruments**, so onset F is only reported as evidence for 가야금 and 거문고.
10. **Early stop did not trigger**: plenty of matches.

---

## 5. Decisions I made on your behalf

1. **Main labels = one alignment per song.** The mix of the matched stems drives one timing path that every part follows.
   - Per-stem alignment was also run. It scores slightly higher on pitch (대금 0.82 vs 0.71), but each stem then drifts on its own and the timing is less consistent across parts. Both MIDI sets are on storage.
2. **Listening clips use the per-song alignment.**
3. **Title rules** (`configs/labels/match.yaml`):
   - 현악영산회상 ↔ six dragon "영산회상".
   - 관악영산회상 삼현도드리 ↔ "관악영산회상 상현도드리".
   - 만파정식지곡 / 절화 / 일승월향 / 금전악 ↔ 취타 or 현악취타 (the audio picks between the two).
4. **가곡**: every six dragon 가곡 piece with the same form name is a candidate (남창/여창 × 우조/계면/반우반계), and the audio picks.
   - Example: 0086 "평조 두거" → 여창우조 두거.
5. **Usable rule**: span overlaps a neighbouring 장 by at most 10%, and the per-song pitch hit rate is at least chance + 0.3.
6. **Joint placement kept for 관악 상령산/중령산; one-at-a-time spans kept for 평조.** Both groups are excluded either way.
7. **0021 (평조 중령산 4장) re-admitted** after being rejected alone: its span 330–420 sits exactly between 3장 and 5장. It passed the usable rule.
8. **Grace notes** (the 시김새 acciaccaturas) get a fixed 80 ms taken from the start of their main note.
9. **Sounding pitch** is taken per (piece, part) from the measured f0 (median over that piece's songs) and written into the MIDI.
10. **Velocity** is a constant 80 (the MusicXML has no dynamics).
11. **Stems with no MusicXML part** (소금, 단소, 양금, all percussion) are left out of the alignment audio and get no labels.
12. **Each clip's start** is the 30 s window with the most note onsets of that clip's parts.
13. **Caches, aligned note tables and all aligned MIDI** live on `storage/experiments/exp001_261022_strand_a_pilot/` (`cache/`, `aligned_midi/{per_song,per_stem}`), not in the repo.

---

## 6. What's next

> 5.5 h is far too little to train from scratch, but plausible for adapting a pretrained transcription model — the prof's plan.

- **Training data available now**: 3.75 h train / 0.96 h val / 0.76 h test, about 28 stem-hours, 81k notes, 6 instruments.
- **First experiment I'd suggest** (not started; needs your sign-off):
  - Fine-tune a piano-pretrained Onsets and Frames on **single stems** with the per-song labels: one shared model, notes only, no instrument output.
  - Baseline: the same pretrained model, zero-shot.
  - Metric: note F-measure (onset ±50 ms) on the val split, per instrument.
  - Why stems first: it tests "do these labels teach anything" without the 6-channel mix problem. The 6-instrument mix model is the step after.
- **Cheap wins before or alongside**:
  - Listen to the 8 clips. If 01–06 sound right, the labels are good enough to train on.
  - Recover the slow movements (about 1 h): joint placement with a tempo prior from the 현악 version of the same movement, or a finer DTW once the 장 boundaries are fixed.
  - Remix aligned stems (the separation repo's random-mixing sampler) to multiply training mixtures for the later mix model.

---

## 7. Files changed

**Tracked files modified**:
- `pyproject.toml`: adds synctoolbox, partitura, pretty_midi, plus the pandas override (memo 1).
- `uv.lock`: regenerated by `uv add`.

**New, would be tracked if committed**:
- `src/labels/`, the label pipeline:
  - `convert.py`: MusicXML → note table → MIDI.
  - `features.py`: chroma / DLNCO / CENS.
  - `locate.py`: subsequence DTW.
  - `match.py`: stage 1.
  - `relocate.py`: joint 장 placement.
  - `align.py`: MrMsDTW.
  - `quality.py`: YIN f0 and onset checks.
  - `pipeline.py`: align + measure every match.
  - `render.py` and `listening.py`: rendering and clips.
- `configs/labels/*.yaml`: one config per module.
- `experiments/exp001_261022_strand_a_pilot/`:
  - `REPORT.md`, `git_head.txt`, `git_status.txt`.
  - `*.parquet` tables: numbers and paths only, no audio or notes.
  - `*_stage1spans.parquet` and `match_table_stage1.parquet` keep the before-relocation state behind memo 6.

**Untracked or ignored**:
- `scripts/`: `explore_one_song.py`, `inspect_musicxml.py`, `summarize_pilot.py` (one-offs).
- `listening/*.wav`, `*.mid`, `*.log`: git-ignored, as they must be (public repo).

**What I'd commit**:
- `pyproject.toml` + `uv.lock`, `src/labels/`, `configs/labels/`.
- The report folder minus the `*_stage1spans` tables, unless you want the before/after.
- A one-line `PROJECT.md` layout entry for `src/labels/` (not edited).
