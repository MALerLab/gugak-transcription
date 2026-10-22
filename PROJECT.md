# PROJECT.md — gugak-transcription
<!-- Stable project facts for any coding agent. Claude Code reads this via CLAUDE.md.
     Agent-specific behaviour lives there, not here. -->

## What belongs here vs Notion
- **Here:** what a coding agent would get wrong without it — environment, folder rules,
  vocabulary, invariants the code must enforce, data locations, verified gotchas. Facts too
  specific or too current for Notion.
- **Notion:** the why and the what-next — project definition, reasoning, roadmap/status,
  decisions + options considered, results.
- **Overlap:** state the rule here in one line + a Notion link; the reasoning lives in Notion.
- **Conflict:** Notion wins on plan; this file wins on repo facts (checked against code).
- **Bloat:** long detail → `docs/*.md`, linked from here.

## Notion
- Hub: **Gugak Transcription** — `401cb377874380f8936bc49915cede1a`
  (https://app.notion.com/p/jgye/Gugak-Transcription-401cb377874380f8936bc49915cede1a)
- Read the hub before any planning or status work.

## Project
- Gugak (traditional Korean music) audio → MIDI transcription.
- Two strands, each in its own worktree on its own branch:
  - **strand-a** (`../gugak-transcription-strand-a`, branch `strand-a`): a transcription model
    trained on labels made from MusicXML, aligned to the separation dataset's stems.
  - **strand-b** (`../gugak-transcription-strand-b`, branch `strand-b`): f0 pitch tracking on
    single stems (민속악), no MusicXML.
- **Separation repo** (sister project): gugak stem separation,
  `~/userdata/repos/gugak_stem_separation` (`MALerLab/gugak-stem-separation`). Read-only from here.

## Environment
- `uv` project, Python 3.11. Never `pip install`.
- PyTorch from the cu128 index (sym8's Blackwell GPUs are sm_120), pinned in `pyproject.toml`.
- Audio I/O: `soundfile`. Avoid `librosa` (lab convention).
- sym8: wrap the outermost command of any heavy job in `python3 src/launch/no_thp_exec.py <cmd>`.
  Without it, jobs stall once page cache fills the container's NUMA node (why → its docstring).
- wandb: entity `maler-gye`, project `gugak_transcription`, set in an untracked `.env`, never
  hardcoded. Nothing loads `.env` automatically: launch scripts run `set -a; source .env; set +a`.
  Never log audio or checkpoints as artifacts.
- Launch scripts: record `git rev-parse HEAD` + `git status --short` in the experiment folder,
  keep checkpoints on `~/storage`, stop runs with SIGINT only.

## Worktree setup
- New worktree → `./setup_worktree.sh`: links to the datasets and the separation repo's
  manifests, then `.venv` from `uv.lock`.
- Untracked but needed path → add one `link` line to the script.

## Git remotes
- One remote, `origin`: fetch from one URL, push to two. Both repos are **public**.
  - fetch: `git@github.com:jae-gye/gugak-transcription.git`
  - push: the same URL + `git@github.com:MALerLab/gugak-transcription.git`
- One `git push` updates both; there is no separate MALerLab remote to fetch from.
- Check: `git remote -v` → 1 fetch line, 2 push lines.

## Layout rules
- `src/` subpackages: `data/` (`audio.py`: audio I/O + DSP primitives, copied from the
  separation repo) · `launch/` (`no_thp_exec.py`: the sym8 THP wrapper).
- `configs/<subpackage>/` mirrors `src/<subpackage>/` — no hardcoded parameters in `src/`.
- `scripts/` = one-off / temporary, **untracked**. May hardcode values; its settings never
  enter `configs/`. Used twice or proven useful → review and promote to `src/`.
- `manifests/separation/` = link to the separation repo's `manifests/parquet/` in its `main`
  checkout, made by `setup_worktree.sh`. Read-only: that repo owns and commits these tables.
- `data/` = symlinks to `~/storage/...`, untracked. Link names match the separation repo's,
  because its manifests store repo-relative paths (`data/gugak_ensemble_71955/ingest/...`).
- `experiments/`: `exp<NNN>_<YYMMDD>_<rest>`.
- The repo is public: audio, MIDI, MusicXML and checkpoints are never committed.

## Vocabulary (one word per concept, in code and docs)
- **MusicXML** — the score files strand-a's labels are made from. Never called MIDI.
- **MIDI** — note-event files (`.mid`). Never a synonym for MusicXML.
- **jeongganbo** — 정간보, the traditional Korean grid notation. Never a synonym for
  MusicXML or MIDI.
- **stem** — one instrument's isolated audio track of one song (the separation dataset's
  per-instrument WAVs).
- **mix** — audio of several instruments together. The separation dataset's publisher
  `master` is a mix, but not the sum of its stems (see gotchas).

## Invariants
- Reuse the separation repo's frozen song split (`manifests/separation/eval_manifest.parquet`);
  never re-split. No song crosses splits.

## Data on disk
- **Separation dataset** (AI Hub 71955, gugak ensemble multitrack):
  `data/gugak_ensemble_71955/` → `~/storage/nia-gugak/gugak_ensemble_71955`.
  - `ingest/<song>/` = `<song>_master.wav` + `<song>_<instrument>.wav`, 903 songs, 44.1 kHz PCM_24.
  - `metadata.csv` (per-song table), `labels/` (tagging JSONs), `sumstem_9stem/{val,test}`.
- **Solo dataset** (AI Hub 71470, solo phrase clips): `data/gugak_solo_71470/` → `~/storage/ngc-gugak`.
  - `ingest/` (clips) · `midi/` (one `.mid` per clip, 9,945) · `labels/` (per-clip JSON).
  - Linked because the separation manifests reference it.
- **Six dragon dataset** (MusicXML converted from jeongganbo): `data/jeongganbo_musicxml/`
  → `~/storage/gugak-transcription/jeongganbo-musicxml`.
  - 91 `.musicxml` files, one per piece: `<piece>_<instruments>.musicxml`. No MIDI.
  - From `musicxml_dataset.zip` on danbinaerin.com/Jeongganbo_dataset, kept alongside.
- **Manifests** (`manifests/separation/`): `source_manifest_v2.parquet` = one row per ingested
  file of both datasets, keyed by `file_id`; `eval_manifest.parquet` = the frozen song split.

## Gotchas
_(verified facts only, grouped by dataset)_

### Separation dataset (AI Hub 71955)
Verified in the separation repo; detail → its `CLAUDE.md`.
- **master ≠ Σ(stems).** The publisher master is mastered (per-stem faders, reverb/FX),
  genre-dependent (민요 ≈ clean sum, 창작국악 heavily processed).
- **master↔stem lag** in 19 of 135 test songs: a fixed whole-sample offset (59, 338 or −2138
  samples). Lag-correct before comparing master to stems; offsets in
  `manifests/separation/master_lag.parquet`.
- **판소리 length mismatch:** in 28 판소리 songs one stem (almost always 가야금) runs longer than
  the master. Trim every stem to the shortest per song; never pad.
- **Numbered stems** (피리1, 피리2, 피리3…) are separate files of the same instrument.
  피리 / 아쟁 labels are subtype-agnostic (피리 = 향피리, 세피리 or 당피리), unresolvable from
  the metadata.
- **No vocal/소리 stem** anywhere, incl. 판소리.
- **Sparse ≠ dead:** 박 and similar play rarely by design; never auto-drop low-activity stems.
- **Silence is not zero:** ingest left a ±1-LSB residue (`1.192e-07`, about −138 dBFS). Test
  silence with a tolerance (≥ `1e-6`, from config), never `== 0`; never group silence by hash.
- **Korean filenames need NFC** for any name join. The master is `<song>_master.wav` on disk
  but `<song>.wav` in the metadata.
- **`metadata.csv` is authoritative**; its `악기구성` column is the per-song instrument list.
  The label JSONs cover only 297 of 903 songs.
- **CSV encodings differ per file:** `metadata.csv` UTF-8, `genre_split.csv` CP949,
  `instrument_split.csv` UTF-8. Try utf-8-sig, then cp949.
- **Test audio is withheld by the publisher:** 903 of the 1,004 songs in the metadata have audio.
- The separation repo's `CLAUDE.md` still describes a raw `source/` folder; on disk only
  `ingest/` exists (checked 2026-10-22).

### Six dragon dataset
- **Not fully proofread** (the dataset page says so): MusicXML was converted from
  OMR'd jeongganbo by rules.
- **Research use only, with citation**: cite Kim et al. 2025 (JOCCH). The repo has no license file.
