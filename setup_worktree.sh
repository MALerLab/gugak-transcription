#!/usr/bin/env bash
# Set up a fresh worktree of this repo: everything git doesn't track.
# Usage, from inside the new worktree: ./setup_worktree.sh
set -euo pipefail  # stop at the first error (-e), on unset variables (-u), on failures inside pipes (-o pipefail)

cd "$(git rev-parse --show-toplevel)"  # always run from the worktree's root folder

STORAGE="$HOME/storage"
SEPARATION_REPO="$HOME/userdata/repos/gugak_stem_separation"  # its main checkout, not a worktree
MAIN_CHECKOUT="$(git worktree list --porcelain | awk 'NR==1 {print $2}')"  # git lists the main checkout first

# 1. Untracked paths every worktree needs, linked in from outside the repo
link() {  # usage: link <absolute source path> <path in repo>
  local source="$1" target="$2"
  if [ ! -e "$source" ]; then echo "skip: $source doesn't exist yet"; return; fi
  if [ -e "$target" ] && [ ! -L "$target" ]; then echo "refuse: $target is a real file, not a link"; return; fi
  mkdir -p "$(dirname "$target")"  # -p: create parent folders if missing
  ln -sfn "$source" "$target"  # -s symlink, -f replace an old link, -n treat an old link as a file
}
# Datasets. Link names match the separation repo's, because its manifests store
# repo-relative paths (data/gugak_ensemble_71955/ingest/...).
link "$STORAGE/nia-gugak/gugak_ensemble_71955" data/gugak_ensemble_71955
link "$STORAGE/ngc-gugak" data/gugak_solo_71470
# Six dragon dataset: MusicXML only, not referenced by the separation manifests
link "$STORAGE/gugak-transcription/jeongganbo-musicxml" data/jeongganbo_musicxml
# This project's own storage folder: experiment artifacts, soundfonts, datasets
link "$STORAGE/gugak-transcription" storage
# Each experiment folder reaches its outputs on storage through an `artifacts` link
for experiment in experiments/exp*/; do
  link "$STORAGE/gugak-transcription/experiments/$(basename "$experiment")" "${experiment}artifacts"
done
# The separation repo's manifests; that repo owns and commits them, read-only here
link "$SEPARATION_REPO/manifests/parquet" manifests/separation
# CLAUDE.md is untracked (public repo); every worktree links to the main checkout's copy
if [ "$PWD" != "$MAIN_CHECKOUT" ]; then link "$MAIN_CHECKOUT/CLAUDE.md" CLAUDE.md; fi

# 2. Python env (.venv), exactly as uv.lock says
uv sync --frozen  # --frozen: install the lockfile as-is, never update it

# 3. Strip notebook outputs on commit: .gitattributes names the filter, this defines it in git config.
#    Worktrees share one git config, and the filter stores this checkout's absolute .venv path,
#    so only the first checkout installs it; later worktrees keep pointing at that one.
if ! git config --get filter.nbstripout.clean > /dev/null; then
  uv run --frozen nbstripout --install
fi

echo "worktree ready"
