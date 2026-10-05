#!/usr/bin/env bash
# Tag the current train/val/test splits as the next numbered dataset version: dataset/vMAJOR.MINOR
#
#   MAJOR  - test.json changed  -> test metrics are NOT comparable with the previous version
#   MINOR  - only train/val changed (test is the same) -> test metrics stay comparable
#
# The md5 of each split is written into the tag message (train_md5=..., val_md5=..., test_md5=...),
# so the training hook can find the version by the CONTENT of the files, from any later commit.
#
# Every annotation set (halpe26, halpe136, ...) has its own numbering: dataset/halpe26/v1.2, dataset/halpe136/v3.0
#
# Usage (run after `dvc repro` and committing dvc.lock):
#   ./dataset_release.sh halpe26                       # create the tag locally
#   ./dataset_release.sh halpe136 -m "batch 2026-10"   # with a description
#   ./dataset_release.sh halpe26 --push                # + dvc push and git push of the tag
#   ./dataset_release.sh halpe26 --list                # list versions of the set
set -euo pipefail

# ---- adjust to your repo layout (paths relative to the git root) ----
DVC_DIR="project/annotations"            # directory with dvc.yaml / dvc.lock
SPLITS_ROOT="project/annotations/custom" # <SPLITS_ROOT>/<set>/train.json ...
# ----------------------------------------------------------------------

SET="${1:-}"
if [[ -z "$SET" || "$SET" == -* ]]; then
  echo "Usage: $0 <annotation set, e.g. halpe26|halpe136> [-m msg] [--push] [--list]" >&2
  exit 2
fi
shift
SPLIT_DIR="$SPLITS_ROOT/$SET"
STAGE="split_coco@$SET"          # foreach stage in dvc.yaml
PREFIX="dataset/$SET/v"

MESSAGE=""
PUSH=0
LIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message) MESSAGE="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"
SPLITS=(train val test)

tag_field() {  # tag_field <tag> <key>  -> value from the tag message
  git tag -l --format='%(contents)' "$1" | sed -n "s/^$2=//p" | head -1
}

versions() {   # all dataset tags, newest first
  git tag -l "${PREFIX}*" --sort=-v:refname | grep -E "^${PREFIX}[0-9]+\.[0-9]+$" || true
}

if [[ $LIST -eq 1 ]]; then
  for t in $(versions); do
    printf '%-16s test %.8s  %s\n' "$t" "$(tag_field "$t" test_md5)" \
      "$(git tag -l --format='%(contents:subject)' "$t")"
  done
  exit 0
fi

# 1. the files on disk must be exactly what the committed dvc.lock describes ----
if ! (cd "$DVC_DIR" && dvc status "$STAGE" --quiet); then
  echo "Splits are out of date with dvc.lock: run 'dvc repro' (or 'dvc commit') first." >&2
  exit 1
fi
if ! git diff --quiet HEAD -- "$DVC_DIR/dvc.lock" || [[ -n "$(git ls-files --others -- "$DVC_DIR/dvc.lock")" ]]; then
  echo "$DVC_DIR/dvc.lock has uncommitted changes: commit it first." >&2
  exit 1
fi

declare -A H
for s in "${SPLITS[@]}"; do
  H[$s]=$(md5sum "$SPLIT_DIR/$s.json" | cut -d' ' -f1)
done

# 2. already released?
for t in $(versions); do
  same=1
  for s in "${SPLITS[@]}"; do
    [[ "$(tag_field "$t" "${s}_md5")" == "${H[$s]}" ]] || { same=0; break; }
  done
  if [[ $same -eq 1 ]]; then
    echo "Nothing to release: these splits are already $t"
    exit 0
  fi
done

# 3. next number
PREV=$(versions | head -1)
if [[ -z "$PREV" ]]; then
  MAJOR=1; MINOR=0; REASON="first version"
else
  IFS=. read -r MAJOR MINOR <<< "${PREV#"$PREFIX"}"
  if [[ "$(tag_field "$PREV" test_md5)" != "${H[test]}" ]]; then
    MAJOR=$((MAJOR + 1)); MINOR=0; REASON="test changed vs $PREV"
  else
    MINOR=$((MINOR + 1)); REASON="train/val changed, test same as $PREV"
  fi
fi
TAG="${PREFIX}${MAJOR}.${MINOR}"

# 4. tag message: subject, machine-readable hashes, human-readable stats
BODY=""
for s in "${SPLITS[@]}"; do BODY+="${s}_md5=${H[$s]}"$'\n'; done
if command -v jq >/dev/null; then
  BODY+=$'\n'
  for s in "${SPLITS[@]}"; do
    BODY+="$s: $(jq '.images | length' "$SPLIT_DIR/$s.json") images, $(jq '.annotations | length' "$SPLIT_DIR/$s.json") annotations"$'\n'
  done
fi

git tag -a "$TAG" -m "${MESSAGE:-$REASON}" -m "$BODY"
echo "Created $TAG ($REASON)"
echo "$BODY"

if [[ $PUSH -eq 1 ]]; then
  (cd "$DVC_DIR" && dvc push)
  git push origin "$TAG"
  echo "Pushed $TAG and DVC data"
else
  echo "Don't forget: (cd $DVC_DIR && dvc push) && git push origin $TAG"
fi
