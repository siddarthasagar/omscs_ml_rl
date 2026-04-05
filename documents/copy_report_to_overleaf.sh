#!/usr/bin/env bash
#
# copy_report_to_overleaf.sh
#
# Sync REPORT_RL/ into the Overleaf git repo at ~/github/overleaf_omscs_ml/RL/
# Excludes LaTeX build artefacts (.aux, .log, .out, .toc, .bbl, etc.)
#
# Usage:
#   bash documents/copy_report_to_overleaf.sh            # default destination
#   bash documents/copy_report_to_overleaf.sh /other/path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$REPO_ROOT/REPORT_RL"
DEFAULT_DEST_DIR="$HOME/github/overleaf_omscs_ml/RL"
DEST_DIR="${1:-$DEFAULT_DEST_DIR}"

echo "Source:      $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo

mkdir -p "$DEST_DIR"

rsync -av \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='*.aux' \
  --exclude='*.log' \
  --exclude='*.out' \
  --exclude='*.toc' \
  --exclude='*.lof' \
  --exclude='*.lot' \
  --exclude='*.bbl' \
  --exclude='*.blg' \
  --exclude='*.fls' \
  --exclude='*.fdb_latexmk' \
  --exclude='*.synctex.gz' \
  --exclude='*.synctex(busy)' \
  "$SOURCE_DIR/" "$DEST_DIR/"

echo
echo "Sync complete."
echo "Destination contents:"
ls -lh "$DEST_DIR"
