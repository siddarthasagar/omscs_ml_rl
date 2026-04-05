#!/usr/bin/env bash
# Copy RL project code/artifacts into the external submission repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_DEST_DIR="$HOME/github/cs-7641-2026-spring-schinne3/rl"
DEST_DIR="${1:-$DEFAULT_DEST_DIR}"

echo "Repo root:            $REPO_ROOT"
echo "Destination directory: $DEST_DIR"
echo

mkdir -p "$DEST_DIR"

echo "Cleaning destination root (preserving .git, README.md, and PDFs)..."
find "$DEST_DIR" -mindepth 1 -maxdepth 1 \
  ! -name ".git" \
  ! -name "README.md" \
  ! -name "*.pdf" \
  -exec rm -rf {} +
echo "Destination cleaned."
echo

copy_dir() {
  local rel="$1"
  if [[ -d "$REPO_ROOT/$rel" ]]; then
    echo "Copying directory: $rel"
    rsync -av \
      --exclude='.git/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      --exclude='.DS_Store' \
      --exclude='.ruff_cache/' \
      --exclude='.pytest_cache/' \
      --exclude='.venv/' \
      "$REPO_ROOT/$rel/" "$DEST_DIR/$rel/"
  else
    echo "Warning: missing directory $rel"
  fi
}

copy_file() {
  local rel="$1"
  if [[ -f "$REPO_ROOT/$rel" ]]; then
    cp "$REPO_ROOT/$rel" "$DEST_DIR/$rel"
    echo "Copied file: $rel"
  else
    echo "Warning: missing file $rel"
  fi
}

# Primary code and reproducibility assets
copy_dir "data"
copy_dir "src"
copy_dir "scripts"
copy_dir "tests"

# Root metadata/config files
copy_file ".gitignore"
copy_file ".python-version"
copy_file "Makefile"
copy_file "pyproject.toml"
copy_file "ml_run.sh"
copy_file "requirements.txt"
copy_file "uv.lock"
copy_file "README.md"

echo
echo "Copy complete."
echo "Top-level destination contents:"
ls -la "$DEST_DIR" | head -40
