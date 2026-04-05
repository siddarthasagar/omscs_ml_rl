#!/usr/bin/env bash
# =============================================================================
# ml_run.sh — Cross-platform ML training runner
#
# Wraps any command with OS-specific sleep prevention so macOS/Linux never
# throttles or suspends a long-running training job.
#
# INLINE mode (default):
#   Runs the command in the current terminal, synchronously, with sleep
#   prevention active. Safe to use inside Makefile targets — Make correctly
#   waits for completion before moving to the next step.
#
#   Usage: bash ml_run.sh "make pipeline"
#
# DETACHED mode (--detach):
#   Launches the command inside a background tmux/screen session and returns
#   immediately. Use this when you want to close the terminal and let training
#   continue overnight.
#
#   Usage: bash ml_run.sh --detach "make pipeline" [session_name]
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[ml_run]${RESET} $*"; }
success() { echo -e "${GREEN}[ml_run]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[ml_run]${RESET} $*"; }
error()   { echo -e "${RED}[ml_run]${RESET} $*" >&2; exit 1; }

# ── Argument Parsing ─────────────────────────────────────────────────────────
DETACH=false
if [[ "${1:-}" == "--detach" ]]; then
  DETACH=true
  shift
fi

CMD="${1:-}"
SESSION="${2:-ml_training}"

[[ -z "$CMD" ]] && error "No command supplied.\n\nUsage:\n  bash ml_run.sh \"make pipeline\"             # inline (default)\n  bash ml_run.sh --detach \"make pipeline\"    # background session"

# ── OS Detection ─────────────────────────────────────────────────────────────
detect_os() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux)  echo "linux" ;;
    *)      echo "unknown" ;;
  esac
}

OS=$(detect_os)

# ── Sleep-prevention wrapper ──────────────────────────────────────────────────
SLEEP_PREFIX=""

setup_macos() {
  command -v caffeinate &>/dev/null || { warn "caffeinate not found. Running without sleep guard."; return; }

  local term_bundle
  term_bundle=$(osascript -e 'id of app "'"$(ps -o comm= -p $PPID | xargs basename)"'"' 2>/dev/null || true)
  if [[ -n "$term_bundle" ]]; then
    defaults write "$term_bundle" NSAppSleepDisabled -bool YES 2>/dev/null \
      && info "App Nap disabled for: $term_bundle" \
      || warn "Could not disable App Nap for $term_bundle (non-fatal)"
  fi

  SLEEP_PREFIX="caffeinate -dims"
  info "caffeinate active — system/display/disk sleep blocked"
}

setup_linux() {
  if command -v systemd-inhibit &>/dev/null; then
    SLEEP_PREFIX="systemd-inhibit --what=idle:sleep:handle-lid-switch --who=ml_run --why=TrainingJob"
    info "systemd-inhibit active — idle/sleep/lid-close blocked"
  elif command -v xdg-screensaver &>/dev/null; then
    ( while true; do xdg-screensaver reset 2>/dev/null; sleep 50; done ) &
    warn "systemd-inhibit not found; using xdg-screensaver keep-alive (PID $!)"
  else
    warn "No sleep-inhibit tool found — system may sleep during long runs"
  fi
}

case "$OS" in
  macos)   setup_macos ;;
  linux)   setup_linux ;;
  *)       warn "Unknown OS — running without sleep prevention" ;;
esac

# ── INLINE mode ───────────────────────────────────────────────────────────────
run_inline() {
  if [[ -n "$SLEEP_PREFIX" ]]; then
    exec $SLEEP_PREFIX bash -c "$CMD"
  else
    exec bash -c "$CMD"
  fi
}

# ── DETACHED mode ─────────────────────────────────────────────────────────────
pick_multiplexer() {
  command -v tmux &>/dev/null && echo "tmux" && return
  command -v screen &>/dev/null && echo "screen" && return
  echo "none"
}

run_detached() {
  local mux
  mux=$(pick_multiplexer)
  [[ "$mux" == "none" ]] && error "Neither tmux nor screen found.\n  macOS: brew install tmux\n  Linux: sudo apt install tmux"

  local logfile="$(pwd)/artifacts/logs/${SESSION}.runner.log"
  mkdir -p "$(dirname "$logfile")"

  local full_cmd
  if [[ -n "$SLEEP_PREFIX" ]]; then
    full_cmd="$SLEEP_PREFIX $CMD 2>&1 | tee \"$logfile\""
  else
    full_cmd="$CMD 2>&1 | tee \"$logfile\""
  fi

  info "Log file: $logfile"

  if [[ "$mux" == "tmux" ]]; then
    tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
    tmux new-session -d -s "$SESSION" -x 220 -y 50
    tmux send-keys -t "$SESSION" "$full_cmd" Enter
    success "Launched in tmux session '${BOLD}${SESSION}${RESET}${GREEN}'."
    echo ""
    echo -e "  ${CYAN}Attach:${RESET}   tmux attach -t $SESSION"
    echo -e "  ${CYAN}Detach:${RESET}   Ctrl+B then D"
    echo -e "  ${CYAN}Logs:${RESET}     tail -f $logfile"
    echo -e "  ${CYAN}Kill:${RESET}     tmux kill-session -t $SESSION"
  else
    screen -list 2>/dev/null | grep -q "$SESSION" && { warn "screen session '$SESSION' exists. Attach with: screen -r $SESSION"; exit 0; }
    screen -dmS "$SESSION" bash -c "$full_cmd"
    success "Launched in screen session '${BOLD}${SESSION}${RESET}${GREEN}'."
    echo ""
    echo -e "  ${CYAN}Attach:${RESET}   screen -r $SESSION"
    echo -e "  ${CYAN}Detach:${RESET}   Ctrl+A then D"
    echo -e "  ${CYAN}Logs:${RESET}     tail -f $logfile"
    echo -e "  ${CYAN}Kill:${RESET}     screen -S $SESSION -X quit"
  fi

  echo ""
  info "Training is running in the background. Safe to close your terminal."
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
if [[ "$DETACH" == "true" ]]; then
  run_detached
else
  run_inline
fi
