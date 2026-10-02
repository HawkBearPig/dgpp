#!/bin/bash
# Serve or stop a dgpp cluster config on $PORT (default 8000).
#   dgpp_serve.sh start [CONFIG]   - launch dgpp-serve in the background
#   dgpp_serve.sh stop             - SIGTERM the server (SIGKILL on timeout)
#   dgpp_serve.sh restart [CONFIG] - stop, then start
#   dgpp_serve.sh status           - report whether the server is up
# Env overrides: DGPP_SERVE_CONFIG, DGPP_SERVE_PORT, DGPP_SERVE_LOG_DIR.
# Examples:
#   dgpp_serve.sh start                                              # RadixArk Flash-Next NVFP4 (default)
#   dgpp_serve.sh start deploy/cluster_qwen3.8-27b-fp8_w1_mtp2.example.json  # Qwen3.8-27B-FP8 + MTP
#   dgpp_serve.sh start deploy/cluster_qwen3.8-27b-fp8_w1_dflash2.example.json  # Qwen3.8-27B-FP8 + DFlash2 drafter (eager)
#   dgpp_serve.sh start [CONFIG]   - launch dgpp-serve in the background
#   dgpp_serve.sh stop             - SIGTERM the server (SIGKILL on timeout)
#   dgpp_serve.sh restart [CONFIG] - stop, then start
#   dgpp_serve.sh status           - report whether the server is up
# Env overrides: DGPP_SERVE_CONFIG, DGPP_SERVE_PORT, DGPP_SERVE_LOG_DIR.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build-release/dgpp-serve"
DEFAULT_CONFIG="$ROOT/deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json"
CONFIG="${DGPP_SERVE_CONFIG:-$DEFAULT_CONFIG}"
PORT="${DGPP_SERVE_PORT:-8000}"
STATE_DIR="${DGPP_SERVE_LOG_DIR:-$ROOT/runs/dgpp_serve}"
PIDFILE="$STATE_DIR/dgpp-serve.pid"
LOG="$STATE_DIR/dgpp-serve.log"

alive() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
port_open() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)" 2>/dev/null; }

# Interactive picker: numbered list of one-node (world_size 1) cluster
# recipes. Reads from /dev/tty so it works even with stdin redirected.
pick_recipe() {
  local files=() f
  while IFS= read -r f; do files+=("$f"); done < <(grep -l '"world_size": 1' "$ROOT"/deploy/cluster_*.json 2>/dev/null | sort)
  [[ ${#files[@]} -gt 0 ]] || { echo "no one-node recipes in $ROOT/deploy" >&2; exit 1; }
  echo "One-node recipes:" >&2
  local i name
  for i in "${!files[@]}"; do
    name="$(basename "${files[$i]}")"
    echo "  $((i + 1))) $name" >&2
  done
  local choice
  printf 'Select recipe [1-%d]: ' "${#files[@]}" >&2
  IFS= read -r choice </dev/tty || { echo "aborted" >&2; exit 1; }
  [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#files[@]} )) \
    || { echo "invalid selection: $choice" >&2; exit 1; }
  printf '%s' "${files[$((choice - 1))]}"
}

do_status() {
  if alive; then echo "running (pid $(cat "$PIDFILE"), port $PORT)"; return 0; fi
  if port_open; then echo "port $PORT is held by an unknown process (no pidfile)"; return 1; fi
  echo "stopped"; return 1
}

do_start() {
  if [[ -n "${1:-}" ]]; then CONFIG="$1"; fi
  if [[ -z "${1:-}" && -z "${DGPP_SERVE_CONFIG:-}" && -t 0 ]]; then CONFIG="$(pick_recipe)" || exit 1; fi
  [[ -x "$BIN" ]] || { echo "missing binary: $BIN (build first)" >&2; exit 1; }
  [[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
  if alive; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
  if port_open; then echo "port $PORT already in use by another process" >&2; exit 1; fi
  mkdir -p "$STATE_DIR"; rm -f "$PIDFILE"
  cd "$ROOT" || exit 1
  # The recipe's engine.kv_capacity is the pool size (its max context); the
  # binary's 8192-token default would silently cap long contexts, so forward
  # the recipe value when it parses. Absent/unparseable: binary default.
  local kv=""
  kv="$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["engine"]["kv_capacity"])
except Exception: pass' "$CONFIG" 2>/dev/null)"
  if [[ "$kv" =~ ^[0-9]+$ ]]; then
    nohup "$BIN" --config "$CONFIG" --kv-capacity "$kv" > "$LOG" 2>&1 &
  else
    nohup "$BIN" --config "$CONFIG" > "$LOG" 2>&1 &
  fi
  echo $! > "$PIDFILE"
  echo "started pid $! (log $LOG); waiting for port $PORT ..."
  for _ in $(seq 1 120); do port_open && { echo "up on port $PORT"; exit 0; }; alive || { echo "server died during boot; tail of $LOG:"; tail -20 "$LOG"; exit 1; }; sleep 5; done
  echo "timed out waiting for port $PORT; tail of $LOG:"; tail -20 "$LOG"; exit 1
}

do_stop() {
  alive || { rm -f "$PIDFILE"; echo "not running"; port_open && { echo "port $PORT held by another process" >&2; exit 1; }; exit 0; }
  local pid; pid="$(cat "$PIDFILE")"
  kill "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  if kill -0 "$pid" 2>/dev/null; then echo "SIGTERM ignored; SIGKILL pid $pid"; kill -9 "$pid"; sleep 2; fi
  rm -f "$PIDFILE"
  echo "stopped"
}

do_logs() {
  [[ -f "$LOG" ]] || { echo "no log yet: $LOG" >&2; exit 1; }
  tail -n "${DGPP_SERVE_LOG_LINES:-50}" "$LOG"
  [[ "${1:-}" == "-f" ]] && tail -F "$LOG"
  return 0
}

USAGE="usage: $0 start|stop|restart|status|logs [-f] [CONFIG]"
case "${1:?$USAGE}" in
  start)   do_start "${2:-}";;
  stop)    do_stop;;
  restart) do_stop; do_start "${2:-}";;
  status)  do_status;;
  logs)    do_logs "${2:-}";;
  *) echo "$USAGE" >&2; exit 2;;
esac
