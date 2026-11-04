#!/usr/bin/env bash
# Launch an application across the fabric, with rank 0 local and peers over SSH.
# Stage files before starting rank 0, then launch peers once its rendezvous
# listener is ready. Observe their process records, monitor the head, and
# compare generated-token digests when the run finishes. Cleanup signals only
# processes recorded for this output directory. Use an idle test cluster.
# Each node needs its own model cache and compatible runtime libraries.
#
# Usage (from the repo root, ON THE HEAD NODE — rank 0 is this box):
#   scripts/fabric_run.sh [options] -- APP_ARGS...
#
#   APP_ARGS are passed verbatim to glm_gen_check on EVERY rank; the
#   script appends --world/--rank/--peer/--port itself (do not pass
#   them yourself).
#
# Options:
#   --app PATH      binary to stage + run (default $BUILD/glm_gen_check)
#   --port N        rendezvous port (default: first free starting at DGPP_FABRIC_PORT)
#   --timeout SECS  head watchdog; 0 disables (default 1800)
#   --no-stage      skip the peer scp (peers already carry this binary)
#   --stage-file F  ALSO scp F to the peers and rewrite the peers'
#                  --requests / --teacher-file argument to the copy (rank 0 keeps
#                  the local path — the app's manifest-hash log line is
#                  the cross-rank identity check). Stages even under
#                  --no-stage: the manifest changes more often than the
#                  binary and costs nothing to ship.
#   --force         stop processes recorded for this output directory
#   --log-dir DIR   logs land here (default $BUILD/fabric-runs/<UTC ts>)
#   --head-wrap CMD prefix rank 0's command line with CMD (word-split) —
#                  the profiling hook: e.g. --head-wrap "nsys profile
#                  --trace=cuda -o /tmp/r0" traces rank 0's kernels while
#                  the peers run bare. The wrapper must pass stdout
#                  through (the launch waits on the "rendezvous
#                  listening" log line) and exit with the app.
#   --fetch-logs    after the run, scp every peer's log into LOG_DIR as
#                  rN.log next to r0.log — the cross-rank analysis tools
#                  (scripts/fabric_xrank.py) read the directory as a unit.
#                  Bus-generation IDs align rank events even when clocks differ.
#   --node-probe    run scripts/node_probe.sh on EVERY node for the run's
#                  duration (1 Hz /proc/vmstat + meminfo deltas into the
#                  node's $PEER_DIR/probe_rN.log; fetched with the logs).
#   --node-cmd CMD  run CMD (a shell string) on every node in the
#                  background for the run's duration, killed at the end —
#                  useful for controlled background-load experiments. Runs as
#                  DGPP_SSH_USER; the script does not arrange extra privileges.
#
# Examples:
#   scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
#       --chat "The capital of France is" \
#       --system "You are a concise assistant." --steps 64
#   scripts/fabric_run.sh --no-stage -- \
#       --model unsloth/GLM-5.3-Flash-FP8 --text "Hello" --steps 2
#   scripts/fabric_run.sh --stage-file build-ci/sched_smoke.jsonl -- \
#       --model unsloth/GLM-5.3-Flash-FP8 --requests build-ci/sched_smoke.jsonl \
#       --max-concurrency 2 --kv-capacity 256
#
# .env supplies DGPP_NODES, DGPP_SSH_USER and DGPP_STAGE_DIR.
# The selected deployment's world_size determines how many nodes to use.
# Export those same keys to override the site settings for one run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
BUILD="${DGPP_BUILD_DIR:-$ROOT/build-ci}"
APP="$BUILD/glm_gen_check"
FABRIC_HEAD=$(dgpp_head) || exit 1
PEER_LIST=$(dgpp_peers) || exit 1
read -r -a FABRIC_PEERS <<< "$PEER_LIST"
FABRIC_USER=$(dgpp_ssh_user) || exit 1
PEER_DIR="$DGPP_STAGE_DIR"
SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=8)
MONITOR_TIMEOUT=1800
STAGE=1
STAGE_FILE=""
FORCE=0
LOG_DIR=""
HEAD_WRAP=()
FETCH_LOGS=0
NODE_PROBE=0
NODE_CMD=""

die() { echo "fabric_run: $*" >&2; exit 1; }
peer_ssh() { timeout 20 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$1" "${2:-true}"; }

# ---------------------------------------------------------------- options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --timeout) MONITOR_TIMEOUT="$2"; shift 2 ;;
    --no-stage) STAGE=0; shift ;;
    --stage-file) STAGE_FILE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --head-wrap) read -r -a HEAD_WRAP <<< "$2"; shift 2 ;;
    --fetch-logs) FETCH_LOGS=1; shift ;;
    --node-probe) NODE_PROBE=1; shift ;;
    --node-cmd) NODE_CMD="$2"; shift 2 ;;
    --) shift; break ;;
    *) die "unknown option $1 (app args go after --)" ;;
  esac
done
APP_ARGS=("$@")
[[ ${#APP_ARGS[@]} -gt 0 ]] || die "no app args after -- (e.g. -- --model ID --chat TEXT)"
[[ -x "$APP" ]] || die "app not found/executable: $APP (build it, or pass --app)"
# Refuse to stage over an executable that another run is using.
APP_NAME="$(basename "$APP")"
# Linux exposes only the first 15 bytes in comm (what pgrep -x matches).
# Keep this guard conservative when two executable names share that prefix.
APP_COMM="${APP_NAME:0:15}"
WORLD=$((1 + ${#FABRIC_PEERS[@]}))
LOG_DIR="${LOG_DIR:-$BUILD/fabric-runs/$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
exec {RUN_LOCK}> "$LOG_DIR/launcher.lock"
flock -n "$RUN_LOCK" || die "another launcher owns $LOG_DIR"
RUN_ID=$(python3 "$ROOT/scripts/run_helpers.py" path-id "$LOG_DIR") || exit 1
PEER_DIR="$DGPP_STAGE_DIR/fabric/$RUN_ID"
PROCESS_HELPER="$ROOT/scripts/cluster_process.py"
STATE="$LOG_DIR/head.process.json"
export DGPP_LOG_LEVEL="${DGPP_LOG_LEVEL:-info}"  # render/eos lines are INFO
PROBE_SCRIPT="$ROOT/scripts/node_probe.sh"
ALL_NODES=(127.0.0.1 "${FABRIC_PEERS[@]}")  # index == rank

# ------------------------------------------------------------- cleanup
kill_all() {
  python3 "$PROCESS_HELPER" signal --state "$STATE" --signal KILL || true
  for i in "${!FABRIC_PEERS[@]}"; do
    ip=${FABRIC_PEERS[$i]}; rank=$((i + 1))
    peer_ssh "$ip" "test ! -f $PEER_DIR/cluster_process.py || python3 $PEER_DIR/cluster_process.py signal --state $PEER_DIR/fabric_${RUN_ID}_rank$rank.process.json --signal KILL" || true
  done
  stop_node_helpers
}
# Helpers have separate identity records; the tag labels their output.
HELPER_TAG="fabric_run_helper_$RUN_ID"
stop_node_helpers() {
  [[ $NODE_PROBE -eq 1 || -n "$NODE_CMD" ]] || return 0
  # Local cleanup does not depend on SSH access back to the head.
  for kind in probe nodecmd; do
    python3 "$PROCESS_HELPER" signal --state "$LOG_DIR/$kind.process.json" --signal TERM || true
  done
  for ip in "${FABRIC_PEERS[@]}"; do
    timeout 15 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$ip" \
      "test ! -f $PEER_DIR/cluster_process.py || { python3 $PEER_DIR/cluster_process.py signal --state $PEER_DIR/probe.process.json --signal TERM; python3 $PEER_DIR/cluster_process.py signal --state $PEER_DIR/nodecmd.process.json --signal TERM; }" >/dev/null 2>&1 || true
  done
}
OWN_RUN=0
trap 'if [[ $OWN_RUN -eq 1 ]]; then kill_all; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------- stale / staging
if [[ $FORCE -eq 1 ]]; then
  echo "fabric_run: --force — stopping only processes recorded for $LOG_DIR"
  kill_all
  sleep 2
fi
python3 "$PROCESS_HELPER" status --state "$STATE" >/dev/null \
  && die "this output directory has a running recorded process; use --force to replace it"
for i in "${!FABRIC_PEERS[@]}"; do
  ip=${FABRIC_PEERS[$i]}; rank=$((i + 1))
  peer_ssh "$ip" "test -f $PEER_DIR/cluster_process.py && python3 $PEER_DIR/cluster_process.py status --state $PEER_DIR/fabric_${RUN_ID}_rank$rank.process.json" >/dev/null 2>&1 \
    && die "this output directory has a running recorded peer on $ip; use --force to replace it"
done
pgrep -x "$APP_COMM" >/dev/null && die "a $APP_NAME already runs here; stop its owning run first"
for ip in "${FABRIC_PEERS[@]}"; do
  peer_ssh "$ip" "pgrep -x $APP_COMM" >/dev/null 2>&1 \
    && die "a $APP_NAME already runs on $ip; stop its owning run first"
done

OWN_RUN=1
for ip in "${FABRIC_PEERS[@]}"; do
  peer_ssh "$ip" "mkdir -p $PEER_DIR" || die "cannot create staging directory on $ip"
  scp -q -o BatchMode=yes -o ConnectTimeout=8 "$PROCESS_HELPER" "$FABRIC_USER@$ip:$PEER_DIR/cluster_process.py" || die "cannot stage process helper on $ip"
done

if [[ $STAGE -eq 1 ]]; then
  for ip in "${FABRIC_PEERS[@]}"; do
    echo "fabric_run: staging $APP -> $FABRIC_USER@$ip:$PEER_DIR/"
    scp -o BatchMode=yes -o ConnectTimeout=8 "$APP" \
      "$FABRIC_USER@$ip:$PEER_DIR/" >/dev/null || die "staging to $ip failed"
  done
fi
if [[ $NODE_PROBE -eq 1 || -n "$NODE_CMD" ]]; then
  [[ -x "$PROBE_SCRIPT" ]] || die "node probe script missing: $PROBE_SCRIPT"
  for ip in "${FABRIC_PEERS[@]}"; do
    scp -o BatchMode=yes -o ConnectTimeout=8 "$PROBE_SCRIPT" \
      "$FABRIC_USER@$ip:$PEER_DIR/" >/dev/null || die "staging probe to $ip failed"
  done
fi
if [[ -n "$STAGE_FILE" ]]; then
  [[ -f "$STAGE_FILE" ]] || die "--stage-file not found: $STAGE_FILE"
  for ip in "${FABRIC_PEERS[@]}"; do
    scp -o BatchMode=yes -o ConnectTimeout=8 "$STAGE_FILE" \
      "$FABRIC_USER@$ip:$PEER_DIR/" >/dev/null \
      || die "staging $STAGE_FILE to $ip failed"
  done
  echo "fabric_run: staged $STAGE_FILE (peers read it as $PEER_DIR/$(basename "$STAGE_FILE"))"
fi

# ------------------------------------------------------------ the port
if [[ -z "${PORT:-}" ]]; then
  for p in $(seq "$DGPP_FABRIC_PORT" "$((DGPP_FABRIC_PORT + 19 > 65535 ? 65535 : DGPP_FABRIC_PORT + 19))"); do
    # connect_ex == 0 means something answers: not free. Probe, don't bind —
    # rank 0 owns the listener, TIME_WAIT on peers' sides is not our problem.
    if ! python3 "$ROOT/scripts/run_helpers.py" port-open "$p"; then
      PORT="$p"; break
    fi
  done
fi
[[ -n "${PORT:-}" ]] || die "no free port starting at $DGPP_FABRIC_PORT (pass --port)"

# ------------------------------------------------------------- rank 0
echo "fabric_run: world $WORLD, port $PORT, logs in $LOG_DIR"

# Per-node helpers start BEFORE the ranks so the probe's first sample is
# the pre-load baseline. Rank 0's helpers write next to r0.log; peers'
# into $PEER_DIR (fetched with --fetch-logs). Record each node's offset from the head
# (peer_now - head_now, seconds; half an ssh round trip of error, well
# under the probes' 1 s grain) so fabric_xrank.py can line probe rows up
# with rank 0's log.
record_clock_offsets() {
  : > "$LOG_DIR/clock_offsets.txt"
  for i in "${!ALL_NODES[@]}"; do
    ip="${ALL_NODES[$i]}"
    local t0 t1 peer
    t0=$(date -u +%s.%N)
    peer=$( [[ $i -eq 0 ]] && date -u +%s.%N || peer_ssh "$ip" "date -u +%s.%N" ) || peer=""
    t1=$(date -u +%s.%N)
    [[ -n "$peer" ]] || { echo "r$i ?" >> "$LOG_DIR/clock_offsets.txt"; continue; }
    awk -v r="$i" -v p="$peer" -v a="$t0" -v b="$t1" \
      'BEGIN { printf "r%s %.3f\n", r, p - (a + b) / 2 }' >> "$LOG_DIR/clock_offsets.txt"
  done
}

start_node_helpers() {
  [[ $NODE_PROBE -eq 1 ]] && record_clock_offsets
  for i in "${!ALL_NODES[@]}"; do
    ip="${ALL_NODES[$i]}"
    local dir="$PEER_DIR" runner="peer_ssh $ip" helper="cluster_process.py" probe="node_probe.sh"
    [[ $i -eq 0 ]] && {
      dir=$(printf '%q' "$LOG_DIR"); runner="bash -c"
      helper=$(printf '%q' "$PROCESS_HELPER"); probe=$(printf '%q' "$PROBE_SCRIPT")
    }
    if [[ $NODE_PROBE -eq 1 ]]; then
      $runner "cd $dir && python3 $helper launch --state probe.process.json --log probe_r$i.log --cwd . -- bash $probe $HELPER_TAG" || true
    fi
    if [[ -n "$NODE_CMD" ]]; then
      # sh -c runs the caller's command with the run tag as its $0.
      $runner "cd $dir && python3 $helper launch --state nodecmd.process.json --log nodecmd_r$i.log --cwd . -- sh -c $(printf '%q' "$NODE_CMD") $HELPER_TAG" || true
    fi
  done
}
start_node_helpers

: > "$LOG_DIR/r0.log"

dgpp_run_rank 0 python3 "$PROCESS_HELPER" run --state "$STATE" --log "$LOG_DIR/r0.log" --cwd "$PWD" -- \
  "${HEAD_WRAP[@]}" "$APP" "${APP_ARGS[@]}" --world "$WORLD" --rank 0 --port "$PORT" &
HEAD_PID=$!
# The helper waits for the recorded process and preserves its exit status.

for _ in $(seq 1 60); do
  grep -q "rendezvous listening" "$LOG_DIR/r0.log" 2>/dev/null && break
  kill -0 "$HEAD_PID" 2>/dev/null \
    || { tail -5 "$LOG_DIR/r0.log"; die "rank 0 died before listening"; }
  sleep 0.5
done
grep -q "rendezvous listening" "$LOG_DIR/r0.log" \
  || die "rank 0 never listened (30s) — see $LOG_DIR/r0.log"
echo "fabric_run: rank 0 listening on :$PORT"

# ------------------------------------------------------- peers 1..N-1
# The peers' args: verbatim, except a staged --requests / --teacher-file
# path points at the peer's copy (rank 0 reads the local file — identical
# bytes, and the app logs the file's hash on every rank as the identity
# check).
REMOTE_ARGS=""
prev=""
for a in "${APP_ARGS[@]}"; do
  if [[ -n "$STAGE_FILE" && ( "$prev" == "--requests" || "$prev" == "--teacher-file" ) ]]; then
    REMOTE_ARGS+="$(printf '%q ' "$PEER_DIR/$(basename "$STAGE_FILE")")"
  else
    REMOTE_ARGS+="$(printf '%q ' "$a")"
  fi
  prev="$a"
done
# Every DGPP_* knob in our environment reaches the peers too (the app's
# knobs: DGPP_MLOCK, DGPP_RESIDENT_CACHE*, DGPP_LOG_LEVEL, ...) — except
# this script's own, which describe the head's side of the world.
REMOTE_ENV=""
while IFS='=' read -r name value; do
  case "$name" in
    DGPP_ENV_FILE|DGPP_CLUSTER_CONFIG|DGPP_NODES|DGPP_NODE_OVERRIDES|DGPP_ROCE_*|DGPP_HTTP_*|DGPP_RESIDENT_CACHE_DIR|DGPP_DATA_DIR|DGPP_SSH_USER|DGPP_JOURNAL_PORT|DGPP_LOG_DIR|DGPP_STAGE_DIR|DGPP_RELEASE_DIR|DGPP_BUILD_DIR|DGPP_FABRIC_*) continue ;;
    DGPP_*) REMOTE_ENV+="$name=$(printf '%q' "$value") " ;;
  esac
done < <(env)
for i in "${!FABRIC_PEERS[@]}"; do
  rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
  RANK_ENV=$(dgpp_rank_prefix "$rank") || exit 1
  # Fire-and-forget on purpose (see header): the remote side is fully
  # detached; whether THIS ssh returns is irrelevant to the launch.
  ( timeout 25 ssh "${SSH_OPTS[@]}" "$FABRIC_USER@$ip" \
      "cd $PEER_DIR && $REMOTE_ENV $RANK_ENV python3 cluster_process.py launch --state fabric_${RUN_ID}_rank$rank.process.json --log fabric_r$rank.log --cwd . -- ./$APP_NAME $REMOTE_ARGS \
       --world $WORLD --rank $rank --peer $FABRIC_HEAD --port $PORT \
       " \
      >/dev/null 2>&1 ) &
done
for i in "${!FABRIC_PEERS[@]}"; do
  rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
  up=0
  for _ in $(seq 1 20); do
    peer_ssh "$ip" "python3 $PEER_DIR/cluster_process.py status --state $PEER_DIR/fabric_${RUN_ID}_rank$rank.process.json" >/dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  if [[ $up -ne 1 ]]; then
    peer_ssh "$ip" "tail -5 $PEER_DIR/fabric_r$rank.log" || true
    kill_all
    die "peer rank $rank ($ip) never started (20s) — everything killed"
  fi
  echo "fabric_run: rank $rank up on $ip"
done

# ------------------------------------------------------------- monitor
elapsed=0
while kill -0 "$HEAD_PID" 2>/dev/null; do
  if [[ $MONITOR_TIMEOUT -gt 0 && $elapsed -ge $MONITOR_TIMEOUT ]]; then
    kill_all
    die "watchdog: rank 0 still running after ${MONITOR_TIMEOUT}s — everything killed; logs in $LOG_DIR"
  fi
  sleep 10; elapsed=$((elapsed + 10))
done
HEAD_RC=0; wait "$HEAD_PID" || HEAD_RC=$?

# A failed head can leave peers waiting in collectives. Stop those recorded
# processes before collecting their logs.
if [[ $HEAD_RC -ne 0 ]]; then
  echo "fabric_run: head failed — killing wedged peers"
  kill_all
fi
stop_node_helpers

# ------------------------------------------------------------ collect
echo "fabric_run: rank 0 exited rc=$HEAD_RC — collecting verdicts"
# md5 over EVERY "generated ids" line (scheduler mode logs one per
# request, in deterministic order) — the payload after the colon, so the
# embedded rank prefix cannot differ. Missing lines mean failure.
md5_of_generated() { grep "generated ids" "$1" | sed 's/^.*generated ids: //' | md5sum | awk '{print $1}'; }
head_md5="$(md5_of_generated "$LOG_DIR/r0.log" || true)"
echo "  head : ${head_md5:-<no generated line>}"
for i in "${!FABRIC_PEERS[@]}"; do
  rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
  peer_md5="$(peer_ssh "$ip" \
    "grep 'generated ids' $PEER_DIR/fabric_r$rank.log | sed 's/^.*generated ids: //' | md5sum | awk '{print \$1}'" 2>/dev/null || true)"
  echo "  rank $rank: ${peer_md5:-<no generated line>}"
  [[ -n "$peer_md5" && "$peer_md5" == "$head_md5" ]] || HEAD_RC=1
done
echo "fabric_run: rank consistency $([[ -n "$head_md5" ]] && echo "$head_md5" || echo 'n/a') — $([[ $HEAD_RC -eq 0 ]] && echo 'ALL RANKS IDENTICAL' || echo 'MISMATCH or missing')"
grep -m1 "eos stop" "$LOG_DIR/r0.log" || true
grep "retired (" "$LOG_DIR/r0.log" || true
grep -m1 -E "steps in|step cap in" "$LOG_DIR/r0.log" || true

if [[ $FETCH_LOGS -eq 1 ]]; then
  for i in "${!FABRIC_PEERS[@]}"; do
    rank=$((i + 1)); ip="${FABRIC_PEERS[$i]}"
    scp -q -o BatchMode=yes -o ConnectTimeout=8 \
      "$FABRIC_USER@$ip:$PEER_DIR/fabric_r$rank.log" "$LOG_DIR/r$rank.log" \
      || echo "fabric_run: could not fetch rank $rank's log"
    # One scp per file: OpenSSH's sftp-backed scp expands no remote globs.
    [[ $NODE_PROBE -eq 1 ]] && scp -q -o BatchMode=yes -o ConnectTimeout=8 \
        "$FABRIC_USER@$ip:$PEER_DIR/probe_r$rank.log" "$LOG_DIR/" 2>/dev/null || true
    [[ -n "$NODE_CMD" ]] && scp -q -o BatchMode=yes -o ConnectTimeout=8 \
        "$FABRIC_USER@$ip:$PEER_DIR/nodecmd_r$rank.log" "$LOG_DIR/" 2>/dev/null || true
  done
  # The cross-rank verdict the hunt lives on: per-rank step distribution,
  # inter-step host gaps, and every gen-0 stall with all ranks' views.
  if [[ -x "$ROOT/scripts/fabric_xrank.py" ]]; then
    python3 "$ROOT/scripts/fabric_xrank.py" "$LOG_DIR" --summary || true
  fi
fi
echo "fabric_run: full logs in $LOG_DIR $([[ $FETCH_LOGS -eq 1 ]] && echo '(r0..rN.log fetched)' || echo '(r0.log local, fabric_rN.log on peers)')"
exit "$HEAD_RC"
