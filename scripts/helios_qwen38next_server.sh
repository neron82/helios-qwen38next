#!/usr/bin/env bash
# Start, stop, restart, or inspect the Helios Qwen3.8-Flash-Next server.
#
# Commands: start | stop | restart | status, plus --host / --port / --cap /
# --chunk / --api-key / --reasoning-effort / --no-mtp and environment overrides.
# Binds 0.0.0.0 by default so LAN clients can reach it; the readiness probe
# always goes to loopback.
set -euo pipefail

HELIOS_DIR=${HELIOS_DIR:-$HOME/projects/new_engine/helios-qwen}
BIN=${BIN:-$HELIOS_DIR/build/helios}

MODEL_DIR=${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-exl3}
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
# KV capacity in tokens. Qwen3.8-Flash-Next's config max_position_embeddings is
# 262144, which is also the engine's own default --cap, so this matches the
# model's native window rather than exceeding it. The cache is allocated up
# front, so this is what GPU0 must be able to hold; the script waits for that
# and says so rather than aborting inside the allocator.
#
# Measured at cap 262144 / chunk 1024 (startup banner, this machine):
#   12 full-attention layers x 128 MB KV = 6.00 GB, i.e. ~24 KiB per token of
#   context, on top of ~15.8 GB of fixed trunk allocation. That puts GPU0 at
#   ~22.2 GB used and leaves the cards very tight, so lowering CAP is the first
#   thing to try if something else needs to share a card.
CAP=${CAP:-262144}
# Prefill batch size, clamped to the prompt length, so a large value only costs
# scratch memory and makes long prompts faster. 1024 is both the engine's own
# default and the value every number in this file was measured at, so it is the
# default here too. Raising it trades VRAM for long-prompt speed; see the
# GPU0_NEED_MIB fit below for what that costs.
CHUNK=${CHUNK:-1024}
# Empty = no auth. Set for anything reachable beyond localhost.
API_KEY=${API_KEY:-}
# Default reasoning effort for requests that do not set one. Qwen3.8-Flash-Next's
# template takes xhigh | medium | low and rejects anything else, falling back to
# xhigh; the engine's own default is xhigh. Unset here leaves that in place.
REASONING_EFFORT=${REASONING_EFFORT:-}
# Multi-token prediction (speculative decoding). ON by default, and it pays for
# itself on this model: measured greedy decode at 8k context, 256 tokens,
#   MTP off (HELIOS_MTP=0) : 57.12 tok/s
#   MTP on, HELIOS_SPEC_K=1: 64.59 tok/s   (+13.1%)
# Deeper drafts are much worse - K=2 41.26, K=3 42.46, K=4 37.95 tok/s - so leave
# SPEC_K at its default of 1. A verify row costs more than the extra tokens save.
# Unlike the GLM-5.3 build, where speculation measured a net loss.
#
# The trade-off: a batched verify forward reduces differently from a width-1
# one, so speculative greedy text drifts off the non-speculative stream after a
# few dozen tokens. Pass --no-mtp for bit-exact greedy output; it also re-enables
# two-sequence generation (--pair), which is refused while MTP is on.
MTP=${MTP:-1}
# Cross-request prefix caching. OFF by default (PREFIX_CACHE=1, or
# HELIOS_PREFIX_CACHE=1 in the environment, turns it on). There is no CLI flag
# for it. Snapshots live in pinned HOST memory, so they cost no VRAM, and the
# ring is partitioned per sequence slot - a conversation can only resume from
# its own captures. The snapshot interval is --chunk (default 1024) and the ring
# depth is HELIOS_PREFIX_SLOTS (default 8), so a lower --chunk means less
# recompute after a mid-history divergence. It refuses to run with HELIOS_QSA,
# and disables itself if the pinned allocation fails.
PREFIX_CACHE=${PREFIX_CACHE:-0}
# Default output length for requests that omit max_tokens. The engine has no
# other output cap.
MAX_TOKENS=${MAX_TOKENS:-32768}

SERVER_LOCK_FILE=${SERVER_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-qwen-port-${PORT}.lock}
# Loading the weights and allocating the cache takes ~20-40 s on this machine.
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-300}
# The driver keeps a dead context's VRAM for a few seconds after the process
# exits, so a stop-then-start can otherwise fail outright or silently shrink the
# expert pool. Wait up to VRAM_WAIT seconds for both cards to have what they need.
#
# The two cards are not symmetric here and must be checked separately:
#   GPU0  trunk + KV cache. At cap 262144 / chunk 1024 the engine measured
#         22.2 GB used on this card, i.e. it wants essentially all of a 24 GB
#         card. The figure below is fitted to that single measured point - one
#         point, so treat the chunk term as an allowance rather than a
#         measurement - and decomposes it as: 15840 MiB of fixed trunk
#         allocation (the two [model] alloc dev0 lines, 14627 + 1212), plus KV
#         at 24 KiB per token of capacity (the banner's "12 full-attn layers x
#         128 MB = 6.00 GB" at 262144), plus a scratch allowance. It reproduces
#         the measured 22.2 GB to within ~120 MiB. The 512 MiB term is the CUDA
#         context: this check reads nvidia-smi's "free" while the fit is
#         against cudaMemGetInfo's "used", and the two differ by exactly that.
#   GPU1  expert pool + routers + MTP draft. The pool auto-sizes from whatever
#         is free, so this is a floor rather than a prediction: below it the
#         engine still runs, with a smaller pool and proportionally slower
#         decode.
#
# If something else must share a card (a small model living alongside the
# engine), lower CAP or CHUNK until GPU0_NEED_MIB plus that model fits in 24 GB.
GPU0_NEED_MIB=${GPU0_NEED_MIB:-$(( 512 + 15840 + CHUNK / 8 + CAP * 24 / 1000 ))}
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
VRAM_WAIT=${VRAM_WAIT:-60}

COMMAND=start
while (($#)); do
    case "$1" in
        start|stop|restart|status) COMMAND=$1 ;;
        --host)
            (($# >= 2)) || { echo "--host requires an address" >&2; exit 2; }
            HOST=$2; shift ;;
        --host=*) HOST=${1#*=} ;;
        --port)
            (($# >= 2)) || { echo "--port requires a number" >&2; exit 2; }
            PORT=$2; shift ;;
        --port=*) PORT=${1#*=} ;;
        --cap)
            (($# >= 2)) || { echo "--cap requires a number" >&2; exit 2; }
            CAP=$2; shift ;;
        --cap=*) CAP=${1#*=} ;;
        --chunk)
            (($# >= 2)) || { echo "--chunk requires a number" >&2; exit 2; }
            CHUNK=$2; shift ;;
        --chunk=*) CHUNK=${1#*=} ;;
        --api-key)
            (($# >= 2)) || { echo "--api-key requires a value" >&2; exit 2; }
            API_KEY=$2; shift ;;
        --api-key=*) API_KEY=${1#*=} ;;
        --reasoning-effort)
            (($# >= 2)) || { echo "--reasoning-effort requires a value" >&2; exit 2; }
            REASONING_EFFORT=$2; shift ;;
        --reasoning-effort=*) REASONING_EFFORT=${1#*=} ;;
        --no-mtp) MTP=0 ;;
        -h|--help)
            echo "usage: $0 [start|stop|restart|status] [--host ADDR] [--port N] [--cap N] [--chunk N] [--api-key KEY] [--reasoning-effort L] [--no-mtp]"
            exit 0 ;;
        *)
            echo "usage: $0 [start|stop|restart|status] [--host ADDR] [--port N] [--cap N] [--chunk N] [--api-key KEY] [--reasoning-effort L] [--no-mtp]" >&2
            exit 2 ;;
    esac
    shift
done

# Derived from the resolved port, so --port moves them with it. An explicit
# PID_FILE/LOG_FILE in the environment still wins.
PID_FILE=${PID_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-qwen38next-${PORT}.pid}
LOG_FILE=${LOG_FILE:-${XDG_RUNTIME_DIR:-/tmp}/helios-qwen38next-${PORT}.log}

case "$HOST" in
    0.0.0.0) HEALTH_HOST=127.0.0.1 ;;
    ::) HEALTH_HOST=::1 ;;
    *) HEALTH_HOST=$HOST ;;
esac
if [[ "$HEALTH_HOST" == *:* ]]; then
    HEALTH_URL="http://[${HEALTH_HOST}]:${PORT}/health"
else
    HEALTH_URL="http://${HEALTH_HOST}:${PORT}/health"
fi

proc_start_time() {
    [[ -r "/proc/$1/stat" ]] || return 1
    awk '{print $22}' "/proc/$1/stat"
}

read_pid() {
    local pid start
    read -r pid start <"$PID_FILE"
    printf '%s\n' "$pid"
}

# A PID alone is not proof: it may have been recycled by an unrelated process.
# Re-verify the start time, the command line and the model before believing it.
is_running() {
    [[ -s "$PID_FILE" ]] || return 1
    local pid expected_start current_start cmdline
    read -r pid expected_start <"$PID_FILE" || return 1
    [[ "$pid" =~ ^[0-9]+$ && "$expected_start" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    current_start=$(proc_start_time "$pid") || return 1
    [[ "$current_start" == "$expected_start" ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    [[ "$cmdline" == *"$BIN"* && "$cmdline" == *"serve"* && "$cmdline" == *"$MODEL_DIR"* ]]
}

pid_start_line() {
    local pid=$1 start
    start=$(proc_start_time "$pid") || {
        echo "server process exited before its PID could be recorded" >&2
        return 1
    }
    printf '%s %s\n' "$pid" "$start"
}

# The two cards need different things, so check them separately: GPU0's figure
# follows --cap and --chunk (GPU0_NEED_MIB) while GPU1 only has a floor.
wait_vram() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local deadline=$((SECONDS + VRAM_WAIT)) f0 f1
    while :; do
        # `tr` leaves no trailing newline, so this read always reports EOF - which `set -e` would
        # treat as a failure and abort the script before a single line of output.
        read -r f0 f1 < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ') || true
        if [[ -n "${f0:-}" && -n "${f1:-}" ]] && (( f0 >= GPU0_NEED_MIB && f1 >= MIN_FREE_MIB )); then
            return 0
        fi
        if (( SECONDS >= deadline )); then
            echo "warning: after ${VRAM_WAIT}s GPU0 has ${f0:-?} MiB free (cap $CAP / chunk $CHUNK needs $GPU0_NEED_MIB) and GPU1 has ${f1:-?} MiB (wants $MIN_FREE_MIB) - starting anyway" >&2
            return 0
        fi
        sleep 2
    done
}

wait_ready() {
    local pid=$1
    for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
        kill -0 "$pid" 2>/dev/null || return 1
        if curl --fail --silent --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

if command -v flock >/dev/null 2>&1; then
    exec 9>"$SERVER_LOCK_FILE"
    if ! flock -n 9; then
        echo "another helios-qwen server command is already running" >&2
        exit 1
    fi
fi

start() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required for the server readiness probe" >&2
        return 1
    fi
    if [[ ! -x "$BIN" ]]; then
        echo "engine binary not found or not executable: $BIN" >&2
        echo "build it with: cmake -B $HELIOS_DIR/build -G Ninja -S $HELIOS_DIR && cmake --build $HELIOS_DIR/build" >&2
        return 1
    fi
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "model directory not found: $MODEL_DIR" >&2
        return 1
    fi
    if is_running; then
        echo "helios-qwen server is already running ($(read_pid))"
        return 0
    fi
    # Never start a second server on the same port - most likely the other
    # model's server (exllamav3's qwen38_next_server.sh) is already up.
    if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
        echo "port $PORT is already in use - stop the other server first" >&2
        return 1
    fi
    wait_vram
    rm -f "$PID_FILE"
    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
    # Each run gets a clean log; the previous one is kept as .1. Without this, a traceback from an
    # earlier run sits in the file and looks live to anyone tailing it.
    if [[ -s "$LOG_FILE" ]]; then
        mv -f "$LOG_FILE" "$LOG_FILE.1"
    fi

    local args=(serve "$MODEL_DIR" --host "$HOST" --port "$PORT" --cap "$CAP" --chunk "$CHUNK"
                --max-tokens "$MAX_TOKENS")
    [[ -n "$API_KEY" ]] && args+=(--api-key "$API_KEY")
    [[ -n "$REASONING_EFFORT" ]] && args+=(--reasoning-effort "$REASONING_EFFORT")
    [[ "$MTP" == "0" ]] && args+=(--no-mtp)

    # Export too, so /proc/<pid>/environ carries the identity the checks look for,
    # and so the prefix cache setting reaches the engine.
    MODEL_DIR="$MODEL_DIR" PORT="$PORT" HOST="$HOST" CAP="$CAP" CHUNK="$CHUNK" \
        REASONING_EFFORT="$REASONING_EFFORT" MTP="$MTP" MAX_TOKENS="$MAX_TOKENS" \
        HELIOS_PREFIX_CACHE="$PREFIX_CACHE" \
        nohup "$BIN" "${args[@]}" >>"$LOG_FILE" 2>&1 9>&- &
    local pid=$!
    pid_start_line "$pid" >"$PID_FILE"
    if ! wait_ready "$pid"; then
        echo "server failed readiness check; inspect $LOG_FILE" >&2
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        return 1
    fi
    echo "started helios-qwen server pid=$(read_pid) port=$PORT context=$CAP bound=$HOST"
    echo "log: $LOG_FILE"
}

stop() {
    if ! is_running; then
        # The wrapper PID can die while the engine keeps the GPUs: kill any
        # stray helios serve for THIS port before declaring it stopped.
        local stray
        stray=$(pgrep -f "$BIN serve.*--port $PORT" || true)
        if [[ -n "$stray" ]]; then
            echo "reaping stray helios serve process(es): $stray" >&2
            kill $stray 2>/dev/null || true
            sleep 2
            stray=$(pgrep -f "$BIN serve.*--port $PORT" || true)
            [[ -n "$stray" ]] && kill -KILL $stray 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "helios-qwen server is not running"
        return 0
    fi
    local pid
    pid=$(read_pid)
    kill "$pid"
    for _ in {1..30}; do
        if ! is_running; then
            rm -f "$PID_FILE"
            echo "stopped helios-qwen server"
            return 0
        fi
        sleep 1
    done
    if is_running; then
        echo "server did not stop gracefully; sending SIGKILL" >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
}

status() {
    if is_running; then
        echo "helios-qwen server is running ($(read_pid)) model=$MODEL_DIR bound=${HOST:-0.0.0.0}:$PORT cap=$CAP mtp=$MTP"
        if command -v curl >/dev/null 2>&1; then
            curl --fail --silent "$HEALTH_URL" || true
            echo
        fi
    else
        echo "helios-qwen server is stopped"
        return 1
    fi
}

case "$COMMAND" in
    start) start ;;
    stop) stop ;;
    restart) stop || true; start ;;
    status) status ;;
esac
