#!/bin/bash
# GPU smoke test for helios: stops the user's exllamav3 server, runs the engine, restarts the
# server no matter what (trap), logs everything.
set -u
LOG=/tmp/helios_gpu_test.log
HELIOS=/home/neron/projects/new_engine/helios
MODEL=/home/neron/models/glm53flash
SRV=/home/neron/models/qwen38_next_server.sh
export CUDA_VISIBLE_DEVICES=0,1

restart_server() {
  echo "--- restarting exllamav3 server $(date -Is)" >>"$LOG"
  "$SRV" start >>"$LOG" 2>&1 || true
  sleep 20
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >>"$LOG" 2>&1
  curl -s -m 5 http://127.0.0.1:8080/health >>"$LOG" 2>&1 || echo " (server health check failed)" >>"$LOG"
  echo "=== server restarted $(date -Is) ===" >>"$LOG"
}
trap restart_server EXIT INT TERM

{
  echo "=== helios GPU test start $(date -Is) ==="
  echo "--- stopping exllamav3 server"
  "$SRV" stop 2>&1 || true
  sleep 8
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

  echo "--- phase 1: full GPU load"
  cd "$HELIOS" || exit 1
  timeout 1800 ./build/helios load "$MODEL" 2>&1
  echo "load exit=$?"

  echo "--- phase 2: greedy generation (32 tokens)"
  timeout 2400 ./build/helios gen "$MODEL" --cap 32768 --chunk 128 --tokens 32 --temp 0 \
      --prompt "What is 2+2? Answer briefly." 2>&1
  echo "gen exit=$?"

  echo "--- phase 3: short prefill benchmark"
  timeout 2400 ./build/helios gen "$MODEL" --cap 32768 --chunk 256 --tokens 8 --temp 0 \
      --prompt "Explain in one sentence why the sky is blue." 2>&1
  echo "gen2 exit=$?"
} >>"$LOG" 2>&1
RC=$?
echo "=== helios GPU test done $(date -Is) rc=$RC ===" >>"$LOG"

trap - EXIT
restart_server