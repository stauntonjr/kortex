#!/usr/bin/env bash
# launch_models.sh — Boot core always-on models for Kortex.
#
# Models started here are designed to co-exist within VRAM budget:
#   Qwen2.5-VL-7B  ~19.7 GB  (port 8001) — GraphRAG vector indexing
#   Qwen2.5-Coder-32B-Instruct-GPTQ-Int4  ~52.5 GB  (port 8002) — Tab-autocomplete & agent loop
#
# Total baseline VRAM: ~72.2 GB  (leaves ~48.8 GB headroom on 121 GB node)
#
# Usage:
#   ./scripts/launch_models.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

LOG_DIR="${LOG_DIR:-/tmp/kortex/logs}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

run_cmd() {
  if $DRY_RUN; then
    log "[DRY-RUN] $*"
  else
    log "Launching: $*"
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
EMBEDDING_VRAM="19.7"

CODER_MODEL="${CODER_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4}"
CODER_PORT="${CODER_PORT:-8002}"
CODER_VRAM="52.5"

# ---------------------------------------------------------------------------
# Helper: check if a sparkrun / vLLM process is already listening on a port
# ---------------------------------------------------------------------------

port_in_use() {
  local port="$1"
  ss -tlnp 2>/dev/null | grep -q ":${port} " || \
    nc -z 127.0.0.1 "$port" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Launch Qwen3-VL-Embedding (port 8001) — always-on embedding engine
# ---------------------------------------------------------------------------

if port_in_use "$EMBEDDING_PORT"; then
  log "Embedding engine already running on port $EMBEDDING_PORT — skipping."
else
  log "Starting Qwen3-VL-Embedding (~${EMBEDDING_VRAM} GB VRAM) on port $EMBEDDING_PORT …"
  run_cmd sparkrun \
    --model "$EMBEDDING_MODEL" \
    --port  "$EMBEDDING_PORT" \
    --backend vllm-ray \
    --tensor-parallel 1 \
    --task embedding \
    --trust-remote-code \
    >> "$LOG_DIR/embedding.log" 2>&1 &

  log "Embedding engine PID $! — log: $LOG_DIR/embedding.log"
fi

# ---------------------------------------------------------------------------
# Launch Qwen3-Coder-Next Int4 (port 8002) — always-on coding engine
# ---------------------------------------------------------------------------

if port_in_use "$CODER_PORT"; then
  log "Coder engine already running on port $CODER_PORT — skipping."
else
  log "Starting Qwen3-Coder-Next Int4 (~${CODER_VRAM} GB VRAM) on port $CODER_PORT …"
  run_cmd sparkrun \
    --model "$CODER_MODEL" \
    --port  "$CODER_PORT" \
    --backend vllm-ray \
    --tensor-parallel 1 \
    --quantization gptq \
    --trust-remote-code \
    >> "$LOG_DIR/coder.log" 2>&1 &

  log "Coder engine PID $! — log: $LOG_DIR/coder.log"
fi

# ---------------------------------------------------------------------------
# Wait for both services to become healthy (max 300 s each)
# ---------------------------------------------------------------------------

wait_healthy() {
  local name="$1" port="$2" timeout=300 elapsed=0
  log "Waiting for $name (port $port) to become healthy …"
  until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
    if (( elapsed >= timeout )); then
      log "ERROR: $name did not become healthy within ${timeout}s."
      return 1
    fi
    sleep 5
    elapsed=$(( elapsed + 5 ))
  done
  log "$name is healthy after ${elapsed}s."
}

if ! $DRY_RUN; then
  wait_healthy "Qwen3-VL-Embedding" "$EMBEDDING_PORT"
  wait_healthy "Qwen3-Coder-Next"   "$CODER_PORT"
  log "All baseline models are online."
else
  log "[DRY-RUN] Skipping health-wait."
fi
