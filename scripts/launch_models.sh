#!/usr/bin/env bash
# launch_models.sh — Boot core always-on models for Kortex.
#
# Models started here are designed to co-exist within VRAM budget:
#   qwen-embedding  ~19.7 GB  (port 8001) — GraphRAG vector indexing
#   qwen-coder      ~52.5 GB  (port 8002) — Tab-autocomplete & agent loop
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

EMBEDDING_RECIPE="${EMBEDDING_RECIPE:-@official/qwen3-vl-embedding-8b-vllm}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
EMBEDDING_VRAM="19.7"

CODER_RECIPE="${CODER_RECIPE:-@official/qwen3-coder-next-int4-autoround-vllm}"
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
# Launch qwen-embedding (port 8001) — always-on embedding engine
# ---------------------------------------------------------------------------

if port_in_use "$EMBEDDING_PORT"; then
  log "Embedding engine already running on port $EMBEDDING_PORT — skipping."
else
  log "Starting qwen-embedding (~${EMBEDDING_VRAM} GB VRAM) on port $EMBEDDING_PORT …"
  run_cmd sparkrun run "$EMBEDDING_RECIPE" \
    --tensor-parallel 1 \
    --port "$EMBEDDING_PORT" \
    >> "$LOG_DIR/qwen-embedding.log" 2>&1 &
  EMBEDDING_PID=$!
  log "Embedding engine PID $EMBEDDING_PID — log: $LOG_DIR/qwen-embedding.log"
fi

# ---------------------------------------------------------------------------
# Launch qwen-coder (port 8002) — always-on coding engine
# ---------------------------------------------------------------------------

if port_in_use "$CODER_PORT"; then
  log "Coder engine already running on port $CODER_PORT — skipping."
else
  log "Starting qwen-coder (~${CODER_VRAM} GB VRAM) on port $CODER_PORT …"
  run_cmd sparkrun run "$CODER_RECIPE" \
    --tensor-parallel 1 \
    --port "$CODER_PORT" \
    >> "$LOG_DIR/qwen-coder.log" 2>&1 &
  CODER_PID=$!
  log "Coder engine PID $CODER_PID — log: $LOG_DIR/qwen-coder.log"
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
  wait_healthy "qwen-embedding" "$EMBEDDING_PORT"
  wait_healthy "qwen-coder"     "$CODER_PORT"
  log "All baseline models are online."
else
  log "[DRY-RUN] Skipping health-wait."
fi
