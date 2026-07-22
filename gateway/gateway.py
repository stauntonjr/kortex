"""
Kortex Dynamic Model Gateway
==============================
OpenAI-compatible proxy daemon running at http://localhost:8080/v1.

Responsibilities:
  • Maintain a registry of local vLLM model instances with VRAM profiles.
  • Route /v1/{path} requests to the correct backend port.
  • Hot-swap models on demand: spin down conflicting exclusive models before
    launching a newly requested one, and restart them when no longer needed.
  • Expose /health and /v1/models endpoints for IDE / agent introspection.

Run::

    uvicorn gateway.gateway:app --host 0.0.0.0 --port 8080 --log-level info

Environment variables (all optional, sensible defaults shown):
    GATEWAY_POLL_INTERVAL   — seconds between health-poll cycles (default: 15)
    GATEWAY_STARTUP_TIMEOUT — seconds to wait for a model to become ready (default: 180)
    REDIS_URL               — Redis connection URL (default: redis://localhost:6379/0)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

POLL_INTERVAL   = int(os.getenv("GATEWAY_POLL_INTERVAL",   "15"))
STARTUP_TIMEOUT = int(os.getenv("GATEWAY_STARTUP_TIMEOUT", "180"))
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Forwarded headers that must not be re-sent to the backend
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    ]
)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
# Each entry describes a local vLLM instance managed by `sparkrun`.
#
# Fields:
#   port        — TCP port the vLLM server listens on
#   vram_gb     — estimated VRAM footprint in GB
#   exclusive   — if True, all other exclusive models must be stopped first
#   sparkrun_args — CLI arguments forwarded verbatim to `sparkrun`
#   aliases     — additional model-name strings routed to this backend
#   process     — live subprocess.Popen handle (None when stopped)
#   healthy     — last known health state
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "embedding": {
        "port":      8001,
        "vram_gb":   19.7,
        "exclusive": False,
        "aliases":   ["qwen-vl-embedding", "Qwen/Qwen2.5-VL-7B-Instruct"],
        "sparkrun_args": [
            "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
            "--port",  "8001",
            "--backend", "vllm-ray",
            "--tensor-parallel", "1",
            "--task", "embedding",
            "--trust-remote-code",
        ],
        "process": None,
        "healthy": False,
    },
    "coder": {
        "port":      8002,
        "vram_gb":   52.5,
        "exclusive": False,
        "aliases":   ["qwen-coder", "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"],
        "sparkrun_args": [
            "--model", "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
            "--port",  "8002",
            "--backend", "vllm-ray",
            "--tensor-parallel", "1",
            "--quantization", "gptq",
            "--trust-remote-code",
        ],
        "process": None,
        "healthy": False,
    },
    "reasoning": {
        "port":      8003,
        "vram_gb":   21.8,
        "exclusive": True,
        "aliases":   ["qwen-35b", "Qwen/Qwen3-30B-A3B-FP8"],
        "sparkrun_args": [
            "--model", "Qwen/Qwen3-30B-A3B-FP8",
            "--port",  "8003",
            "--backend", "vllm-ray",
            "--tensor-parallel", "1",
            "--trust-remote-code",
        ],
        "process": None,
        "healthy": False,
    },
    "planning": {
        "port":      8000,
        "vram_gb":   90.0,
        "exclusive": True,
        "aliases":   [
            "nemotron",
            "nemotron-120b",
            "nvidia/Llama-3.1-Nemotron-Ultra-253B-v1",
        ],
        "sparkrun_args": [
            "--model", "nvidia/Llama-3.1-Nemotron-Ultra-253B-v1",
            "--port",  "8000",
            "--backend", "vllm-ray",
            "--tensor-parallel", "1",
            "--quantization", "nvfp4",
            "--trust-remote-code",
        ],
        "process": None,
        "healthy": False,
    },
}

# Reverse alias lookup: alias → canonical model key
_ALIAS_MAP: dict[str, str] = {}
for _key, _cfg in MODEL_REGISTRY.items():
    _ALIAS_MAP[_key] = _key
    for _alias in _cfg.get("aliases", []):
        _ALIAS_MAP[_alias] = _key


def resolve_model(model_name: str) -> str | None:
    """Return the canonical registry key for *model_name*, or None."""
    return _ALIAS_MAP.get(model_name)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _backend_url(key: str) -> str:
    return f"http://127.0.0.1:{MODEL_REGISTRY[key]['port']}"


async def _is_healthy(key: str) -> bool:
    cfg = MODEL_REGISTRY[key]
    url = f"{_backend_url(key)}/health"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(url)
            healthy = resp.status_code == 200
    except Exception:
        healthy = False
    cfg["healthy"] = healthy
    return healthy


async def _wait_healthy(key: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await _is_healthy(key):
            logger.info("Model '%s' is healthy.", key)
            return True
        await asyncio.sleep(5)
    logger.error("Model '%s' did not become healthy within %ds.", key, timeout)
    return False


def _launch_process(key: str) -> None:
    cfg = MODEL_REGISTRY[key]
    if cfg["process"] is not None:
        logger.debug("Model '%s' already has a process — skipping launch.", key)
        return
    log_dir = "/tmp/kortex/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(f"{log_dir}/{key}.log", "a")  # noqa: WPS515
    cmd = ["sparkrun"] + cfg["sparkrun_args"]
    logger.info("Launching model '%s': %s", key, " ".join(cmd))
    cfg["process"] = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )


def _stop_process(key: str) -> None:
    cfg = MODEL_REGISTRY[key]
    proc = cfg.get("process")
    if proc is None:
        return
    logger.info("Stopping model '%s' (PID %s).", key, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    cfg["process"] = None
    cfg["healthy"] = False
    logger.info("Model '%s' stopped.", key)


async def ensure_model_online(key: str) -> None:
    """
    Bring model *key* online, evicting conflicting exclusive models first.

    For non-exclusive (background) models this is always safe.
    For exclusive models we must stop all other exclusive models whose VRAM
    would push us over the 121 GB limit.
    """
    cfg = MODEL_REGISTRY[key]

    if cfg.get("healthy") and cfg.get("process") is not None:
        return  # Already running

    # If this model is exclusive, stop other exclusive models first
    if cfg["exclusive"]:
        for other_key, other_cfg in MODEL_REGISTRY.items():
            if other_key == key:
                continue
            if other_cfg["exclusive"] and other_cfg.get("process") is not None:
                logger.info(
                    "Evicting exclusive model '%s' to free VRAM for '%s'.",
                    other_key,
                    key,
                )
                _stop_process(other_key)

    _launch_process(key)
    ok = await _wait_healthy(key)
    if not ok:
        raise RuntimeError(f"Model '{key}' failed to start within {STARTUP_TIMEOUT}s.")


# ---------------------------------------------------------------------------
# Background health-poll task
# ---------------------------------------------------------------------------

async def _health_poll_loop() -> None:
    """Periodically probe all running model processes and update health flags."""
    while True:
        for key, cfg in MODEL_REGISTRY.items():
            if cfg.get("process") is not None:
                await _is_healthy(key)
        await asyncio.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background health-polling task
    poll_task = asyncio.create_task(_health_poll_loop())
    yield
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kortex Model Gateway",
    description="OpenAI-compatible proxy with dynamic VRAM management.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# /health — gateway-level liveness probe
# ---------------------------------------------------------------------------

@app.get("/health")
async def gateway_health() -> dict[str, Any]:
    statuses = {
        key: {
            "healthy": cfg["healthy"],
            "port":    cfg["port"],
            "vram_gb": cfg["vram_gb"],
            "running": cfg["process"] is not None,
        }
        for key, cfg in MODEL_REGISTRY.items()
    }
    return {"status": "ok", "models": statuses}


# ---------------------------------------------------------------------------
# /v1/models — list available models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    model_list = []
    for key, cfg in MODEL_REGISTRY.items():
        model_list.append(
            {
                "id":       key,
                "object":   "model",
                "owned_by": "kortex",
                "created":  0,
                "aliases":  cfg.get("aliases", []),
                "vram_gb":  cfg["vram_gb"],
                "exclusive": cfg["exclusive"],
                "healthy":  cfg["healthy"],
            }
        )
    return {"object": "list", "data": model_list}


# ---------------------------------------------------------------------------
# /v1/{path:path} — universal OpenAI-compatible proxy
# ---------------------------------------------------------------------------

async def _extract_model_key(request: Request) -> str | None:
    """
    Attempt to determine which model the request targets.

    Strategy (in order):
      1. Parse ``model`` field from JSON body (chat/completion requests).
      2. Check ``X-Kortex-Model`` custom header.
      3. Fall back to ``coder`` (the always-on default).
    """
    model_name: str | None = None

    # Try JSON body
    try:
        body = await request.json()
        model_name = body.get("model")
    except Exception:
        pass

    # Try custom header
    if not model_name:
        model_name = request.headers.get("x-kortex-model")

    if model_name:
        key = resolve_model(model_name)
        if key:
            return key
        logger.warning("Unknown model '%s'; falling back to 'coder'.", model_name)

    return "coder"


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
)
async def proxy(path: str, request: Request) -> Response:
    """
    Forward the request to the appropriate backend, starting the model first
    if it is not yet running.
    """
    model_key = await _extract_model_key(request)

    # Bring the model online (may evict conflicting exclusive models)
    try:
        await ensure_model_online(model_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    backend = _backend_url(model_key)
    url = f"{backend}/v1/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    # Filter hop-by-hop headers
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    body = await request.body()

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            backend_resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Backend for model '{model_key}' is not reachable: {exc}",
            ) from exc

    # Detect streaming responses and forward them
    content_type = backend_resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        async def _stream():
            async for chunk in backend_resp.aiter_bytes():
                yield chunk

        return StreamingResponse(
            _stream(),
            status_code=backend_resp.status_code,
            headers=dict(backend_resp.headers),
            media_type=content_type,
        )

    return Response(
        content=backend_resp.content,
        status_code=backend_resp.status_code,
        headers=dict(backend_resp.headers),
        media_type=content_type,
    )
