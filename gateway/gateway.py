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
import atexit
import contextlib
import logging
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
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
    "qwen-embedding": {
        "port":      8001,
        "vram_gb":   19.7,
        "exclusive": False,
        "recipe":    "@official/qwen3-vl-embedding-8b-vllm",
        "args": ["--tensor-parallel", "1", "--port", "8001"],
        "process": None,
        "healthy": False,
    },
    "qwen-coder": {
        "port":      8002,
        "vram_gb":   52.5,
        "exclusive": False,
        "recipe":    "@official/qwen3-coder-next-int4-autoround-vllm",
        "args": ["--tensor-parallel", "1", "--port", "8002"],
        "process": None,
        "healthy": False,
    },
    "qwen-35b": {
        "port":      8003,
        "vram_gb":   21.8,
        "exclusive": True,
        "recipe":    "@eugr/qwen3.6-35b-a3b-nvfp4",
        "args": ["--tensor-parallel", "1", "--port", "8003"],
        "process": None,
        "healthy": False,
    },
    "nemotron-120b": {
        "port":      8000,
        "vram_gb":   90.0,
        "exclusive": True,
        "recipe":    "@eugr/nemotron-3-super-nvfp4",
        "aliases":   ["nemotron"],
        "args": ["--tensor-parallel", "1", "--port", "8000"],
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


# Lock that serialises VRAM-exclusive model swap operations so concurrent
# agent invocations cannot race to evict/launch exclusive models simultaneously.
_SWAP_LOCK = asyncio.Lock()


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
    log_file = open(f"{log_dir}/{key}.log", "a")  # noqa: SIM115
    cfg["_log_file"] = log_file
    cmd = ["sparkrun", "run", cfg["recipe"]] + cfg["args"]
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
    # Close the log file handle that was opened during launch
    log_file = cfg.pop("_log_file", None)
    if log_file is not None:
        log_file.close()
    logger.info("Model '%s' stopped.", key)


async def ensure_model_online(key: str) -> None:
    """
    Bring model *key* online, evicting conflicting exclusive models first.

    An asyncio.Lock serialises concurrent calls so that only one VRAM swap
    runs at a time, preventing race conditions under concurrent agent load.
    """
    async with _SWAP_LOCK:
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
# Process cleanup — atexit and signal handlers
# ---------------------------------------------------------------------------

def _shutdown_all_models() -> None:
    """Stop every running sparkrun subprocess. Safe to call multiple times."""
    for key in list(MODEL_REGISTRY):
        with contextlib.suppress(Exception):
            _stop_process(key)


atexit.register(_shutdown_all_models)


def _install_signal_handlers() -> None:
    """Chain SIGINT/SIGTERM so sparkrun children are always reaped on exit."""
    _prev: dict[int, Any] = {}

    def _handler(signum: int, frame: object) -> None:
        logger.info("Signal %d received; shutting down all model processes.", signum)
        _shutdown_all_models()
        prev = _prev.get(signum)
        if callable(prev):
            prev(signum, frame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        _prev[sig] = signal.getsignal(sig)
        signal.signal(sig, _handler)


_install_signal_handlers()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background health-polling task
    poll_task = asyncio.create_task(_health_poll_loop())
    try:
        yield
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        # Ensure all model processes are stopped on graceful shutdown
        _shutdown_all_models()


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
        logger.warning("Unknown model '%s'; falling back to 'qwen-coder'.", model_name)

    return "qwen-coder"


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


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the Kortex gateway with uvicorn (used by the kortex-gateway script)."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "gateway.gateway:app",
        host="0.0.0.0",
        port=int(os.getenv("GATEWAY_PORT", "8080")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
