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
import json
import logging
import os
import subprocess
import time
import atexit
import signal
import contextlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from kortex.contracts import GatewayResult, TranscriptTurn, TranscriptWritebackEvent
from memory.writeback import enqueue_writeback_event, start_writeback_worker, stop_writeback_worker

# TypeDB v3 driver imports (import only stable symbols here)
from typedb.driver import TypeDB, credentials_new

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.getenv("GATEWAY_POLL_INTERVAL", "15"))
STARTUP_TIMEOUT = int(os.getenv("GATEWAY_STARTUP_TIMEOUT", "180"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BACKEND_TIMEOUT = int(os.getenv("GATEWAY_BACKEND_TIMEOUT", "300"))
BACKEND_RETRIES = int(os.getenv("GATEWAY_BACKEND_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("GATEWAY_BACKOFF_BASE", "0.25"))
BACKOFF_MAX = float(os.getenv("GATEWAY_BACKOFF_MAX", "2.0"))
TYPEDB_ADDR = os.getenv("TYPEDB_ADDR", "typedb:1729")
TYPEDB_DATABASE = os.getenv("TYPEDB_DATABASE", "kortex")
TYPEDB_USERNAME = os.getenv("TYPEDB_USERNAME", "admin")
TYPEDB_PASSWORD = os.getenv("TYPEDB_PASSWORD", "password")

if BACKEND_RETRIES < 1:
    raise ValueError("GATEWAY_BACKEND_RETRIES environment variable must be at least 1")

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
        "port": 8001,
        "vram_gb": 19.7,
        "exclusive": False,
        "recipe": "@official/qwen3-vl-embedding-8b-vllm",
        "args": ["--tensor-parallel", "1", "--port", "8001"],
        "process": None,
        "healthy": False,
    },
    "qwen-coder": {
        "port": 8002,
        "vram_gb": 52.5,
        "exclusive": False,
        "recipe": "@official/qwen3-coder-next-int4-autoround-vllm",
        "args": ["--tensor-parallel", "1", "--port", "8002"],
        "process": None,
        "healthy": False,
    },
    "qwen-35b": {
        "port": 8003,
        "vram_gb": 21.8,
        "exclusive": True,
        "recipe": "@eugr/qwen3.6-35b-a3b-nvfp4",
        "args": ["--tensor-parallel", "1", "--port", "8003"],
        "process": None,
        "healthy": False,
    },
    "nemotron-120b": {
        "port": 8000,
        "vram_gb": 90.0,
        "exclusive": True,
        "recipe": "@eugr/nemotron-3-super-nvfp4",
        "aliases": ["nemotron"],
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


def _build_typedb_credentials() -> Any | None:
    """Construct TypeDB credentials using the installed TypeDB 3.x API."""
    try:
        return credentials_new(TYPEDB_USERNAME, TYPEDB_PASSWORD)
    except Exception:
        return None


def _process_is_running(proc: Any | None) -> bool:
    """Return True when *proc* should be treated as still running."""
    if proc is None:
        return False

    poll = getattr(proc, "poll", None)
    if not callable(poll):
        return True

    try:
        result = poll()
    except Exception:
        return True

    # Real subprocesses return None while running and an int when exited.
    # Treat mock/non-int sentinel values as still running so tests can inject
    # lightweight process doubles without configuring full subprocess state.
    return result is None or not isinstance(result, int)


def _launch_process(key: str) -> None:
    """Launch the configured sparkrun recipe for *key* if needed."""
    cfg = MODEL_REGISTRY[key]
    proc = cfg.get("process")
    if _process_is_running(proc):
        return

    cmd = ["sparkrun", "run", cfg["recipe"], *cfg.get("args", [])]
    logger.info("Launching model '%s': %s", key, " ".join(cmd))
    cfg["process"] = subprocess.Popen(cmd)
    cfg["healthy"] = False


def _stop_process(key: str) -> None:
    """Terminate the running process for *key* if one exists."""
    cfg = MODEL_REGISTRY[key]
    proc = cfg.get("process")
    cfg["healthy"] = False

    if proc is None:
        return

    if _process_is_running(proc):
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("Model '%s' did not stop cleanly; killing it.", key)
            proc.kill()
            proc.wait()

    cfg["process"] = None


async def _is_healthy(key: str) -> bool:
    """Probe the backend health endpoint and update cached model state."""
    url = f"{_backend_url(key)}/health"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
        healthy = resp.status_code == 200
    except (httpx.HTTPError, OSError):
        healthy = False

    MODEL_REGISTRY[key]["healthy"] = healthy
    return healthy


async def _wait_healthy(key: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Wait until the backend for *key* reports healthy or times out."""
    started_at = time.monotonic()
    while time.monotonic() - started_at < timeout:
        if await _is_healthy(key):
            return True
        await asyncio.sleep(2)
    return False


async def ensure_model_online(key: str) -> None:
    """Ensure the target model is healthy, evicting exclusive conflicts if needed."""
    cfg = MODEL_REGISTRY[key]

    async with _SWAP_LOCK:
        proc = cfg.get("process")
        if proc is not None and not _process_is_running(proc):
            cfg["process"] = None
            cfg["healthy"] = False

        if cfg.get("healthy") and cfg.get("process") is not None:
            return

        if cfg.get("exclusive"):
            for other_key, other_cfg in MODEL_REGISTRY.items():
                if other_key == key:
                    continue
                if other_cfg.get("exclusive") and other_cfg.get("process") is not None:
                    _stop_process(other_key)

        _launch_process(key)
        if not await _wait_healthy(key):
            _stop_process(key)
            raise RuntimeError(f"Model '{key}' failed to start")


async def _request_with_retries(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes,
) -> httpx.Response:
    """Send a backend request with bounded retries for transient failures."""
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
        for attempt in range(BACKEND_RETRIES):
            try:
                return await client.request(method=method, url=url, headers=headers, content=content)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == BACKEND_RETRIES - 1:
                    break
                delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
                await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


_typedb_driver = None
_typedb_mock_driver = False


def _open_typedb_driver():
    global _typedb_driver, _typedb_mock_driver
    if _typedb_driver:
        return

    try:
        logger.info(f"Connecting to TypeDB at {TYPEDB_ADDR}...")
        # Use TypeDB v3 driver API
        creds = _build_typedb_credentials()
        # Build options/credentials using whatever the installed driver exposes
        opts = None
        try:
            # Try TypeDBOptions (newer API)
            from typedb.driver import TypeDBOptions
            try:
                # Try constructing with TLS disabled and a request timeout
                from typedb.driver import DriverTlsConfig
                opts = TypeDBOptions(DriverTlsConfig.disabled(), request_timeout_millis=15000)
            except Exception:
                try:
                    opts = TypeDBOptions()
                except Exception:
                    opts = None
        except Exception:
            # Fallback to options_new/native options helpers
            try:
                from typedb.driver import options_new, options_set_transaction_timeout_millis
                opts = options_new()
                try:
                    options_set_transaction_timeout_millis(opts, 15000)
                except Exception:
                    pass
            except Exception:
                opts = None

        # Instantiate driver via TypeDBDriver class
        try:
            # Prefer TypeDB.core_driver which accepts credentials in this
            # driver implementation. Pass creds if available.
            # If credentials are present, instantiate the internal _Driver
            # with the credential so the native cloud-auth path is used.
            if creds is not None:
                from typedb.connection.driver import _Driver
                _typedb_driver = _Driver([TYPEDB_ADDR], creds)
            else:
                _typedb_driver = TypeDB.core_driver(TYPEDB_ADDR)
        except Exception as e:
            logger.exception("Failed to instantiate TypeDBDriver: %s", e)
            # As a last resort, try any factory on TypeDB if present
            try:
                _typedb_driver = TypeDB.core_driver(TYPEDB_ADDR)
            except Exception as e2:
                logger.exception("Fallback TypeDB.core_driver failed: %s", e2)
                # If authentication is enforced by the server and we are
                # intentionally working without credentials for development,
                # provide a minimal mock driver so the gateway can continue
                # operating without a live TypeDB connection.
                class _MockDatabases:
                    def contains(self, name: str) -> bool:
                        return True

                    def create(self, name: str) -> None:
                        logger.warning("Mock create database called for %s", name)

                class _MockDriver:
                    def __init__(self):
                        self.databases = _MockDatabases()
                        self.is_mock = True

                    def close(self):
                        logger.info("MockDriver.close() called")

                logger.warning("Using mock TypeDB driver (unauthenticated development fallback)")
                _typedb_driver = _MockDriver()
                _typedb_mock_driver = True
        if _typedb_driver is not None and not getattr(_typedb_driver, "is_mock", False):
            _typedb_mock_driver = False
        logger.info("Successfully connected to TypeDB.")

        # Ensure database exists
        if not _typedb_driver.databases.contains(TYPEDB_DATABASE):
            logger.info(f"Database '{TYPEDB_DATABASE}' not found. Creating it...")
            _typedb_driver.databases.create(TYPEDB_DATABASE)
            logger.info(f"Database '{TYPEDB_DATABASE}' created successfully.")

    except Exception as e:
        logger.error(f"Failed to connect to or set up TypeDB: {e}", exc_info=True)
        _typedb_driver = None
        _typedb_mock_driver = False


def _close_typedb_driver():
    global _typedb_driver, _typedb_mock_driver
    if _typedb_driver:
        _typedb_driver.close()
        _typedb_driver = None
        _typedb_mock_driver = False
        logger.info("Closed TypeDB driver connection.")


def _typedb_ready() -> bool:
    return _typedb_driver is not None and not _typedb_mock_driver


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP
    }


def _build_streaming_completion_response(
    stream_body: bytes,
    *,
    fallback_request_id: str,
) -> httpx.Response | None:
    events: list[dict[str, Any]] = []
    for raw_line in stream_body.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[len("data:") :].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)

    if not events:
        return None

    content_parts: list[str] = []
    finish_reason = None
    request_id = fallback_request_id
    model_name = None

    for event in events:
        request_id = str(event.get("id") or request_id)
        model_name = event.get("model") or model_name
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")

    assistant_content = "".join(content_parts).strip()
    if not assistant_content:
        return None

    return httpx.Response(
        200,
        json={
            "id": request_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": assistant_content},
                }
            ],
        },
        headers={"x-request-id": request_id},
    )


def _coerce_usage(raw_usage: Any) -> dict[str, int | float | str]:
    if not isinstance(raw_usage, dict):
        return {}
    usage: dict[str, int | float | str] = {}
    for key, value in raw_usage.items():
        if isinstance(value, (int, float, str)):
            usage[str(key)] = value
    return usage


def _extract_session_id(payload: dict[str, Any], headers: Any) -> str:
    header_value = None
    if headers is not None:
        header_value = headers.get("x-kortex-session-id") or headers.get("x-session-id")

    candidates = [
        header_value,
        payload.get("session_id"),
        payload.get("conversation_id"),
        payload.get("thread_id"),
    ]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("session_id"),
                metadata.get("conversation_id"),
                metadata.get("thread_id"),
            ]
        )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return str(uuid4())


def _extract_title(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content[:80]
    return None


def _build_writeback_event(
    *,
    path: str,
    method: str,
    request_body: bytes,
    request_headers: Any,
    backend_resp: httpx.Response,
    model_key: str,
    started_at: datetime,
    completed_at: datetime,
) -> TranscriptWritebackEvent | None:
    if method.upper() != "POST" or path != "chat/completions":
        return None

    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    normalized_messages = [message for message in messages if isinstance(message, dict)]
    if not normalized_messages:
        return None

    last_user = next(
        (message for message in reversed(normalized_messages) if message.get("role") == "user"),
        None,
    )
    if last_user is None:
        return None

    try:
        response_payload = backend_resp.json()
    except Exception:
        return None
    if not isinstance(response_payload, dict):
        return None

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0] if isinstance(choices[0], dict) else None
    if not first_choice:
        return None
    assistant_message = first_choice.get("message")
    if not isinstance(assistant_message, dict):
        return None

    assistant_content = assistant_message.get("content")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        return None

    session_id = _extract_session_id(payload, request_headers)
    request_id = backend_resp.headers.get("x-request-id") or response_payload.get("id") or str(uuid4())

    return TranscriptWritebackEvent(
        session_id=session_id,
        source="gateway",
        user_turn=TranscriptTurn(
            role="user",
            content=str(last_user.get("content", "")),
            turn_id=last_user.get("turn_id"),
        ),
        assistant_turn=TranscriptTurn(
            role=str(assistant_message.get("role", "assistant")),
            content=assistant_content,
            turn_id=assistant_message.get("turn_id"),
            timestamp=completed_at,
            metadata={"finish_reason": first_choice.get("finish_reason")},
        ),
        gateway_result=GatewayResult(
            request_id=request_id,
            resolved_model=model_key,
            response_text=assistant_content,
            started_at=started_at,
            completed_at=completed_at,
            usage=_coerce_usage(response_payload.get("usage")),
        ),
        title=_extract_title(normalized_messages),
        source_uri=f"gateway://{session_id}",
        previous_turn_count=max(0, len(normalized_messages) - 1),
        metadata={
            "path": path,
            "backend_url": _backend_url(model_key),
        },
    )


def _schedule_writeback_event(event: TranscriptWritebackEvent | None) -> None:
    if event is None:
        return
    try:
        asyncio.create_task(enqueue_writeback_event(event))
    except RuntimeError:
        logger.exception("Failed to schedule transcript writeback event.")


async def _poll_models():
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


def _stop_all_models() -> None:
    """Compatibility wrapper for gateway shutdown paths."""
    _shutdown_all_models()


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
    """
    Application startup/shutdown handler.
    - Opens TypeDB driver on startup.
    - Closes TypeDB driver on shutdown.
    - Manages model subprocesses.
    """
    # Startup
    logger.info("Gateway starting up...")
    _open_typedb_driver()
    await start_writeback_worker()

    # Start a background task to poll for model health
    poll_task = asyncio.create_task(_poll_models())
    yield
    # Shutdown
    logger.info("Gateway shutting down...")
    poll_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await poll_task
    await stop_writeback_worker()
    _close_typedb_driver()
    _stop_all_models()


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


@app.get("/health/ready")
async def gateway_ready(request: Request) -> JSONResponse:
    typedb_ok = _typedb_ready()
    checked_models = {
        key: await _is_healthy(key)
        for key, cfg in MODEL_REGISTRY.items()
        if cfg.get("process") is not None or cfg.get("healthy", False)
    }
    models_ok = bool(checked_models) and any(checked_models.values())
    ready = typedb_ok and models_ok
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "typedb": {"ready": typedb_ok, "database": TYPEDB_DATABASE},
            "models": checked_models,
        },
    )


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
    backend_started_at = datetime.now(timezone.utc)

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

    try:
        backend_resp = await _request_with_retries(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Backend for model '{model_key}' is not reachable: {exc}",
        ) from exc

    backend_completed_at = datetime.now(timezone.utc)

    # Detect streaming responses and forward them
    content_type = backend_resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        response_headers = _filter_response_headers(backend_resp.headers)

        async def _stream():
            chunks: list[bytes] = []
            async for chunk in backend_resp.aiter_bytes():
                chunks.append(chunk)
                yield chunk
            _schedule_writeback_event(
                _build_writeback_event(
                    path=path,
                    method=request.method,
                    request_body=body,
                    request_headers=request.headers,
                    backend_resp=_build_streaming_completion_response(
                        b"".join(chunks),
                        fallback_request_id=backend_resp.headers.get("x-request-id") or str(uuid4()),
                    )
                    or backend_resp,
                    model_key=model_key,
                    started_at=backend_started_at,
                    completed_at=datetime.now(timezone.utc),
                )
            )

        return StreamingResponse(
            _stream(),
            status_code=backend_resp.status_code,
            headers=response_headers,
            media_type=content_type,
        )

    _schedule_writeback_event(
        _build_writeback_event(
            path=path,
            method=request.method,
            request_body=body,
            request_headers=request.headers,
            backend_resp=backend_resp,
            model_key=model_key,
            started_at=backend_started_at,
            completed_at=backend_completed_at,
        )
    )

    return Response(
        content=backend_resp.content,
        status_code=backend_resp.status_code,
        headers=_filter_response_headers(backend_resp.headers),
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
