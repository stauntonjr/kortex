"""
Kortex Gateway — Unit & Integration Tests
==========================================
Tests are intentionally self-contained and do not require live model
processes, TypeDB, Qdrant, or Redis. Subprocesses and HTTP calls are
fully mocked.

Run with::

    pytest tests/test_gateway.py -v
"""

from __future__ import annotations

import json
import subprocess
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
pytest_asyncio = pytest.importorskip("pytest_asyncio")
pytest.importorskip("respx")
pytest.importorskip("typedb.driver")

from memory import writeback

# Import the gateway module under test
from gateway.gateway import (
    MODEL_REGISTRY,
    _is_healthy,
    _launch_process,
    _request_with_retries,
    _build_streaming_completion_response,
    _stop_process,
    app,
    ensure_model_online,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """
    Reset model process/healthy state before every test so tests are
    independent of each other.
    """
    for cfg in MODEL_REGISTRY.values():
        cfg["process"] = None
        cfg["healthy"] = False


@pytest.fixture(autouse=True)
def disable_writeback(monkeypatch):
    monkeypatch.setattr(writeback, "WRITEBACK_ENABLED", False)
    monkeypatch.setattr(writeback, "_writeback_queue", None)
    monkeypatch.setattr(writeback, "_writeback_task", None)
    yield
    for cfg in MODEL_REGISTRY.values():
        proc = cfg.get("process")
        if proc is not None:
            proc.terminate()
            proc.wait()
        cfg["process"] = None
        cfg["healthy"] = False


@pytest_asyncio.fixture
async def async_client():
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    app.router.lifespan_context = noop_lifespan
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client
    finally:
        app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_canonical_key_resolves_to_itself(self):
        for key in MODEL_REGISTRY:
            assert resolve_model(key) == key

    def test_alias_resolves_to_canonical(self):
        assert resolve_model("nemotron") == "nemotron-120b"

    def test_unknown_model_returns_none(self):
        assert resolve_model("does-not-exist") is None

    def test_canonical_key_round_trips(self):
        assert resolve_model("qwen-coder") == "qwen-coder"


# ---------------------------------------------------------------------------
# _is_healthy
# ---------------------------------------------------------------------------


class TestIsHealthy:
    @pytest.mark.asyncio
    async def test_healthy_when_backend_returns_200(self, respx_mock):
        port = MODEL_REGISTRY["qwen-coder"]["port"]
        respx_mock.get(f"http://127.0.0.1:{port}/health").mock(
            return_value=httpx.Response(200)
        )
        result = await _is_healthy("qwen-coder")
        assert result is True
        assert MODEL_REGISTRY["qwen-coder"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_unhealthy_when_backend_returns_503(self, respx_mock):
        port = MODEL_REGISTRY["qwen-coder"]["port"]
        respx_mock.get(f"http://127.0.0.1:{port}/health").mock(
            return_value=httpx.Response(503)
        )
        result = await _is_healthy("qwen-coder")
        assert result is False
        assert MODEL_REGISTRY["qwen-coder"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_unhealthy_on_connection_error(self, respx_mock):
        port = MODEL_REGISTRY["qwen-coder"]["port"]
        respx_mock.get(f"http://127.0.0.1:{port}/health").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await _is_healthy("qwen-coder")
        assert result is False


# ---------------------------------------------------------------------------
# _launch_process / _stop_process
# ---------------------------------------------------------------------------


class TestProcessLifecycle:
    def test_launch_creates_popen_handle(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        with patch("gateway.gateway.subprocess.Popen", return_value=mock_proc) as mock_popen:
            _launch_process("qwen-coder")
            mock_popen.assert_called_once()
            assert MODEL_REGISTRY["qwen-coder"]["process"] is mock_proc

    def test_launch_is_idempotent_when_process_exists(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        MODEL_REGISTRY["qwen-coder"]["process"] = mock_proc
        with patch("gateway.gateway.subprocess.Popen") as mock_popen:
            _launch_process("qwen-coder")
            mock_popen.assert_not_called()

    def test_stop_terminates_process(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        MODEL_REGISTRY["qwen-coder"]["process"] = mock_proc
        _stop_process("qwen-coder")
        mock_proc.terminate.assert_called_once()
        assert MODEL_REGISTRY["qwen-coder"]["process"] is None
        assert MODEL_REGISTRY["qwen-coder"]["healthy"] is False

    def test_stop_kills_on_timeout(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="sparkrun", timeout=15), None]
        MODEL_REGISTRY["qwen-coder"]["process"] = mock_proc
        _stop_process("qwen-coder")
        mock_proc.kill.assert_called_once()

    def test_stop_noop_when_not_running(self):
        # Should not raise
        _stop_process("qwen-coder")


# ---------------------------------------------------------------------------
# ensure_model_online
# ---------------------------------------------------------------------------


class TestEnsureModelOnline:
    @pytest.mark.asyncio
    async def test_skips_if_already_healthy(self):
        MODEL_REGISTRY["qwen-coder"]["process"] = MagicMock()
        MODEL_REGISTRY["qwen-coder"]["healthy"] = True
        with patch("gateway.gateway._launch_process") as mock_launch:
            await ensure_model_online("qwen-coder")
            mock_launch.assert_not_called()

    @pytest.mark.asyncio
    async def test_launches_background_model_without_eviction(self):
        with (
            patch("gateway.gateway._launch_process") as mock_launch,
            patch("gateway.gateway._wait_healthy", new_callable=AsyncMock, return_value=True),
        ):
            await ensure_model_online("qwen-embedding")
            mock_launch.assert_called_once_with("qwen-embedding")

    @pytest.mark.asyncio
    async def test_exclusive_model_evicts_other_exclusive(self):
        # Simulate qwen-35b already running
        MODEL_REGISTRY["qwen-35b"]["process"] = MagicMock(spec=subprocess.Popen)
        MODEL_REGISTRY["qwen-35b"]["healthy"] = True

        with (
            patch("gateway.gateway._stop_process") as mock_stop,
            patch("gateway.gateway._launch_process"),
            patch("gateway.gateway._wait_healthy", new_callable=AsyncMock, return_value=True),
        ):
            await ensure_model_online("nemotron-120b")
            # qwen-35b must be evicted since nemotron-120b is also exclusive
            mock_stop.assert_called_with("qwen-35b")

    @pytest.mark.asyncio
    async def test_exclusive_model_does_not_evict_background(self):
        # qwen-embedding (non-exclusive) is running
        MODEL_REGISTRY["qwen-embedding"]["process"] = MagicMock(spec=subprocess.Popen)
        MODEL_REGISTRY["qwen-embedding"]["healthy"] = True

        stopped: list[str] = []

        def record_stop(key):
            stopped.append(key)

        with (
            patch("gateway.gateway._stop_process", side_effect=record_stop),
            patch("gateway.gateway._launch_process"),
            patch("gateway.gateway._wait_healthy", new_callable=AsyncMock, return_value=True),
        ):
            await ensure_model_online("nemotron-120b")
            assert "qwen-embedding" not in stopped

    @pytest.mark.asyncio
    async def test_raises_on_startup_failure(self):
        with (
            patch("gateway.gateway._launch_process"),
            patch("gateway.gateway._wait_healthy", new_callable=AsyncMock, return_value=False),
        ):
            with pytest.raises(RuntimeError, match="failed to start"):
                await ensure_model_online("qwen-coder")


# ---------------------------------------------------------------------------
# _request_with_retries
# ---------------------------------------------------------------------------


class TestRequestWithRetries:
    @pytest.mark.asyncio
    async def test_retries_backend_connect_errors(self):
        fake_response = httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        with (
            patch(
                "gateway.gateway.httpx.AsyncClient.request",
                new_callable=AsyncMock,
                side_effect=[httpx.ConnectError("refused"), fake_response],
            ) as mock_req,
            patch("gateway.gateway.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            resp = await _request_with_retries(
                method="POST",
                url="http://127.0.0.1:8002/v1/chat/completions",
                headers={"content-type": "application/json"},
                content=b'{"model":"qwen-coder"}',
            )

        assert resp.status_code == 200
        assert mock_req.await_count == 2
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_backend_timeouts(self):
        fake_response = httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        with (
            patch(
                "gateway.gateway.httpx.AsyncClient.request",
                new_callable=AsyncMock,
                side_effect=[httpx.TimeoutException("timed out"), fake_response],
            ) as mock_req,
            patch("gateway.gateway.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            resp = await _request_with_retries(
                method="POST",
                url="http://127.0.0.1:8002/v1/chat/completions",
                headers={"content-type": "application/json"},
                content=b'{"model":"qwen-coder"}',
            )

        assert resp.status_code == 200
        assert mock_req.await_count == 2
        mock_sleep.assert_awaited_once()


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestGatewayEndpoints:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert set(data["models"].keys()) == set(MODEL_REGISTRY.keys())

    @pytest.mark.asyncio
    async def test_ready_returns_200_when_typedb_and_model_are_available(self, async_client):
        MODEL_REGISTRY["qwen-coder"]["process"] = MagicMock()

        with (
            patch("gateway.gateway._typedb_ready", return_value=True),
            patch("gateway.gateway._is_healthy", new_callable=AsyncMock, return_value=True),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["typedb"]["ready"] is True
        assert data["models"]["qwen-coder"] is True

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_typedb_is_unavailable(self, async_client):
        MODEL_REGISTRY["qwen-coder"]["process"] = MagicMock()

        with (
            patch("gateway.gateway._typedb_ready", return_value=False),
            patch("gateway.gateway._is_healthy", new_callable=AsyncMock, return_value=True),
        ):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_no_models_are_healthy(self, async_client):
        with patch("gateway.gateway._typedb_ready", return_value=True):
            resp = await async_client.get("/health/ready")

        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_list_models_returns_all_entries(self, async_client):
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = {m["id"] for m in data["data"]}
        assert ids == set(MODEL_REGISTRY.keys())

    @pytest.mark.asyncio
    async def test_proxy_routes_to_correct_backend(self, async_client):
        """A POST to /v1/chat/completions with model=qwen-coder reaches port 8002."""
        MODEL_REGISTRY["qwen-coder"]["process"] = MagicMock()
        MODEL_REGISTRY["qwen-coder"]["healthy"] = True

        fake_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

        with (
            patch("gateway.gateway.ensure_model_online", new_callable=AsyncMock),
            patch(
                "gateway.gateway._request_with_retries",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_backend,
        ):
            body = json.dumps({"model": "qwen-coder", "messages": [{"role": "user", "content": "hi"}]})
            resp = await async_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200
            # Verify the URL pointed to the qwen-coder backend port
            call_url: str = mock_backend.call_args.kwargs.get("url", "")
            assert "8002" in str(call_url)

    @pytest.mark.asyncio
    async def test_proxy_falls_back_to_qwen_coder_for_unknown_model(self, async_client):
        fake_response = httpx.Response(200, json={"ok": True})

        with (
            patch("gateway.gateway.ensure_model_online", new_callable=AsyncMock),
            patch(
                "gateway.gateway._request_with_retries",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_backend,
        ):
            body = json.dumps({"model": "unknown-model-xyz", "messages": []})
            resp = await async_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200
            call_url = str(mock_backend.call_args.kwargs.get("url", ""))
            assert "8002" in call_url  # qwen-coder port

    @pytest.mark.asyncio
    async def test_proxy_returns_503_when_model_fails_to_start(self, async_client):
        with patch(
            "gateway.gateway.ensure_model_online",
            new_callable=AsyncMock,
            side_effect=RuntimeError("failed to start"),
        ):
            body = json.dumps({"model": "qwen-coder", "messages": []})
            resp = await async_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_proxy_returns_503_on_backend_connect_error(self, async_client):
        with (
            patch("gateway.gateway.ensure_model_online", new_callable=AsyncMock),
            patch(
                "gateway.gateway._request_with_retries",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("refused"),
            ),
        ):
            body = json.dumps({"model": "qwen-coder", "messages": []})
            resp = await async_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_proxy_streams_and_schedules_writeback(self, async_client):
        backend_resp = httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "connection": "keep-alive",
                "x-request-id": "req-stream-1",
            },
            content=(
                b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
                b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

        scheduled = []

        with (
            patch("gateway.gateway.ensure_model_online", new_callable=AsyncMock),
            patch(
                "gateway.gateway._request_with_retries",
                new_callable=AsyncMock,
                return_value=backend_resp,
            ),
            patch("gateway.gateway._schedule_writeback_event", side_effect=scheduled.append),
        ):
            resp = await async_client.post(
                "/v1/chat/completions",
                content=json.dumps(
                    {
                        "model": "qwen-coder",
                        "session_id": "session-stream-1",
                        "messages": [{"role": "user", "content": "Say hello"}],
                    }
                ),
                headers={"content-type": "application/json"},
            )
            body = await resp.aread()

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "connection" not in resp.headers
        assert b'"content":"Hel"' in body
        assert b'"content":"lo"' in body
        assert len(scheduled) == 1
        assert scheduled[0] is not None
        assert scheduled[0].assistant_turn.content == "Hello"


class TestStreamingWritebackHelpers:
    def test_build_streaming_completion_response_reconstructs_assistant_message(self):
        response = _build_streaming_completion_response(
            (
                b'data: {"id":"chatcmpl-stream","model":"qwen-coder","choices":[{"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
                b'data: {"id":"chatcmpl-stream","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            fallback_request_id="req-fallback",
        )

        assert response is not None
        payload = response.json()
        assert payload["id"] == "chatcmpl-stream"
        assert payload["choices"][0]["message"]["content"] == "Hello"
        assert payload["choices"][0]["finish_reason"] == "stop"
