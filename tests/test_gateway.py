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
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

# Import the gateway module under test
from gateway.gateway import (
    MODEL_REGISTRY,
    _is_healthy,
    _launch_process,
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
    yield
    for cfg in MODEL_REGISTRY.values():
        proc = cfg.get("process")
        if proc is not None:
            proc.terminate()
            proc.wait()
        cfg["process"] = None
        cfg["healthy"] = False


@pytest.fixture
def test_client():
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


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
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestGatewayEndpoints:
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert set(data["models"].keys()) == set(MODEL_REGISTRY.keys())

    def test_list_models_returns_all_entries(self, test_client):
        resp = test_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = {m["id"] for m in data["data"]}
        assert ids == set(MODEL_REGISTRY.keys())

    def test_proxy_routes_to_correct_backend(self, test_client):
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
                "gateway.gateway.httpx.AsyncClient.request",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_req,
        ):
            body = json.dumps({"model": "qwen-coder", "messages": [{"role": "user", "content": "hi"}]})
            resp = test_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200
            # Verify the URL pointed to the qwen-coder backend port
            call_url: str = mock_req.call_args.kwargs.get("url", mock_req.call_args.args[1] if mock_req.call_args.args else "")
            assert "8002" in str(call_url)

    def test_proxy_falls_back_to_qwen_coder_for_unknown_model(self, test_client):
        fake_response = httpx.Response(200, json={"ok": True})

        with (
            patch("gateway.gateway.ensure_model_online", new_callable=AsyncMock),
            patch(
                "gateway.gateway.httpx.AsyncClient.request",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_req,
        ):
            body = json.dumps({"model": "unknown-model-xyz", "messages": []})
            resp = test_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 200
            call_url = str(mock_req.call_args.kwargs.get("url", ""))
            assert "8002" in call_url  # qwen-coder port

    def test_proxy_returns_503_when_model_fails_to_start(self, test_client):
        with patch(
            "gateway.gateway.ensure_model_online",
            new_callable=AsyncMock,
            side_effect=RuntimeError("failed to start"),
        ):
            body = json.dumps({"model": "qwen-coder", "messages": []})
            resp = test_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 503

    def test_proxy_returns_503_on_backend_connect_error(self, test_client):
        with (
            patch("gateway.gateway.ensure_model_online", new_callable=AsyncMock),
            patch(
                "gateway.gateway.httpx.AsyncClient.request",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("refused"),
            ),
        ):
            body = json.dumps({"model": "qwen-coder", "messages": []})
            resp = test_client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 503
