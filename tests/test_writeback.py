from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from gateway.gateway import _build_writeback_event
from kortex.contracts import (
    GatewayResult,
    TranscriptTurn,
    TranscriptWritebackEvent,
    transcript_writeback_event_to_dict,
)
from memory import writeback


class TestContractsSerialization:
    def test_transcript_writeback_event_to_dict_serializes_datetimes(self):
        event = TranscriptWritebackEvent(
            session_id="session-1",
            source="gateway",
            user_turn=TranscriptTurn(role="user", content="hello"),
            assistant_turn=TranscriptTurn(
                role="assistant",
                content="hi",
                timestamp=datetime(2026, 7, 23, 16, 0, tzinfo=timezone.utc),
            ),
            gateway_result=GatewayResult(
                request_id="req-1",
                resolved_model="qwen-coder",
                response_text="hi",
                started_at=datetime(2026, 7, 23, 16, 0, tzinfo=timezone.utc),
                completed_at=datetime(2026, 7, 23, 16, 0, 1, tzinfo=timezone.utc),
            ),
        )

        payload = transcript_writeback_event_to_dict(event)

        assert payload["assistant_turn"]["timestamp"].endswith("+00:00")
        assert payload["gateway_result"]["completed_at"].endswith("+00:00")


class TestGatewayWritebackEventBuilder:
    def test_builds_event_for_chat_completion(self):
        backend_resp = httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Done."},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            headers={"x-request-id": "req-1"},
        )

        event = _build_writeback_event(
            path="chat/completions",
            method="POST",
            request_body=json.dumps(
                {
                    "session_id": "session-123",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Update gateway/gateway.py."},
                    ],
                }
            ).encode("utf-8"),
            request_headers={},
            backend_resp=backend_resp,
            model_key="qwen-coder",
            started_at=datetime(2026, 7, 23, 16, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 7, 23, 16, 0, 1, tzinfo=timezone.utc),
        )

        assert event is not None
        assert event.session_id == "session-123"
        assert event.previous_turn_count == 1
        assert event.gateway_result.request_id == "req-1"
        assert event.gateway_result.usage["prompt_tokens"] == 10
        assert event.assistant_turn.metadata["finish_reason"] == "stop"

    def test_returns_none_for_non_chat_requests(self):
        backend_resp = httpx.Response(200, json={"ok": True})

        event = _build_writeback_event(
            path="models",
            method="GET",
            request_body=b"",
            request_headers={},
            backend_resp=backend_resp,
            model_key="qwen-coder",
            started_at=datetime(2026, 7, 23, 16, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 7, 23, 16, 0, 1, tzinfo=timezone.utc),
        )

        assert event is None


class TestWritebackWorker:
    @pytest.mark.asyncio
    async def test_writeback_worker_appends_jsonl(self, tmp_path, monkeypatch):
        path = tmp_path / "writeback.jsonl"
        monkeypatch.setattr(writeback, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(writeback, "_writeback_queue", None)
        monkeypatch.setattr(writeback, "_writeback_task", None)
        await writeback.start_writeback_worker(path)
        try:
            event = TranscriptWritebackEvent(
                session_id="session-1",
                source="gateway",
                user_turn=TranscriptTurn(role="user", content="hello"),
                assistant_turn=TranscriptTurn(role="assistant", content="hi"),
                gateway_result=GatewayResult(
                    request_id="req-1",
                    resolved_model="qwen-coder",
                    response_text="hi",
                ),
            )
            await writeback.enqueue_writeback_event(event)
            assert writeback._writeback_queue is not None
            await writeback._writeback_queue.join()
        finally:
            await writeback.stop_writeback_worker()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["session_id"] == "session-1"
        assert payload["assistant_turn"]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_writeback_worker_optionally_persists_event(self, tmp_path, monkeypatch):
        path = tmp_path / "writeback.jsonl"
        calls = []
        monkeypatch.setattr(writeback, "WRITEBACK_ENABLED", True)
        monkeypatch.setattr(writeback, "WRITEBACK_PERSIST_ENABLED", True)
        monkeypatch.setattr(writeback, "_writeback_queue", None)
        monkeypatch.setattr(writeback, "_writeback_task", None)

        async def fake_persist(event):
            calls.append(event.session_id)

        monkeypatch.setattr(writeback, "persist_writeback_event", fake_persist)
        await writeback.start_writeback_worker(path)
        try:
            event = TranscriptWritebackEvent(
                session_id="session-2",
                source="gateway",
                user_turn=TranscriptTurn(role="user", content="hello"),
                assistant_turn=TranscriptTurn(role="assistant", content="hi"),
                gateway_result=GatewayResult(
                    request_id="req-2",
                    resolved_model="qwen-coder",
                    response_text="hi",
                ),
            )
            await writeback.enqueue_writeback_event(event)
            assert writeback._writeback_queue is not None
            await writeback._writeback_queue.join()
        finally:
            await writeback.stop_writeback_worker()

        assert calls == ["session-2"]
