from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from memory.chat_ingest import (
    build_ingestion_batch,
    build_qdrant_points,
    classify_directive,
    extract_artifact_paths,
    load_writeback_event,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chat_writeback_event.json"


class TestFixtureLoading:
    def test_load_writeback_event(self):
        event = load_writeback_event(FIXTURE)

        assert event.session_id == "session-demo-1"
        assert event.gateway_result.resolved_model == "qwen-coder"
        assert event.user_turn.role == "user"
        assert event.assistant_turn.turn_id == "assistant-turn-8"


class TestExtraction:
    def test_extract_artifact_paths(self):
        refs = extract_artifact_paths(
            "Touch gateway/gateway.py and docs/spec.md but not plain-text names"
        )

        assert refs == ("gateway/gateway.py", "docs/spec.md")

    def test_classify_constraint_directive(self):
        assert classify_directive("Always keep health checks aligned.") == "workflow"
        assert classify_directive("Never skip schema validation.") == "constraint"
        assert classify_directive("This is just a neutral sentence.") is None


class TestIngestionBatch:
    def test_build_batch_contains_graph_queries_and_embedding_docs(self):
        event = load_writeback_event(FIXTURE)

        batch = build_ingestion_batch(event)

        assert batch.session.session_id == "session-demo-1"
        assert len(batch.turns) == 2
        assert batch.turns[0].turn_index == 7
        assert batch.turns[1].model_name == "qwen-coder"
        assert len(batch.directives) == 1
        assert len(batch.artifacts) == 3
        rendered = "\n\n".join(batch.queries)
        assert "chat-session" in rendered
        assert "directive-node" in rendered
        assert "project-artifact" in rendered
        assert "assistant-turn-8" in rendered
        assert len(batch.embedding_documents) == 2

    def test_build_qdrant_points_rejects_length_mismatch(self):
        event = load_writeback_event(FIXTURE)
        batch = build_ingestion_batch(event)

        with pytest.raises(ValueError, match="same length"):
            build_qdrant_points(batch.embedding_documents, [[0.1, 0.2]])

    def test_build_qdrant_points_preserves_payload(self):
        event = load_writeback_event(FIXTURE)
        batch = build_ingestion_batch(event)

        points = build_qdrant_points(
            batch.embedding_documents,
            [[0.1, 0.2], [0.3, 0.4]],
        )

        assert len(points) == 2
        assert points[0].payload["session_id"] == "session-demo-1"
        assert points[1].payload["resolved_model"] == "qwen-coder"
