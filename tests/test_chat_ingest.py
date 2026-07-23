from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from memory.chat_ingest import (
    build_ingestion_batch,
    build_qdrant_points,
    classify_directive,
    persist_batch_to_qdrant,
    persist_batch_to_typedb,
    persist_writeback_event,
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

    @pytest.mark.asyncio
    async def test_persist_batch_to_qdrant_embeds_and_upserts_points(self):
        event = load_writeback_event(FIXTURE)
        batch = build_ingestion_batch(event)
        seen = {}

        async def fake_embedder(documents):
            seen["documents"] = documents
            return [[0.1, 0.2], [0.3, 0.4]]

        async def fake_upsert(points):
            seen["points"] = points

        points = await persist_batch_to_qdrant(batch, fake_embedder, fake_upsert)

        assert tuple(seen["documents"]) == batch.embedding_documents
        assert len(points) == 2
        assert seen["points"][0].payload["turn_id"] == batch.turns[0].turn_id

    def test_persist_batch_to_typedb_delegates_upsert(self, monkeypatch):
        event = load_writeback_event(FIXTURE)
        batch = build_ingestion_batch(event)
        calls = {}

        def fake_upsert(driver, session, turns, directives, artifacts, database):
            calls["driver"] = driver
            calls["session"] = session
            calls["turns"] = turns
            calls["directives"] = directives
            calls["artifacts"] = artifacts
            calls["database"] = database

        monkeypatch.setattr("memory.chat_ingest.upsert_chat_session_to_typedb", fake_upsert)
        driver = object()

        persist_batch_to_typedb(driver, batch, "kortex")

        assert calls["driver"] is driver
        assert calls["session"] == batch.session
        assert calls["turns"] == batch.turns
        assert calls["database"] == "kortex"

    @pytest.mark.asyncio
    async def test_persist_writeback_event_runs_selected_backends(self):
        event = load_writeback_event(FIXTURE)
        seen = {}

        def fake_typedb(driver, batch, database):
            seen["typedb"] = (driver, batch, database)

        async def fake_qdrant(batch, embedder, upsert):
            seen["qdrant"] = (batch, embedder, upsert)
            return []

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("memory.chat_ingest.persist_batch_to_typedb", fake_typedb)
        monkeypatch.setattr("memory.chat_ingest.persist_batch_to_qdrant", fake_qdrant)
        try:
            driver = object()

            async def fake_embedder(documents):
                return [[0.1, 0.2] for _ in documents]

            async def fake_upsert(points):
                return None

            batch = await persist_writeback_event(
                event,
                typedb_driver=driver,
                typedb_database="kortex",
                embedder=fake_embedder,
                qdrant_upsert=fake_upsert,
            )
        finally:
            monkeypatch.undo()

        assert seen["typedb"][0] is driver
        assert seen["typedb"][2] == "kortex"
        assert seen["qdrant"][0] == batch
