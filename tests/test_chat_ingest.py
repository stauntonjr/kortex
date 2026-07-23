from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from memory.chat_ingest import (
    CHAT_TYPEDB_DATABASE,
    build_ingestion_batch,
    build_qdrant_points,
    classify_directive,
    build_default_qdrant_upsert,
    ingest_writeback_log,
    iter_writeback_events,
    persist_batch_to_qdrant,
    persist_batch_to_typedb,
    persist_writeback_event,
    extract_artifact_paths,
    load_writeback_event,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chat_writeback_event.json"


def fixture_jsonl_line() -> str:
    return json.dumps(json.loads(FIXTURE.read_text()), separators=(",", ":"))


class TestFixtureLoading:
    def test_load_writeback_event(self):
        event = load_writeback_event(FIXTURE)

        assert event.session_id == "session-demo-1"
        assert event.gateway_result.resolved_model == "qwen-coder"
        assert event.user_turn.role == "user"
        assert event.assistant_turn.turn_id == "assistant-turn-8"

    def test_iter_writeback_events_reads_jsonl(self, tmp_path):
        line = fixture_jsonl_line()
        path = tmp_path / "writeback.jsonl"
        path.write_text(f"{line}\n{line}\n")

        events = iter_writeback_events(path)

        assert len(events) == 2
        assert events[0].session_id == "session-demo-1"
        assert events[1].assistant_turn.turn_id == "assistant-turn-8"

    def test_iter_writeback_events_reports_invalid_line(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text(f"{fixture_jsonl_line()}\nnot-json\n")

        with pytest.raises(ValueError, match="line 2"):
            iter_writeback_events(path)


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

    @pytest.mark.asyncio
    async def test_persist_writeback_event_uses_default_backends(self, monkeypatch):
        event = load_writeback_event(FIXTURE)
        seen = {}

        class FakeDriverContext:
            def __enter__(self):
                seen["driver"] = object()
                return seen["driver"]

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_open_driver():
            seen["open_driver"] = True
            return FakeDriverContext()

        def fake_bootstrap(driver, database, schema_path):
            seen["bootstrap"] = (driver, database, schema_path.name)

        def fake_typedb(driver, batch, database):
            seen["typedb"] = (driver, batch, database)

        async def fake_qdrant(batch, embedder, upsert):
            seen["qdrant"] = (batch, embedder, upsert)
            return []

        monkeypatch.setattr("memory.chat_ingest.PERSIST_TO_TYPEDB", True)
        monkeypatch.setattr("memory.chat_ingest.PERSIST_TO_QDRANT", True)
        monkeypatch.setattr("memory.chat_ingest.open_default_typedb_driver", fake_open_driver)
        monkeypatch.setattr("memory.chat_ingest.bootstrap_typedb", fake_bootstrap)
        monkeypatch.setattr("memory.chat_ingest.persist_batch_to_typedb", fake_typedb)
        monkeypatch.setattr("memory.chat_ingest.persist_batch_to_qdrant", fake_qdrant)
        monkeypatch.setattr("memory.chat_ingest.build_default_embedder", lambda: "embedder")
        monkeypatch.setattr("memory.chat_ingest.build_default_qdrant_upsert", lambda: "upsert")

        batch = await persist_writeback_event(event)

        assert seen["open_driver"] is True
        assert seen["bootstrap"][1] == CHAT_TYPEDB_DATABASE
        assert seen["bootstrap"][2] == "schema.tql"
        assert seen["typedb"][0] is seen["driver"]
        assert seen["typedb"][1] == batch
        assert seen["qdrant"][0] == batch
        assert seen["qdrant"][1] == "embedder"
        assert seen["qdrant"][2] == "upsert"

    @pytest.mark.asyncio
    async def test_build_default_qdrant_upsert_creates_collection_before_upsert(self, monkeypatch):
        calls = {}

        class FakeCollections:
            collections = []

        class FakeClient:
            def __init__(self, url):
                calls["url"] = url

            async def get_collections(self):
                calls["get_collections"] = True
                return FakeCollections()

            async def create_collection(self, collection_name, vectors_config):
                calls["create_collection"] = (collection_name, vectors_config.size)

            async def upsert(self, collection_name, points):
                calls["upsert"] = (collection_name, points)

        monkeypatch.setattr("memory.chat_ingest.AsyncQdrantClient", FakeClient)
        upsert = build_default_qdrant_upsert()

        await upsert([])
        assert "upsert" not in calls

        point = build_qdrant_points(
            (
                build_ingestion_batch(load_writeback_event(FIXTURE)).embedding_documents[0],
            ),
            [[0.1, 0.2]],
        )[0]
        await upsert([point])

        assert calls["get_collections"] is True
        assert calls["create_collection"][0] == "kortex_chat"
        assert calls["upsert"][0] == "kortex_chat"

    @pytest.mark.asyncio
    async def test_ingest_writeback_log_replays_each_event(self, tmp_path, monkeypatch):
        line = fixture_jsonl_line()
        path = tmp_path / "writeback.jsonl"
        path.write_text(f"{line}\n{line}\n")
        seen = []

        async def fake_persist(event, **kwargs):
            del kwargs
            seen.append(event.session_id)
            return build_ingestion_batch(event)

        monkeypatch.setattr("memory.chat_ingest.persist_writeback_event", fake_persist)

        count = await ingest_writeback_log(path)

        assert count == 2
        assert seen == ["session-demo-1", "session-demo-1"]
