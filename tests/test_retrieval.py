from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from kortex.contracts import RetrievalRequest, RetrievalResult, RetrievedNode
from memory.retrieval import (
    build_default_memory_retriever,
    build_neighbor_expansion_query,
    build_seed_lookup_query,
    build_memory_context,
    merge_retrieval_nodes,
    qdrant_candidate_search,
    retrieve_memory,
    typedb_expand_with_driver,
)


class TestQdrantCandidateSearch:
    @pytest.mark.asyncio
    async def test_uses_query_points_and_applies_chat_session_filter(self):
        calls = []

        class FakeClient:
            async def query_points(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id="turn-1",
                            score=0.9,
                            payload={
                                "turn_id": "turn-1",
                                "content": "chat memory",
                                "session_id": "session-1",
                                "source_uri": "chat://session-1#1",
                            },
                        )
                    ]
                )

        nodes = await qdrant_candidate_search(
            FakeClient(),
            RetrievalRequest(query="chat memory", modes=("chat",), session_id="session-1", max_nodes=4),
            query_vector=[0.1, 0.2],
        )

        assert len(nodes) == 1
        assert nodes[0].kind == "chat"
        assert calls[0]["query_filter"]["must"][0]["match"]["value"] == "session-1"

    @pytest.mark.asyncio
    async def test_falls_back_to_search_and_dedupes_by_node_id(self):
        class FakeClient:
            async def search(self, **kwargs):
                if kwargs["collection_name"] == "kortex_code":
                    return [
                        SimpleNamespace(
                            id="dup",
                            score=0.7,
                            payload={"entity_id": "dup", "content": "code memory", "path": "a.py"},
                        )
                    ]
                return [
                    SimpleNamespace(
                        id="dup",
                        score=0.9,
                        payload={"turn_id": "dup", "content": "chat memory", "source_uri": "chat://dup"},
                    )
                ]

        nodes = await qdrant_candidate_search(
            FakeClient(),
            RetrievalRequest(query="memory", modes=("code", "chat"), max_nodes=4),
            query_vector=[0.1, 0.2],
        )

        assert len(nodes) == 1
        assert nodes[0].score == 0.9


class TestTypeDBExpansion:
    def test_build_seed_lookup_query_uses_kind_specific_ids(self):
        query = build_seed_lookup_query(
            RetrievedNode(node_id="turn-1", kind="chat", content="", score=0.9)
        )

        assert "chat-turn" in query
        assert 'has turn-id "turn-1"' in query

    def test_build_neighbor_expansion_query_targets_mentions(self):
        query = build_neighbor_expansion_query(
            RetrievedNode(node_id="turn-1", kind="chat", content="", score=0.9)
        )

        assert "(source-turn: $seed, target: $neighbor) isa mention" in query

    def test_build_neighbor_expansion_query_reverses_mentions_for_code_seed(self):
        query = build_neighbor_expansion_query(
            RetrievedNode(node_id="gateway/gateway.py", kind="code", content="", score=0.9)
        )

        assert "(source-turn: $neighbor, target: $seed) isa mention" in query

    def test_typedb_expand_with_driver_hydrates_seed_and_neighbors(self):
        seed = RetrievedNode(
            node_id="turn-1",
            kind="chat",
            content="seed text",
            score=0.9,
            depth=0,
            name="turn-1",
        )

        class FakeTx:
            def __init__(self):
                self.queries = []

            def query(self, query_text):
                self.queries.append(query_text)
                if "fetch\n  $node:" in query_text:
                    return [
                        {
                            "turn-id": "turn-1",
                            "content-text": "hydrated seed text",
                            "source-uri": "chat://session-1#1",
                        }
                    ]
                return [
                    {
                        "entity-id": "gateway/gateway.py",
                        "content-text": "gateway writeback implementation",
                        "entity-name": "gateway.py",
                        "entity-path": "gateway/gateway.py",
                    }
                ]

        class FakeTxContext:
            def __init__(self, tx):
                self.tx = tx

            def __enter__(self):
                return self.tx

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeDriver:
            def __init__(self):
                self.tx = FakeTx()

            def transaction(self, database, transaction_type, options=None):
                del database, transaction_type, options
                return FakeTxContext(self.tx)

        nodes = typedb_expand_with_driver(FakeDriver(), (seed,), max_depth=2)

        assert [node.node_id for node in nodes] == ["turn-1", "gateway/gateway.py"]
        assert nodes[0].content == "hydrated seed text"
        assert nodes[1].kind == "code"
        assert nodes[1].depth == 1


class TestMergeRetrievalNodes:
    def test_respects_depth_and_token_budget(self):
        seeds = (
            RetrievedNode(node_id="seed", kind="chat", content="A" * 16, score=0.9, depth=0, name="seed"),
        )
        neighbors = (
            RetrievedNode(node_id="near", kind="code", content="B" * 16, score=0.8, depth=1, name="near"),
            RetrievedNode(node_id="deep", kind="code", content="C" * 16, score=0.7, depth=4, name="deep"),
            RetrievedNode(node_id="large", kind="code", content="D" * 400, score=0.95, depth=1, name="large"),
        )

        nodes = merge_retrieval_nodes(seeds, neighbors, max_depth=2, max_nodes=5, token_budget=12)

        assert [node.node_id for node in nodes] == ["seed", "near"]


class TestRetrieveMemory:
    @pytest.mark.asyncio
    async def test_combines_qdrant_seeds_with_typedb_neighbors(self):
        class FakeClient:
            async def query_points(self, **kwargs):
                del kwargs
                return SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            id="turn-1",
                            score=0.9,
                            payload={
                                "turn_id": "turn-1",
                                "content": "user asked about gateway writeback",
                                "source_uri": "chat://session-1#1",
                            },
                        )
                    ]
                )

        async def fake_embed(query: str) -> list[float]:
            assert query == "gateway writeback"
            return [0.1, 0.2]

        def fake_expand(seeds, max_depth):
            assert max_depth == 2
            assert seeds[0].node_id == "turn-1"
            return (
                RetrievedNode(
                    node_id="gateway/gateway.py",
                    kind="code",
                    content="gateway writeback worker scheduling",
                    score=0.75,
                    depth=1,
                    name="gateway/gateway.py",
                    source_uri="gateway/gateway.py",
                ),
            )

        result = await retrieve_memory(
            RetrievalRequest(
                query="gateway writeback",
                modes=("chat",),
                max_depth=2,
                token_budget=40,
                max_nodes=4,
            ),
            qdrant_client=FakeClient(),
            embed_query=fake_embed,
            typedb_expand=fake_expand,
        )

        assert [node.node_id for node in result.nodes] == ["turn-1", "gateway/gateway.py"]
        assert "Qdrant returned 1 seed matches" in (result.explanation or "")

    @pytest.mark.asyncio
    async def test_build_default_memory_retriever_wraps_retrieve_memory(self, monkeypatch):
        calls = {}

        async def fake_retrieve_memory(request, *, qdrant_client, embed_query, typedb_expand):
            calls["request"] = request
            calls["qdrant_client"] = qdrant_client
            calls["embed_query"] = embed_query
            calls["typedb_expand"] = typedb_expand
            return RetrievalResult(nodes=(), explanation="ok")

        client = object()
        monkeypatch.setattr("memory.retrieval.retrieve_memory", fake_retrieve_memory)
        retriever = build_default_memory_retriever(qdrant_client=client)

        result = await retriever(RetrievalRequest(query="hello"))

        assert calls["request"].query == "hello"
        assert calls["qdrant_client"] is client
        assert result.explanation == "ok"


class TestMemoryContext:
    def test_build_memory_context_accepts_retrieved_node_dicts(self):
        context = build_memory_context(
            [
                {"node_id": "turn-1", "name": "turn-1", "content": "chat turn", "depth": 0, "score": 0.9},
                {"node_id": "code-1", "name": "gateway.py", "content": "writeback", "depth": 1, "score": 0.8},
            ],
            token_budget=20,
        )

        assert context is not None
        assert "turn-1" in context
        assert "gateway.py" in context
