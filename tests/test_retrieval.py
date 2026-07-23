from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")

from kortex.contracts import RetrievalRequest, RetrievedNode
from memory.retrieval import (
    build_memory_context,
    merge_retrieval_nodes,
    qdrant_candidate_search,
    retrieve_memory,
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
