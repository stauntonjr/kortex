from __future__ import annotations

import pytest

from kortex.contracts import RetrievalRequest, RetrievalResult, RetrievedNode
from memory import service


class TestMemoryService:
    @pytest.mark.asyncio
    async def test_build_memory_service_uses_provided_retriever(self):
        async def fake_retriever(request):
            assert request.query == "gateway writeback"
            return RetrievalResult(nodes=(), explanation="ok")

        svc = service.build_memory_service(memory_retriever=fake_retriever)
        result = await svc["retrieve"](RetrievalRequest(query="gateway writeback"))

        assert result.explanation == "ok"

    @pytest.mark.asyncio
    async def test_retrieve_uses_provided_retriever(self):
        async def fake_retriever(request):
            assert request.query == "hello"
            return RetrievalResult(nodes=(), explanation="done")

        result = await service.retrieve(
            RetrievalRequest(query="hello"),
            memory_retriever=fake_retriever,
        )

        assert result.explanation == "done"

    def test_lookup_helpers_delegate_to_retrieval_module(self, monkeypatch):
        node = RetrievedNode(node_id="turn-1", kind="chat", content="", score=0.9)
        neighbor = RetrievedNode(node_id="code-1", kind="code", content="ctx", score=0.8, depth=1)

        monkeypatch.setattr(service, "lookup_entity", lambda value: node if value.node_id == "turn-1" else None)
        monkeypatch.setattr(
            service,
            "lookup_neighbors",
            lambda value, *, max_depth=1: (neighbor,) if value.node_id == "turn-1" and max_depth == 2 else (),
        )

        assert service.lookup_entity_by_node(node) == node
        assert service.lookup_neighbors_by_node(node, max_depth=2) == (neighbor,)
