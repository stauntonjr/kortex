from __future__ import annotations

from agent import inspect_memory_node
from kortex.contracts import RetrievedNode


class TestAgentMemory:
    def test_inspect_memory_node_uses_memory_service_surface(self):
        node = RetrievedNode(
            node_id="turn-1",
            kind="chat",
            content="remember this",
            score=0.9,
            depth=0,
            name="turn-1",
        )
        neighbor = RetrievedNode(
            node_id="gateway/gateway.py",
            kind="code",
            content="writeback code",
            score=0.8,
            depth=1,
            name="gateway/gateway.py",
        )

        service = {
            "lookup_entity": lambda value: node if value.node_id == "turn-1" else None,
            "lookup_neighbors": lambda value, *, max_depth=1: (neighbor,) if value.node_id == "turn-1" and max_depth == 2 else (),
        }

        result = inspect_memory_node(node, max_depth=2, memory_service=service)

        assert result["entity"] == node
        assert result["neighbors"] == (neighbor,)
