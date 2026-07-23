from __future__ import annotations

from typing import Any, Callable

from kortex.contracts import RetrievedNode
from memory import build_memory_service


def inspect_memory_node(
    node: RetrievedNode,
    *,
    max_depth: int = 1,
    memory_service: dict[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    service = memory_service or build_memory_service()
    entity = service["lookup_entity"](node)
    neighbors = service["lookup_neighbors"](node, max_depth=max_depth)
    return {
        "entity": entity,
        "neighbors": neighbors,
    }
