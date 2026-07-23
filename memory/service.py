from __future__ import annotations

from typing import Any, Awaitable, Callable

from kortex.contracts import RetrievalRequest, RetrievalResult, RetrievedNode
from memory.retrieval import (
    build_default_memory_retriever,
    lookup_entity,
    lookup_neighbors,
)

MemoryRetriever = Callable[[RetrievalRequest], Awaitable[RetrievalResult]]


def build_memory_service(
    *,
    memory_retriever: MemoryRetriever | None = None,
) -> dict[str, Any]:
    retriever = memory_retriever or build_default_memory_retriever()
    return {
        "retrieve": retriever,
        "lookup_entity": lookup_entity,
        "lookup_neighbors": lookup_neighbors,
    }


async def retrieve(
    request: RetrievalRequest,
    *,
    memory_retriever: MemoryRetriever | None = None,
) -> RetrievalResult:
    retriever = memory_retriever or build_default_memory_retriever()
    return await retriever(request)


def lookup_entity_by_node(node: RetrievedNode) -> RetrievedNode | None:
    return lookup_entity(node)


def lookup_neighbors_by_node(
    node: RetrievedNode,
    *,
    max_depth: int = 1,
) -> tuple[RetrievedNode, ...]:
    return lookup_neighbors(node, max_depth=max_depth)
