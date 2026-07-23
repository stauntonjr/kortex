from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from kortex.contracts import RetrievalRequest, RetrievalResult, RetrievedNode
from memory.chat_ingest import CHAT_EMBEDDING_MODEL, CHAT_QDRANT_COLLECTION
from memory.ingest import EMBEDDING_URL, QDRANT_COLLECTION

MAX_TRAVERSAL_DEPTH = int(os.getenv("KORTEX_GRAPHRAG_MAX_DEPTH", "3"))
DEFAULT_TOKEN_BUDGET = int(os.getenv("KORTEX_GRAPHRAG_TOKEN_BUDGET", "1200"))
DEFAULT_MAX_CANDIDATES = int(os.getenv("KORTEX_RETRIEVAL_MAX_CANDIDATES", "6"))
ARTIFACT_QDRANT_COLLECTION = os.getenv("KORTEX_ARTIFACT_QDRANT_COLLECTION", "kortex_artifact")

EmbeddingFunction = Callable[[str], Awaitable[list[float]]]
QdrantSearchFunction = Callable[..., Awaitable[Sequence[Any]]]
TypeDBExpandFunction = Callable[[Sequence[RetrievedNode], int], Sequence[RetrievedNode]]

_MODE_COLLECTIONS = {
    "code": QDRANT_COLLECTION,
    "chat": CHAT_QDRANT_COLLECTION,
    "artifact": ARTIFACT_QDRANT_COLLECTION,
}


def estimate_token_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def limit_traversal_depth(
    nodes: Sequence[Mapping[str, Any]],
    *,
    max_depth: int = MAX_TRAVERSAL_DEPTH,
) -> list[dict[str, Any]]:
    return [dict(node) for node in nodes if int(node.get("depth", 0)) <= max_depth]


def prune_nodes_to_token_budget(
    nodes: Sequence[Mapping[str, Any]],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    if token_budget <= 0:
        return []

    ordered = sorted(
        (dict(node) for node in nodes),
        key=lambda node: (
            int(node.get("depth", 0)),
            -float(node.get("score", 0) or 0),
            str(node.get("node_id") or node.get("entity_id") or node.get("name") or ""),
        ),
    )

    pruned: list[dict[str, Any]] = []
    tokens_used = 0
    for node in ordered:
        token_count = max(
            0,
            int(node.get("token_count") or estimate_token_count(str(node.get("content", "")))),
        )
        if token_count == 0 or tokens_used + token_count > token_budget:
            continue
        node["token_count"] = token_count
        pruned.append(node)
        tokens_used += token_count
    return pruned


def build_memory_context(
    nodes: Sequence[Mapping[str, Any]],
    *,
    max_depth: int = MAX_TRAVERSAL_DEPTH,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str | None:
    bounded_nodes = prune_nodes_to_token_budget(
        limit_traversal_depth(nodes, max_depth=max_depth),
        token_budget=token_budget,
    )
    if not bounded_nodes:
        return None

    lines = ["Relevant repository memory:"]
    for node in bounded_nodes:
        label = node.get("name") or node.get("node_id") or node.get("entity_id") or "node"
        content = str(node.get("content", "")).strip()
        lines.append(f"- depth={int(node.get('depth', 0))} {label}: {content}")
    return "\n".join(lines)


async def embed_query_text(query: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{EMBEDDING_URL}/embeddings",
            json={"model": CHAT_EMBEDDING_MODEL, "input": [query]},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


def _collection_for_mode(mode: str) -> str:
    return _MODE_COLLECTIONS[mode]


def _coerce_search_results(results: Any) -> Sequence[Any]:
    if hasattr(results, "points"):
        return list(results.points)
    return list(results)


def _coerce_payload(point: Any) -> dict[str, Any]:
    payload = getattr(point, "payload", {}) or {}
    return dict(payload)


def _coerce_score(point: Any) -> float:
    score = getattr(point, "score", 0.0)
    try:
        return float(score)
    except Exception:
        return 0.0


def _candidate_to_node(kind: str, point: Any) -> RetrievedNode:
    payload = _coerce_payload(point)
    node_id = str(
        payload.get("turn_id")
        or payload.get("entity_id")
        or payload.get("artifact_id")
        or getattr(point, "id", "")
    )
    return RetrievedNode(
        node_id=node_id,
        kind=kind,
        content=str(payload.get("content", "")),
        score=_coerce_score(point),
        depth=0,
        name=payload.get("name") or payload.get("entity_name") or payload.get("path"),
        source_uri=payload.get("source_uri") or payload.get("path"),
    )


async def qdrant_candidate_search(
    client: Any,
    request: RetrievalRequest,
    *,
    query_vector: list[float],
) -> tuple[RetrievedNode, ...]:
    nodes: list[RetrievedNode] = []

    for mode in request.modes:
        collection = _collection_for_mode(mode)
        limit = max(1, request.max_nodes)
        query_filter = None
        if mode == "chat" and request.session_id:
            query_filter = {"must": [{"key": "session_id", "match": {"value": request.session_id}}]}

        if hasattr(client, "query_points"):
            results = await client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
        else:
            results = await client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
            )

        nodes.extend(_candidate_to_node(mode, point) for point in _coerce_search_results(results))

    deduped: dict[str, RetrievedNode] = {}
    for node in sorted(nodes, key=lambda item: (-item.score, item.node_id)):
        deduped.setdefault(node.node_id, node)
    return tuple(list(deduped.values())[: request.max_nodes])


def _mapping_value(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    getter = getattr(mapping, "get", None)
    if callable(getter):
        try:
            value = getter(key)
            return default if value is None else value
        except Exception:
            return default
    return default


def default_typedb_expand(
    seeds: Sequence[RetrievedNode],
    max_depth: int,
) -> Sequence[RetrievedNode]:
    del seeds, max_depth
    return ()


def merge_retrieval_nodes(
    seeds: Sequence[RetrievedNode],
    neighbors: Sequence[RetrievedNode],
    *,
    max_depth: int,
    max_nodes: int,
    token_budget: int,
) -> tuple[RetrievedNode, ...]:
    merged: dict[str, RetrievedNode] = {}
    for node in [*seeds, *neighbors]:
        existing = merged.get(node.node_id)
        if existing is None or (node.score, -node.depth) > (existing.score, -existing.depth):
            merged[node.node_id] = node

    bounded = prune_nodes_to_token_budget(
        [asdict(node) for node in merged.values() if node.depth <= max_depth],
        token_budget=token_budget,
    )
    ordered = sorted(
        bounded,
        key=lambda node: (int(node.get("depth", 0)), -float(node.get("score", 0)), node.get("node_id", "")),
    )
    return tuple(RetrievedNode(**{k: node[k] for k in RetrievedNode.__dataclass_fields__.keys() if k in node}) for node in ordered[:max_nodes])


def explain_retrieval(
    request: RetrievalRequest,
    seeds: Sequence[RetrievedNode],
    neighbors: Sequence[RetrievedNode],
    final_nodes: Sequence[RetrievedNode],
) -> str:
    return (
        f"Retrieved {len(final_nodes)} nodes for modes={','.join(request.modes)}; "
        f"Qdrant returned {len(seeds)} seed matches and TypeDB expansion added {len(neighbors)} neighbors."
    )


async def retrieve_memory(
    request: RetrievalRequest,
    *,
    qdrant_client: Any,
    embed_query: EmbeddingFunction = embed_query_text,
    typedb_expand: TypeDBExpandFunction = default_typedb_expand,
) -> RetrievalResult:
    query_vector = await embed_query(request.query)
    seeds = await qdrant_candidate_search(qdrant_client, request, query_vector=query_vector)
    neighbors = tuple(typedb_expand(seeds, min(request.max_depth, MAX_TRAVERSAL_DEPTH)))
    nodes = merge_retrieval_nodes(
        seeds,
        neighbors,
        max_depth=request.max_depth,
        max_nodes=request.max_nodes,
        token_budget=request.token_budget,
    )
    return RetrievalResult(
        nodes=nodes,
        explanation=explain_retrieval(request, seeds, neighbors, nodes),
    )
