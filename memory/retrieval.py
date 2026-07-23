from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx
from qdrant_client import AsyncQdrantClient
from typedb.driver import TransactionType, TypeDB

from kortex.contracts import RetrievalRequest, RetrievalResult, RetrievedNode
from memory.chat_ingest import CHAT_EMBEDDING_MODEL, CHAT_QDRANT_COLLECTION
from memory.ingest import (
    EMBEDDING_URL,
    QDRANT_COLLECTION,
    TYPEDB_ADDR,
    TYPEDB_DATABASE,
    build_typedb_credentials,
    build_typedb_driver_options,
    typedb_transaction,
)

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

_KIND_ENTITY_MAP = {
    "chat": "chat-turn",
    "code": "code-entity",
    "artifact": "project-artifact",
}

_KIND_ID_ATTRIBUTE = {
    "chat": "turn-id",
    "code": "entity-id",
    "artifact": "artifact-id",
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


def retrieval_result_to_memory_nodes(result: RetrievalResult | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    return [asdict(node) for node in result.nodes]


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


@contextmanager
def open_default_typedb_driver():
    creds = build_typedb_credentials()
    options = build_typedb_driver_options()
    if creds is not None:
        with TypeDB.driver(TYPEDB_ADDR, creds, options) as driver:
            yield driver
        return
    with TypeDB.driver(TYPEDB_ADDR, options) as driver:
        yield driver


def _quote_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def build_seed_lookup_query(node: RetrievedNode) -> str:
    entity = _KIND_ENTITY_MAP.get(node.kind)
    attr = _KIND_ID_ATTRIBUTE.get(node.kind)
    if entity is None or attr is None:
        raise ValueError(f"unsupported retrieval node kind: {node.kind}")
    return "\n".join(
        [
            "match",
            f"  $node isa {entity}, has {attr} {_quote_string(node.node_id)};",
            "fetch",
            "  $node: content-text,",
            "  $node: entity-name,",
            "  $node: entity-path,",
            "  $node: source-uri,",
            "  $node: speaker-role,",
            "  $node: artifact-name;",
        ]
    )


def build_neighbor_expansion_query(node: RetrievedNode) -> str:
    entity = _KIND_ENTITY_MAP.get(node.kind)
    attr = _KIND_ID_ATTRIBUTE.get(node.kind)
    if entity is None or attr is None:
        raise ValueError(f"unsupported retrieval node kind: {node.kind}")
    relation_line = "  (source-turn: $seed, target: $neighbor) isa mention;"
    if node.kind in {"code", "artifact"}:
        relation_line = "  (source-turn: $neighbor, target: $seed) isa mention;"
    return "\n".join(
        [
            "match",
            f"  $seed isa {entity}, has {attr} {_quote_string(node.node_id)};",
            relation_line,
            "fetch",
            "  $neighbor: turn-id,",
            "  $neighbor: entity-id,",
            "  $neighbor: artifact-id,",
            "  $neighbor: content-text,",
            "  $neighbor: entity-name,",
            "  $neighbor: entity-path,",
            "  $neighbor: artifact-name,",
            "  $neighbor: source-uri;",
        ]
    )


def _infer_kind_from_payload(payload: Mapping[str, Any]) -> str | None:
    if payload.get("turn-id"):
        return "chat"
    if payload.get("entity-id"):
        return "code"
    if payload.get("artifact-id"):
        return "artifact"
    return None


def _node_from_typedb_payload(
    payload: Mapping[str, Any],
    *,
    fallback: RetrievedNode | None,
    score: float,
    depth: int,
) -> RetrievedNode | None:
    kind = _infer_kind_from_payload(payload) or (fallback.kind if fallback else None)
    if kind is None:
        return None
    node_id = str(
        payload.get("turn-id")
        or payload.get("entity-id")
        or payload.get("artifact-id")
        or (fallback.node_id if fallback else "")
    )
    if not node_id:
        return None
    content = str(payload.get("content-text") or (fallback.content if fallback else ""))
    name = (
        payload.get("entity-name")
        or payload.get("artifact-name")
        or payload.get("entity-path")
        or payload.get("source-uri")
        or (fallback.name if fallback else None)
    )
    source_uri = payload.get("source-uri") or payload.get("entity-path") or (fallback.source_uri if fallback else None)
    return RetrievedNode(
        node_id=node_id,
        kind=kind,
        content=content,
        score=score,
        depth=depth,
        name=None if name is None else str(name),
        source_uri=None if source_uri is None else str(source_uri),
    )


def _normalize_typedb_rows(result: Any) -> list[Mapping[str, Any]]:
    rows = list(result)
    normalized: list[Mapping[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(row)
            continue
        as_dict = getattr(row, "as_dict", None)
        if callable(as_dict):
            normalized.append(as_dict())
            continue
        getter = getattr(row, "get", None)
        if callable(getter):
            normalized.append({key: getter(key) for key in ("turn-id", "entity-id", "artifact-id", "content-text", "entity-name", "entity-path", "artifact-name", "source-uri")})
    return normalized


def typedb_expand_with_driver(
    driver: Any,
    seeds: Sequence[RetrievedNode],
    max_depth: int,
    *,
    database: str = TYPEDB_DATABASE,
) -> Sequence[RetrievedNode]:
    if max_depth <= 0 or not seeds:
        return ()

    expanded: dict[str, RetrievedNode] = {}
    with typedb_transaction(driver, database, TransactionType.READ) as tx:
        for seed in seeds:
            for row in _normalize_typedb_rows(tx.query(build_seed_lookup_query(seed))):
                hydrated = _node_from_typedb_payload(row, fallback=seed, score=seed.score, depth=0)
                if hydrated is not None:
                    expanded[hydrated.node_id] = hydrated
            if max_depth < 1:
                continue
            for row in _normalize_typedb_rows(tx.query(build_neighbor_expansion_query(seed))):
                neighbor = _node_from_typedb_payload(row, fallback=None, score=max(seed.score * 0.85, 0.0), depth=1)
                if neighbor is not None and neighbor.node_id != seed.node_id:
                    expanded.setdefault(neighbor.node_id, neighbor)
    return tuple(expanded.values())


def default_typedb_expand(
    seeds: Sequence[RetrievedNode],
    max_depth: int,
) -> Sequence[RetrievedNode]:
    if max_depth <= 0 or not seeds:
        return ()
    with open_default_typedb_driver() as driver:
        return typedb_expand_with_driver(driver, seeds, max_depth)


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


def build_default_memory_retriever(
    *,
    qdrant_client: Any | None = None,
    typedb_expand: TypeDBExpandFunction = default_typedb_expand,
    embed_query: EmbeddingFunction = embed_query_text,
) -> Callable[[RetrievalRequest], Awaitable[RetrievalResult]]:
    client = qdrant_client or AsyncQdrantClient(url=os.getenv("QDRANT_ADDR", "http://localhost:6333"))

    async def _retriever(request: RetrievalRequest) -> RetrievalResult:
        return await retrieve_memory(
            request,
            qdrant_client=client,
            embed_query=embed_query,
            typedb_expand=typedb_expand,
        )

    return _retriever
