from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

MAX_TRAVERSAL_DEPTH = int(os.getenv("KORTEX_GRAPHRAG_MAX_DEPTH", "3"))
DEFAULT_TOKEN_BUDGET = int(os.getenv("KORTEX_GRAPHRAG_TOKEN_BUDGET", "1200"))


def estimate_token_count(text: str) -> int:
    """Approximate token counts assuming roughly four characters per token."""
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
    """Keep bounded nodes within budget and skip empty nodes that estimate to zero tokens."""
    if token_budget <= 0:
        return []

    ordered = sorted(
        (dict(node) for node in nodes),
        key=lambda node: (
            int(node.get("depth", 0)),
            -float(node.get("score", 0) or 0),
            str(node.get("entity_id") or node.get("name") or ""),
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
    """Format bounded graph nodes into a single system-context string."""
    bounded_nodes = prune_nodes_to_token_budget(
        limit_traversal_depth(nodes, max_depth=max_depth),
        token_budget=token_budget,
    )
    if not bounded_nodes:
        return None

    lines = ["Relevant repository memory:"]
    for node in bounded_nodes:
        label = node.get("name") or node.get("entity_id") or "node"
        content = str(node.get("content", "")).strip()
        lines.append(f"- depth={int(node.get('depth', 0))} {label}: {content}")
    return "\n".join(lines)
