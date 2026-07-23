"""
Kortex Agent — Workflow State
==============================
Defines the LangGraph ``WorkflowState`` TypedDict that flows through every
node in the Kortex orchestration graph.

Fields
------
messages
    Conversation history as a list of ``{"role": ..., "content": ...}`` dicts.
    Appended to (never replaced) by the LangGraph reducer.
model_key
    Canonical gateway model key selected by the intake node (e.g.
    ``"qwen-coder"``, ``"qwen-35b"``, ``"nemotron-120b"``).
task_complexity
    Complexity tier: ``"simple"`` | ``"moderate"`` | ``"complex"``.
gateway_url
    Base URL for the Kortex gateway, e.g. ``"http://localhost:8080/v1"``.
response
    Final assistant response text, set by the execute node.
memory_nodes
    Optional hypergraph retrieval nodes to prune before sending to the model.
"""

from __future__ import annotations

from typing import Annotated, Any, Awaitable, Callable
from typing_extensions import NotRequired, TypedDict

from kortex.contracts import RetrievalRequest, RetrievalResult

MemoryRetriever = Callable[[RetrievalRequest], Awaitable[RetrievalResult]]

# ---------------------------------------------------------------------------
# Complexity → gateway model mapping
# ---------------------------------------------------------------------------

#: Maps each complexity tier to its canonical gateway model key.
COMPLEXITY_MAP: dict[str, str] = {
    "simple":   "qwen-coder",       # ~52.5 GB, fast code completions
    "moderate": "qwen-35b",          # ~21.8 GB, MoE reasoning
    "complex":  "nemotron-120b",     # ~90 GB,   deep planning / analysis
}


# ---------------------------------------------------------------------------
# State reducer
# ---------------------------------------------------------------------------

def _append_messages(existing: list, new: list) -> list:
    """Reducer that appends new messages to the existing list."""
    return (existing or []) + (new or [])


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    """Shared state passed between every node of the Kortex LangGraph."""

    #: Conversation history — appended to, never replaced.
    messages: Annotated[list[dict[str, str]], _append_messages]

    #: Canonical model key resolved by the intake/classify node.
    model_key: str

    #: Complexity classification: "simple" | "moderate" | "complex".
    task_complexity: str

    #: Kortex gateway base URL (e.g. ``"http://localhost:8080/v1"``).
    gateway_url: str

    #: Final assistant response produced by the execute node (None until set).
    response: str | None

    #: Optional TypeDB/GraphRAG nodes that can be injected as bounded context.
    memory_nodes: NotRequired[list[dict[str, Any]]]

    #: Optional traversal-depth override for memory_nodes.
    memory_max_depth: NotRequired[int]

    #: Optional token-budget override for memory_nodes.
    memory_token_budget: NotRequired[int]

    #: Optional typed retrieval result for memory context assembly.
    retrieval_result: NotRequired[RetrievalResult]

    #: Optional explicit retrieval request for the memory plane.
    retrieval_request: NotRequired[RetrievalRequest]

    #: Optional memory retrieval service injected by the orchestrator.
    memory_retriever: NotRequired[MemoryRetriever]
