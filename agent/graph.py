"""
Kortex Agent — LangGraph State Machine
========================================
Implements a two-node workflow:

  ``intake``  — classifies task complexity and selects the gateway model.
  ``execute`` — forwards the conversation to the gateway and captures the reply.

Usage::

    from agent.graph import kortex_graph

    result = await kortex_graph.ainvoke({
        "messages":    [{"role": "user", "content": "Fix the off-by-one error."}],
        "model_key":   "",
        "task_complexity": "",
        "gateway_url": "http://localhost:8080/v1",
        "response":    None,
    })
    print(result["response"])
"""

from __future__ import annotations

import logging

import httpx
from langgraph.graph import END, StateGraph

from kortex.contracts import RetrievalRequest
from memory import build_memory_service
from memory.retrieval import (
    DEFAULT_TOKEN_BUDGET,
    MAX_TRAVERSAL_DEPTH,
    build_memory_context,
    retrieval_result_to_memory_nodes,
)
from agent.state import COMPLEXITY_MAP, WorkflowState

logger = logging.getLogger(__name__)


def _last_user_content(messages: list[dict[str, str]]) -> str:
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    return (last_user or {}).get("content", "")


def _build_retrieval_request(state: WorkflowState) -> RetrievalRequest:
    existing = state.get("retrieval_request")
    if existing is not None:
        return existing
    return RetrievalRequest(
        query=_last_user_content(list(state.get("messages", []))),
        max_depth=state.get("memory_max_depth", MAX_TRAVERSAL_DEPTH),
        token_budget=state.get("memory_token_budget", DEFAULT_TOKEN_BUDGET),
    )

def _build_messages_with_context(state: WorkflowState) -> list[dict[str, str]]:
    messages = list(state.get("messages", []))
    memory_nodes = state.get("memory_nodes") or []
    if not memory_nodes:
        memory_nodes = retrieval_result_to_memory_nodes(state.get("retrieval_result"))
    if not memory_nodes:
        return messages

    memory_context = build_memory_context(
        memory_nodes,
        max_depth=state.get("memory_max_depth", MAX_TRAVERSAL_DEPTH),
        token_budget=state.get("memory_token_budget", DEFAULT_TOKEN_BUDGET),
    )
    if not memory_context:
        return messages
    return [{"role": "system", "content": memory_context}, *messages]


# ---------------------------------------------------------------------------
# Keyword sets used by the classifier
# ---------------------------------------------------------------------------

#: Words that strongly indicate a *simple* task (code fix / formatting).
_SIMPLE_KEYWORDS: frozenset[str] = frozenset(
    {
        "fix", "typo", "format", "rename", "lint", "indent",
        "comment", "syntax", "autocomplete", "complete",
    }
)

#: Words that strongly indicate a *complex* task (design / analysis).
_COMPLEX_KEYWORDS: frozenset[str] = frozenset(
    {
        "architect", "design", "plan", "analyze", "analyse",
        "refactor", "explain", "reason", "review", "evaluate",
        "compare", "optimize", "optimise",
    }
)

_SIMPLE_WORD_THRESHOLD: int = 30    # fewer words → simple
_COMPLEX_WORD_THRESHOLD: int = 150  # more words  → complex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(content: str) -> str:
    """Return ``"simple"``, ``"moderate"``, or ``"complex"`` for *content*."""
    words = content.lower().split()
    word_set = set(words)

    # Keyword checks take priority over word-count thresholds
    if word_set & _COMPLEX_KEYWORDS:
        return "complex"
    if word_set & _SIMPLE_KEYWORDS or len(words) < _SIMPLE_WORD_THRESHOLD:
        return "simple"
    if len(words) > _COMPLEX_WORD_THRESHOLD:
        return "complex"
    return "moderate"


# ---------------------------------------------------------------------------
# Node: intake / classify
# ---------------------------------------------------------------------------

def intake_node(state: WorkflowState) -> dict:
    """Inspect the last user message, classify complexity, and pick a model.

    Returns a partial state update containing ``task_complexity`` and
    ``model_key``.
    """
    messages = state.get("messages") or []
    content = _last_user_content(messages)
    complexity = _classify(content)
    model_key = COMPLEXITY_MAP[complexity]

    logger.info(
        "Intake: complexity=%s → model=%s (prompt_words=%d)",
        complexity,
        model_key,
        len(content.split()),
    )

    return {"task_complexity": complexity, "model_key": model_key}


def make_retrieve_node(default_retriever=None):
    async def _retrieve_node(state: WorkflowState) -> dict:
        retriever = state.get("memory_retriever") or default_retriever
        if retriever is None:
            return {}
        request = _build_retrieval_request(state)
        result = await retriever(request)
        return {
            "retrieval_request": request,
            "retrieval_result": result,
            "memory_nodes": retrieval_result_to_memory_nodes(result),
        }

    return _retrieve_node


async def retrieve_node(state: WorkflowState) -> dict:
    retriever = state.get("memory_retriever")
    if retriever is None:
        return {}
    return await make_retrieve_node()(state)


# ---------------------------------------------------------------------------
# Node: execute
# ---------------------------------------------------------------------------

async def execute_node(state: WorkflowState) -> dict:
    """Forward the conversation to the selected model via the Kortex gateway.

    Returns a partial state update containing ``response``.
    """
    gateway_url = state.get("gateway_url", "http://localhost:8080/v1").rstrip("/")
    url = f"{gateway_url}/chat/completions"
    payload: dict = {
        "model":    state["model_key"],
        "messages": _build_messages_with_context(state),
    }

    logger.info("Executing via gateway model '%s' at %s", state["model_key"], url)

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

    content: str = resp.json()["choices"][0]["message"]["content"]
    return {"response": content}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(*, memory_retriever=None) -> StateGraph:
    """Construct and compile the Kortex LangGraph workflow.

    Graph topology::

        [intake] → [execute] → END
    """
    graph: StateGraph = StateGraph(WorkflowState)

    graph.add_node("intake",  intake_node)
    graph.add_node("retrieve", make_retrieve_node(memory_retriever))
    graph.add_node("execute", execute_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake",  "retrieve")
    graph.add_edge("retrieve", "execute")
    graph.add_edge("execute", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph (import-convenient singleton)
# ---------------------------------------------------------------------------

#: Ready-to-use compiled LangGraph for the Kortex agent workflow.
kortex_graph = build_graph(memory_retriever=build_memory_service()["retrieve"])
