"""
Kortex Agent — Unit Tests
==========================
Tests the classify heuristic, intake node, and graph compilation without
requiring a live gateway, TypeDB, or Qdrant.

Run with::

    pytest tests/test_agent.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph import (
    _classify,
    _build_gateway_messages,
    _COMPLEX_WORD_THRESHOLD,
    _SIMPLE_WORD_THRESHOLD,
    build_graph,
    intake_node,
)
from agent.state import COMPLEXITY_MAP, WorkflowState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(content: str, gateway_url: str = "http://localhost:8080/v1") -> WorkflowState:
    return WorkflowState(
        messages=[{"role": "user", "content": content}],
        model_key="",
        task_complexity="",
        gateway_url=gateway_url,
        response=None,
    )


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_short_prompt_is_simple(self):
        assert _classify("fix typo") == "simple"

    def test_empty_prompt_is_simple(self):
        assert _classify("") == "simple"

    def test_simple_keyword_triggers_simple(self):
        assert _classify("please format this function") == "simple"

    def test_lint_keyword_is_simple(self):
        assert _classify("run the linter on this file please") == "simple"

    def test_complex_keyword_triggers_complex(self):
        result = _classify("analyze and design a comprehensive architecture")
        assert result == "complex"

    def test_reason_keyword_is_complex(self):
        assert _classify("reason through the tradeoffs in this approach") == "complex"

    def test_long_prompt_is_complex(self):
        prompt = " ".join(["neutral"] * (_COMPLEX_WORD_THRESHOLD + 10))
        assert _classify(prompt) == "complex"

    def test_moderate_prompt(self):
        # Between thresholds, no strong keywords
        prompt = " ".join(["neutral"] * ((_SIMPLE_WORD_THRESHOLD + _COMPLEX_WORD_THRESHOLD) // 2))
        assert _classify(prompt) == "moderate"

    def test_boundary_just_below_simple_threshold(self):
        prompt = " ".join(["word"] * (_SIMPLE_WORD_THRESHOLD - 1))
        assert _classify(prompt) == "simple"

    def test_boundary_just_above_complex_threshold(self):
        prompt = " ".join(["word"] * (_COMPLEX_WORD_THRESHOLD + 1))
        assert _classify(prompt) == "complex"


# ---------------------------------------------------------------------------
# intake_node
# ---------------------------------------------------------------------------


class TestIntakeNode:
    def test_simple_task_selects_qwen_coder(self):
        state = _make_state("fix the off-by-one error")
        result = intake_node(state)
        assert result["task_complexity"] == "simple"
        assert result["model_key"] == "qwen-coder"

    def test_moderate_task_selects_qwen_35b(self):
        state = _make_state(" ".join(["implement"] * 80))
        result = intake_node(state)
        assert result["task_complexity"] == "moderate"
        assert result["model_key"] == "qwen-35b"

    def test_complex_task_selects_nemotron(self):
        content = "analyze and design " + " ".join(["architecture"] * 160)
        state = _make_state(content)
        result = intake_node(state)
        assert result["task_complexity"] == "complex"
        assert result["model_key"] == "nemotron-120b"

    def test_empty_messages_defaults_to_simple(self):
        state = WorkflowState(
            messages=[],
            model_key="",
            task_complexity="",
            gateway_url="http://localhost:8080/v1",
            response=None,
        )
        result = intake_node(state)
        assert result["task_complexity"] == "simple"

    def test_only_assistant_messages_defaults_to_simple(self):
        state = WorkflowState(
            messages=[{"role": "assistant", "content": "Sure, here is the fix."}],
            model_key="",
            task_complexity="",
            gateway_url="http://localhost:8080/v1",
            response=None,
        )
        result = intake_node(state)
        assert result["task_complexity"] == "simple"

    def test_last_user_message_used_for_classification(self):
        # First message is complex, last is simple — should pick simple
        state = WorkflowState(
            messages=[
                {"role": "user", "content": "analyze design architecture " + " ".join(["x"] * 200)},
                {"role": "assistant", "content": "Here is my analysis."},
                {"role": "user", "content": "fix typo"},
            ],
            model_key="",
            task_complexity="",
            gateway_url="http://localhost:8080/v1",
            response=None,
        )
        result = intake_node(state)
        assert result["task_complexity"] == "simple"

    def test_returns_only_partial_state_keys(self):
        state = _make_state("rename this variable")
        result = intake_node(state)
        assert set(result.keys()) == {"task_complexity", "model_key"}


# ---------------------------------------------------------------------------
# COMPLEXITY_MAP
# ---------------------------------------------------------------------------


class TestComplexityMap:
    def test_all_tiers_present(self):
        for tier in ("simple", "moderate", "complex"):
            assert tier in COMPLEXITY_MAP

    def test_all_values_are_valid_model_keys(self):
        valid = {"qwen-coder", "qwen-35b", "nemotron-120b"}
        for model in COMPLEXITY_MAP.values():
            assert model in valid

    def test_simple_is_coder(self):
        assert COMPLEXITY_MAP["simple"] == "qwen-coder"

    def test_moderate_is_qwen35b(self):
        assert COMPLEXITY_MAP["moderate"] == "qwen-35b"

    def test_complex_is_nemotron(self):
        assert COMPLEXITY_MAP["complex"] == "nemotron-120b"


# ---------------------------------------------------------------------------
# build_graph / graph compilation
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_intake_and_execute_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "intake" in node_names
        assert "execute" in node_names

    @pytest.mark.asyncio
    async def test_graph_invocation_calls_intake_then_execute(self):
        """End-to-end smoke test with a mocked HTTP execute node."""
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "choices": [{"message": {"content": "done"}}]
        }

        with patch("agent.graph.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)

            graph = build_graph()
            result = await graph.ainvoke({
                "messages":        [{"role": "user", "content": "fix typo"}],
                "model_key":       "",
                "task_complexity": "",
                "gateway_url":     "http://localhost:8080/v1",
                "response":        None,
            })

        assert result["response"] == "done"
        assert result["model_key"] == "qwen-coder"
        assert result["task_complexity"] == "simple"


class TestMemoryContext:
    def test_memory_context_is_depth_limited_and_pruned_to_token_budget(self):
        state = _make_state("fix typo")
        state["memory_nodes"] = [
            {"name": "deep", "content": "ignored", "depth": 4, "score": 1.0},
            {"name": "fit", "content": "A" * 20, "depth": 1, "score": 0.9},
            {"name": "overflow", "content": "B" * 80, "depth": 2, "score": 0.8},
        ]
        state["memory_max_depth"] = 3
        state["memory_token_budget"] = 12

        messages = _build_gateway_messages(state)

        assert messages[0]["role"] == "system"
        assert "fit" in messages[0]["content"]
        assert "overflow" not in messages[0]["content"]
        assert "deep" not in messages[0]["content"]

    def test_memory_context_returns_messages_unchanged_when_all_nodes_are_filtered(self):
        state = _make_state("fix typo")
        state["memory_nodes"] = [{"name": "deep", "content": "ignored", "depth": 4, "score": 1.0}]

        assert _build_gateway_messages(state) == state["messages"]
