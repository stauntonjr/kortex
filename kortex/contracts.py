from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

RetrievalMode = Literal["code", "chat", "artifact"]
ReflectionJobType = Literal[
    "compact-session",
    "extract-directives",
    "promote-preferences",
    "repair-links",
]


@dataclass(frozen=True)
class TranscriptTurn:
    role: str
    content: str
    turn_id: str | None = None
    timestamp: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GatewayResult:
    request_id: str
    resolved_model: str
    response_text: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    usage: Mapping[str, int | float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptWritebackEvent:
    session_id: str
    source: str
    user_turn: TranscriptTurn
    assistant_turn: TranscriptTurn
    gateway_result: GatewayResult
    title: str | None = None
    source_uri: str | None = None
    previous_turn_count: int = 0
    tool_calls: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def ordered_turns(self) -> tuple[TranscriptTurn, TranscriptTurn]:
        return (self.user_turn, self.assistant_turn)


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    modes: tuple[RetrievalMode, ...] = ("code", "chat", "artifact")
    max_depth: int = 3
    token_budget: int = 1200
    max_nodes: int = 12
    session_id: str | None = None


@dataclass(frozen=True)
class RetrievedNode:
    node_id: str
    kind: str
    content: str
    score: float
    depth: int = 0
    name: str | None = None
    source_uri: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    nodes: tuple[RetrievedNode, ...]
    explanation: str | None = None


@dataclass(frozen=True)
class ReflectionJob:
    job_id: str
    job_type: ReflectionJobType
    subject_id: str
    created_at: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def transcript_turn_to_dict(turn: TranscriptTurn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "content": turn.content,
        "turn_id": turn.turn_id,
        "timestamp": _serialize_datetime(turn.timestamp),
        "metadata": dict(turn.metadata),
    }


def gateway_result_to_dict(result: GatewayResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "resolved_model": result.resolved_model,
        "response_text": result.response_text,
        "started_at": _serialize_datetime(result.started_at),
        "completed_at": _serialize_datetime(result.completed_at),
        "usage": dict(result.usage),
    }


def transcript_writeback_event_to_dict(event: TranscriptWritebackEvent) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "source": event.source,
        "user_turn": transcript_turn_to_dict(event.user_turn),
        "assistant_turn": transcript_turn_to_dict(event.assistant_turn),
        "gateway_result": gateway_result_to_dict(event.gateway_result),
        "title": event.title,
        "source_uri": event.source_uri,
        "previous_turn_count": event.previous_turn_count,
        "tool_calls": list(event.tool_calls),
        "metadata": dict(event.metadata),
    }
