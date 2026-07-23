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
