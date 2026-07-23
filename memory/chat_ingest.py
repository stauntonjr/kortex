from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client.http.models import PointStruct

from kortex.contracts import GatewayResult, TranscriptTurn, TranscriptWritebackEvent
from memory.ingest import (
    ArtifactRecord,
    ChatSessionRecord,
    ChatTurnRecord,
    DirectiveRecord,
    build_chat_memory_queries,
)

_FILE_REFERENCE_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9_]+")
_DIRECTIVE_PREFIXES = (
    "always ",
    "never ",
    "prefer ",
    "avoid ",
    "use ",
    "keep ",
    "must ",
    "do not ",
    "don't ",
)
_DIRECTIVE_CONSTRAINT_MARKERS = ("must", "never", "do not", "don't", "avoid")
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/stauntonjr/kortex/chat-ingest")


@dataclass(frozen=True)
class ChatEmbeddingDocument:
    vector_id: str
    turn_id: str
    content: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class TranscriptIngestionBatch:
    session: ChatSessionRecord
    turns: tuple[ChatTurnRecord, ...]
    directives: tuple[DirectiveRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]
    queries: tuple[str, ...]
    embedding_documents: tuple[ChatEmbeddingDocument, ...]


def _stable_id(kind: str, value: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, f"{kind}:{value}"))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_writeback_event(path: str | Path) -> TranscriptWritebackEvent:
    payload = json.loads(Path(path).read_text())

    def build_turn(raw: dict[str, Any]) -> TranscriptTurn:
        return TranscriptTurn(
            role=raw["role"],
            content=raw["content"],
            turn_id=raw.get("turn_id"),
            timestamp=_parse_timestamp(raw.get("timestamp")),
            metadata=raw.get("metadata", {}),
        )

    raw_gateway = payload["gateway_result"]
    gateway_result = GatewayResult(
        request_id=raw_gateway["request_id"],
        resolved_model=raw_gateway["resolved_model"],
        response_text=raw_gateway["response_text"],
        started_at=_parse_timestamp(raw_gateway.get("started_at")),
        completed_at=_parse_timestamp(raw_gateway.get("completed_at")),
        usage=raw_gateway.get("usage", {}),
    )
    return TranscriptWritebackEvent(
        session_id=payload["session_id"],
        source=payload["source"],
        user_turn=build_turn(payload["user_turn"]),
        assistant_turn=build_turn(payload["assistant_turn"]),
        gateway_result=gateway_result,
        title=payload.get("title"),
        source_uri=payload.get("source_uri"),
        previous_turn_count=int(payload.get("previous_turn_count", 0)),
        tool_calls=tuple(payload.get("tool_calls", [])),
        metadata=payload.get("metadata", {}),
    )


def extract_artifact_paths(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0) for match in _FILE_REFERENCE_RE.finditer(text)))


def classify_directive(text: str) -> str | None:
    stripped = text.strip().lower()
    if not stripped:
        return None
    if not any(prefix in stripped for prefix in _DIRECTIVE_PREFIXES):
        return None
    if any(marker in stripped for marker in _DIRECTIVE_CONSTRAINT_MARKERS):
        return "constraint"
    return "workflow"


def build_ingestion_batch(event: TranscriptWritebackEvent) -> TranscriptIngestionBatch:
    session_source_uri = event.source_uri or f"{event.source}://{event.session_id}"
    session = ChatSessionRecord(
        session_id=event.session_id,
        title=event.title,
        source_uri=session_source_uri,
    )

    turns: list[ChatTurnRecord] = []
    directives: list[DirectiveRecord] = []
    artifacts_by_id: dict[str, ArtifactRecord] = {}
    embedding_documents: list[ChatEmbeddingDocument] = []

    for offset, turn in enumerate(event.ordered_turns(), start=1):
        turn_index = event.previous_turn_count + offset
        turn_id = turn.turn_id or _stable_id(
            "turn",
            f"{event.session_id}:{turn_index}:{turn.role}:{turn.content}",
        )
        turn_source_uri = f"{session_source_uri}#{turn_id}"
        artifact_ids: list[str] = []
        for artifact_path in extract_artifact_paths(turn.content):
            artifact_id = _stable_id("artifact", artifact_path)
            artifact_ids.append(artifact_id)
            artifacts_by_id.setdefault(
                artifact_id,
                ArtifactRecord(
                    artifact_id=artifact_id,
                    name=Path(artifact_path).name,
                    path=artifact_path,
                    source_uri=f"file://{artifact_path}",
                ),
            )

        vector_id = _stable_id("turn-vector", turn_id)
        model_name = event.gateway_result.resolved_model if turn.role == "assistant" else None
        timestamp = turn.timestamp
        if timestamp is None and turn.role == "assistant":
            timestamp = event.gateway_result.completed_at

        turns.append(
            ChatTurnRecord(
                turn_id=turn_id,
                session_id=event.session_id,
                role=turn.role,
                content=turn.content,
                turn_index=turn_index,
                timestamp=timestamp,
                source_uri=turn_source_uri,
                model_name=model_name,
                vector_id=vector_id,
                mentioned_artifact_ids=tuple(artifact_ids),
            )
        )
        embedding_documents.append(
            ChatEmbeddingDocument(
                vector_id=vector_id,
                turn_id=turn_id,
                content=turn.content,
                payload={
                    "session_id": event.session_id,
                    "turn_id": turn_id,
                    "turn_index": turn_index,
                    "role": turn.role,
                    "source": event.source,
                    "source_uri": turn_source_uri,
                    "resolved_model": model_name,
                    "gateway_request_id": event.gateway_result.request_id,
                    "artifact_ids": artifact_ids,
                },
            )
        )

        directive_kind = classify_directive(turn.content)
        if directive_kind and turn.role in {"user", "system"}:
            directives.append(
                DirectiveRecord(
                    directive_id=_stable_id("directive", f"{turn_id}:{turn.content}"),
                    kind=directive_kind,
                    body=turn.content.strip(),
                    source_turn_id=turn_id,
                    severity="warning" if directive_kind == "constraint" else "info",
                )
            )

    queries = build_chat_memory_queries(
        session,
        turns,
        directives,
        tuple(artifacts_by_id.values()),
    )

    return TranscriptIngestionBatch(
        session=session,
        turns=tuple(turns),
        directives=tuple(directives),
        artifacts=tuple(artifacts_by_id.values()),
        queries=tuple(queries),
        embedding_documents=tuple(embedding_documents),
    )


def build_qdrant_points(
    documents: tuple[ChatEmbeddingDocument, ...],
    vectors: list[list[float]],
) -> list[PointStruct]:
    if len(documents) != len(vectors):
        raise ValueError("documents and vectors must have the same length")
    return [
        PointStruct(
            id=document.vector_id,
            vector=vector,
            payload={
                **document.payload,
                "content": document.content,
            },
        )
        for document, vector in zip(documents, vectors)
    ]
