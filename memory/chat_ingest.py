from __future__ import annotations

import json
import logging
import os
import re
import uuid
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from typedb.driver import TypeDB

from kortex.contracts import GatewayResult, TranscriptTurn, TranscriptWritebackEvent
from memory.ingest import (
    ArtifactRecord,
    ChatSessionRecord,
    ChatTurnRecord,
    DirectiveRecord,
    build_chat_memory_queries,
    build_typedb_credentials,
    build_typedb_driver_options,
    bootstrap_typedb,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    QDRANT_ADDR,
    TYPEDB_ADDR,
    TYPEDB_DATABASE,
    upsert_chat_session_to_typedb,
)

logger = logging.getLogger(__name__)

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


EmbeddingFunction = Callable[[tuple[ChatEmbeddingDocument, ...]], Awaitable[list[list[float]]]]
QdrantUpsertFunction = Callable[[list[PointStruct]], Awaitable[None]]

PERSIST_TO_TYPEDB = os.getenv("KORTEX_CHAT_PERSIST_TYPEDB", "1").lower() not in {
    "0",
    "false",
    "no",
}
PERSIST_TO_QDRANT = os.getenv("KORTEX_CHAT_PERSIST_QDRANT", "1").lower() not in {
    "0",
    "false",
    "no",
}
CHAT_QDRANT_COLLECTION = os.getenv("KORTEX_CHAT_QDRANT_COLLECTION", "kortex_chat")
CHAT_TYPEDB_DATABASE = os.getenv("KORTEX_CHAT_TYPEDB_DATABASE", TYPEDB_DATABASE)
CHAT_EMBEDDING_MODEL = os.getenv("KORTEX_CHAT_EMBEDDING_MODEL", EMBEDDING_MODEL)


def _stable_id(kind: str, value: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, f"{kind}:{value}"))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_writeback_event(path: str | Path) -> TranscriptWritebackEvent:
    payload = json.loads(Path(path).read_text())
    return load_writeback_event_payload(payload)


def load_writeback_event_payload(payload: dict[str, Any]) -> TranscriptWritebackEvent:
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


def iter_writeback_events(path: str | Path) -> Sequence[TranscriptWritebackEvent]:
    events: list[TranscriptWritebackEvent] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(load_writeback_event_payload(json.loads(stripped)))
        except Exception as exc:
            raise ValueError(f"failed to parse writeback event at line {line_number}") from exc
    return tuple(events)


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


def build_default_embedder() -> EmbeddingFunction:
    async def _embedder(documents: tuple[ChatEmbeddingDocument, ...]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "model": CHAT_EMBEDDING_MODEL,
                "input": [document.content for document in documents],
            }
            response = await client.post(f"{EMBEDDING_URL}/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in data]

    return _embedder


def build_default_qdrant_upsert() -> QdrantUpsertFunction:
    client = AsyncQdrantClient(url=QDRANT_ADDR)

    async def _upsert(points: list[PointStruct]) -> None:
        if not points:
            return
        collections = [item.name for item in (await client.get_collections()).collections]
        if CHAT_QDRANT_COLLECTION not in collections:
            await client.create_collection(
                collection_name=CHAT_QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        await client.upsert(collection_name=CHAT_QDRANT_COLLECTION, points=points)

    return _upsert


async def embed_chat_turns(
    documents: tuple[ChatEmbeddingDocument, ...],
    embedder: EmbeddingFunction,
) -> list[list[float]]:
    if not documents:
        return []
    return await embedder(documents)


def persist_batch_to_typedb(driver: Any, batch: TranscriptIngestionBatch, database: str) -> None:
    upsert_chat_session_to_typedb(
        driver,
        batch.session,
        batch.turns,
        batch.directives,
        batch.artifacts,
        database,
    )


async def persist_batch_to_qdrant(
    batch: TranscriptIngestionBatch,
    embedder: EmbeddingFunction,
    upsert_points: QdrantUpsertFunction,
) -> Sequence[PointStruct]:
    vectors = await embed_chat_turns(batch.embedding_documents, embedder)
    points = build_qdrant_points(batch.embedding_documents, vectors)
    if points:
        await upsert_points(points)
    return points


async def persist_writeback_event(
    event: TranscriptWritebackEvent,
    *,
    typedb_driver: Any | None = None,
    typedb_database: str | None = None,
    embedder: EmbeddingFunction | None = None,
    qdrant_upsert: QdrantUpsertFunction | None = None,
) -> TranscriptIngestionBatch:
    batch = build_ingestion_batch(event)
    if typedb_driver is not None and typedb_database:
        persist_batch_to_typedb(typedb_driver, batch, typedb_database)
    elif typedb_driver is None and typedb_database is None and PERSIST_TO_TYPEDB:
        schema_path = Path(__file__).parent / "schema.tql"
        with open_default_typedb_driver() as driver:
            bootstrap_typedb(driver, CHAT_TYPEDB_DATABASE, schema_path)
            persist_batch_to_typedb(driver, batch, CHAT_TYPEDB_DATABASE)
    if embedder is not None and qdrant_upsert is not None:
        await persist_batch_to_qdrant(batch, embedder, qdrant_upsert)
    elif embedder is None and qdrant_upsert is None and PERSIST_TO_QDRANT:
        await persist_batch_to_qdrant(
            batch,
            build_default_embedder(),
            build_default_qdrant_upsert(),
        )
    return batch


async def ingest_writeback_log(path: str | Path) -> int:
    count = 0
    for event in iter_writeback_events(path):
        await persist_writeback_event(event)
        count += 1
    logger.info("Ingested %d transcript writeback events from %s.", count, path)
    return count


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Ingest transcript writeback events into Kortex memory.")
    parser.add_argument("--path", required=True, help="Path to a writeback JSON or JSONL file")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if path.suffix == ".json":
        event = load_writeback_event(path)
        count = 1
        import asyncio

        asyncio.run(persist_writeback_event(event))
    else:
        import asyncio

        count = asyncio.run(ingest_writeback_log(path))
    logger.info("Completed transcript ingest for %d event(s).", count)


if __name__ == "__main__":
    main()
