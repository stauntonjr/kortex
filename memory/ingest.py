"""
Kortex — Memory Ingestion Pipeline
====================================
Maps code chunks and their Qdrant vector references into the TypeDB 3.x
hypergraph, creating or updating ``code-entity`` nodes and ``triplet-link``
relations.

Usage::

    python memory/ingest.py --path ./src --language python

Dependencies (install via pip):
    typedb-driver>=3.0
    qdrant-client>=1.9
    openai>=1.30          # for embedding via gateway
    tree-sitter>=0.22     # optional: for richer AST-level entity extraction
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from typedb.driver import TransactionOptions, TransactionType, TypeDB, credentials_new

# Optional runtime-only imports (DriverOptions/DriverTlsConfig may not exist
# in all installed driver versions). Import when needed inside functions.

logger = logging.getLogger(__name__)


def build_typedb_credentials() -> object | None:
    """Construct TypeDB credentials using the installed TypeDB 3.x API."""
    try:
        return credentials_new(TYPEDB_USERNAME, TYPEDB_PASSWORD)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

TYPEDB_ADDR       = os.getenv("TYPEDB_ADDR",       "localhost:1729")
TYPEDB_DATABASE   = os.getenv("TYPEDB_DATABASE",   "kortex")
TYPEDB_USERNAME   = os.getenv("TYPEDB_USERNAME",   "admin")
TYPEDB_PASSWORD   = os.getenv("TYPEDB_PASSWORD",   "password")
QDRANT_ADDR       = os.getenv("QDRANT_ADDR",       "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kortex_code")
EMBEDDING_URL     = os.getenv("EMBEDDING_URL",      "http://localhost:8080/v1")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",    "embedding")
EMBEDDING_DIM     = int(os.getenv("EMBEDDING_DIM", "3584"))   # Qwen2.5-VL-7B
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",    "512"))    # characters
TYPEDB_REQUEST_TIMEOUT_MILLIS = int(os.getenv("TYPEDB_REQUEST_TIMEOUT_MILLIS", "15000"))
TYPEDB_TRANSACTION_TIMEOUT_MILLIS = int(
    os.getenv("TYPEDB_TRANSACTION_TIMEOUT_MILLIS", "30000")
)
TYPEDB_SCHEMA_LOCK_TIMEOUT_MILLIS = int(
    os.getenv("TYPEDB_SCHEMA_LOCK_TIMEOUT_MILLIS", "15000")
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    entity_id:   str
    kind:        str      # file | function | class | module
    name:        str
    path:        str
    language:    str
    content:     str
    digest:      str = field(init=False)
    vector_id:   str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.digest = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass(frozen=True)
class ChatSessionRecord:
    session_id: str
    title: str | None = None
    source_uri: str | None = None


@dataclass(frozen=True)
class ChatTurnRecord:
    turn_id: str
    session_id: str
    role: str
    content: str
    turn_index: int
    timestamp: datetime | None = None
    source_uri: str | None = None
    model_name: str | None = None
    vector_id: str | None = None
    mentioned_entity_ids: tuple[str, ...] = ()
    mentioned_artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectiveRecord:
    directive_id: str
    kind: str
    body: str
    source_turn_id: str
    severity: str = "info"


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    name: str
    path: str
    content_text: str | None = None
    source_uri: str | None = None
    vector_id: str | None = None


def _quote_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
    )
    return f"\"{escaped}\""


def build_code_entity_insert(chunk: CodeChunk) -> str:
    return f"""
                    insert $e isa code-entity,
                      has entity-id    {_quote_string(chunk.entity_id)},
                      has entity-kind  {_quote_string(chunk.kind)},
                      has entity-name  {_quote_string(chunk.name)},
                      has entity-path  {_quote_string(chunk.path)},
                      has entity-digest {_quote_string(chunk.digest)},
                      has vector-id    {_quote_string(chunk.vector_id)},
                      has language     {_quote_string(chunk.language)};
                    """


def build_chat_session_insert(session: ChatSessionRecord) -> str:
    lines = [
        "insert $session isa chat-session,",
        f"  has session-id {_quote_string(session.session_id)},",
    ]
    if session.title:
        lines.append(f"  has session-name {_quote_string(session.title)},")
    if session.source_uri:
        lines.append(f"  has source-uri {_quote_string(session.source_uri)},")
    lines[-1] = lines[-1].rstrip(",") + ";"
    return "\n".join(lines)


def build_chat_turn_insert(turn: ChatTurnRecord) -> str:
    lines = [
        "insert $turn isa chat-turn,",
        f"  has turn-id {_quote_string(turn.turn_id)},",
        f"  has speaker-role {_quote_string(turn.role)},",
        f"  has turn-index {turn.turn_index},",
        f"  has content-text {_quote_string(turn.content)},",
    ]
    if turn.timestamp:
        lines.append(f"  has turn-timestamp {turn.timestamp.isoformat()},")
    if turn.source_uri:
        lines.append(f"  has source-uri {_quote_string(turn.source_uri)},")
    if turn.vector_id:
        lines.append(f"  has vector-id {_quote_string(turn.vector_id)},")
    if turn.model_name:
        lines.append(f"  has model-name {_quote_string(turn.model_name)},")
    lines[-1] = lines[-1].rstrip(",") + ";"
    return "\n".join(lines)


def build_directive_insert(directive: DirectiveRecord) -> str:
    return "\n".join(
        [
            "insert $directive isa directive-node,",
            f"  has directive-id {_quote_string(directive.directive_id)},",
            f"  has directive-kind {_quote_string(directive.kind)},",
            f"  has directive-body {_quote_string(directive.body)},",
            f"  has severity {_quote_string(directive.severity)};",
        ]
    )


def build_project_artifact_insert(artifact: ArtifactRecord) -> str:
    lines = [
        "insert $artifact isa project-artifact,",
        f"  has artifact-id {_quote_string(artifact.artifact_id)},",
        f"  has artifact-name {_quote_string(artifact.name)},",
        f"  has artifact-path {_quote_string(artifact.path)},",
    ]
    if artifact.content_text:
        lines.append(f"  has content-text {_quote_string(artifact.content_text)},")
    if artifact.source_uri:
        lines.append(f"  has source-uri {_quote_string(artifact.source_uri)},")
    if artifact.vector_id:
        lines.append(f"  has vector-id {_quote_string(artifact.vector_id)},")
    lines[-1] = lines[-1].rstrip(",") + ";"
    return "\n".join(lines)


def build_session_membership_insert(session_id: str, turn_id: str) -> str:
    return "\n".join(
        [
            "match",
            f"  $session isa chat-session, has session-id {_quote_string(session_id)};",
            f"  $turn isa chat-turn, has turn-id {_quote_string(turn_id)};",
            "insert",
            "  (session: $session, turn: $turn) isa session-membership;",
        ]
    )


def build_turn_mention_insert(turn_id: str, entity_id: str) -> str:
    return "\n".join(
        [
            "match",
            f"  $turn isa chat-turn, has turn-id {_quote_string(turn_id)};",
            f"  $entity isa code-entity, has entity-id {_quote_string(entity_id)};",
            "insert",
            "  (source-turn: $turn, target: $entity) isa mention, has confidence 1.0;",
        ]
    )


def build_turn_artifact_mention_insert(turn_id: str, artifact_id: str) -> str:
    return "\n".join(
        [
            "match",
            f"  $turn isa chat-turn, has turn-id {_quote_string(turn_id)};",
            f"  $artifact isa project-artifact, has artifact-id {_quote_string(artifact_id)};",
            "insert",
            "  (source-turn: $turn, target: $artifact) isa mention, has confidence 1.0;",
        ]
    )


def build_directive_source_insert(directive_id: str, source_turn_id: str) -> str:
    return "\n".join(
        [
            "match",
            f"  $directive isa directive-node, has directive-id {_quote_string(directive_id)};",
            f"  $turn isa chat-turn, has turn-id {_quote_string(source_turn_id)};",
            "insert",
            "  (directive: $directive, source-turn: $turn) isa directive-source, has confidence 1.0;",
        ]
    )


def build_chat_memory_queries(
    session: ChatSessionRecord,
    turns: Sequence[ChatTurnRecord],
    directives: Sequence[DirectiveRecord] = (),
    artifacts: Sequence[ArtifactRecord] = (),
) -> list[str]:
    turn_ids = {turn.turn_id for turn in turns}
    queries = [build_chat_session_insert(session)]
    seen_artifact_ids: set[str] = set()

    for artifact in artifacts:
        if artifact.artifact_id in seen_artifact_ids:
            continue
        queries.append(build_project_artifact_insert(artifact))
        seen_artifact_ids.add(artifact.artifact_id)

    for turn in turns:
        if turn.session_id != session.session_id:
            raise ValueError(
                f"turn {turn.turn_id} belongs to session {turn.session_id}, expected {session.session_id}"
            )
        queries.append(build_chat_turn_insert(turn))
        queries.append(build_session_membership_insert(session.session_id, turn.turn_id))
        for entity_id in turn.mentioned_entity_ids:
            queries.append(build_turn_mention_insert(turn.turn_id, entity_id))
        for artifact_id in turn.mentioned_artifact_ids:
            if artifact_id not in seen_artifact_ids:
                raise ValueError(
                    f"turn {turn.turn_id} references unknown artifact {artifact_id}"
                )
            queries.append(build_turn_artifact_mention_insert(turn.turn_id, artifact_id))

    for directive in directives:
        if directive.source_turn_id not in turn_ids:
            raise ValueError(
                f"directive {directive.directive_id} references unknown turn {directive.source_turn_id}"
            )
        queries.append(build_directive_insert(directive))
        queries.append(build_directive_source_insert(directive.directive_id, directive.source_turn_id))

    return queries


# ---------------------------------------------------------------------------
# Source extraction (simple line-based chunker; swap for tree-sitter later)
# ---------------------------------------------------------------------------

def extract_chunks(source_path: Path, language: str) -> list[CodeChunk]:
    """Split a source file into overlapping fixed-size character chunks."""
    text = source_path.read_text(errors="replace")
    chunks: list[CodeChunk] = []
    step = CHUNK_SIZE // 2  # 50 % overlap
    for i, start in enumerate(range(0, len(text), step)):
        content = text[start : start + CHUNK_SIZE]
        if not content.strip():
            continue
        chunks.append(
            CodeChunk(
                entity_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_path}#{i}")),
                kind="file",
                name=f"{source_path.name}#{i}",
                path=str(source_path),
                language=language,
                content=content,
            )
        )
    return chunks


def discover_sources(root: Path, language: str) -> list[Path]:
    """Return all source files under *root* matching the given language."""
    extensions: dict[str, list[str]] = {
        "python":     [".py"],
        "typescript": [".ts", ".tsx"],
        "javascript": [".js", ".jsx"],
        "go":         [".go"],
        "rust":       [".rs"],
        "cpp":        [".cpp", ".cc", ".cxx", ".h", ".hpp"],
        "java":       [".java"],
    }
    exts = set(extensions.get(language, [f".{language}"]))
    return [p for p in root.rglob("*") if p.suffix in exts and p.is_file()]


# ---------------------------------------------------------------------------
# Embedding — call the Kortex gateway (OpenAI-compatible)
# ---------------------------------------------------------------------------

async def embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]:
    """Obtain embeddings for all chunks via the Kortex embedding endpoint."""
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "model": EMBEDDING_MODEL,
            "input": [c.content for c in chunks],
        }
        resp = await client.post(f"{EMBEDDING_URL}/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------

async def upsert_to_qdrant(
    client: AsyncQdrantClient,
    chunks: list[CodeChunk],
    vectors: list[list[float]],
) -> None:
    """Ensure the collection exists, then upsert all chunk vectors."""
    collections = [c.name for c in (await client.get_collections()).collections]
    if QDRANT_COLLECTION not in collections:
        await client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", QDRANT_COLLECTION)

    points = [
        PointStruct(
            id=chunk.vector_id,
            vector=vec,
            payload={
                "entity_id": chunk.entity_id,
                "kind":      chunk.kind,
                "name":      chunk.name,
                "path":      chunk.path,
                "language":  chunk.language,
                "digest":    chunk.digest,
                "content":   chunk.content,
            },
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    await client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("Upserted %d points to Qdrant.", len(points))


# ---------------------------------------------------------------------------
# TypeDB upsert
# ---------------------------------------------------------------------------


def build_typedb_driver_options():
    """Build driver options in a runtime-compatible way.

    Returns either a `TypeDBOptions`/`DriverOptions` object or a native
    options handle created via `options_new()` depending on installed driver.
    """
    try:
        from typedb.driver import DriverOptions, DriverTlsConfig
        try:
            return DriverOptions(DriverTlsConfig.disabled(), request_timeout_millis=TYPEDB_REQUEST_TIMEOUT_MILLIS)
        except Exception:
            return DriverOptions()
    except Exception:
        try:
            from typedb.driver import TypeDBOptions, DriverTlsConfig
            try:
                return TypeDBOptions(DriverTlsConfig.disabled(), request_timeout_millis=TYPEDB_REQUEST_TIMEOUT_MILLIS)
            except Exception:
                return TypeDBOptions()
        except Exception:
            return None


def build_typedb_transaction_options(transaction_type: TransactionType) -> TransactionOptions:
    kwargs = {"transaction_timeout_millis": TYPEDB_TRANSACTION_TIMEOUT_MILLIS}
    if transaction_type == TransactionType.SCHEMA:
        kwargs["schema_lock_acquire_timeout_millis"] = TYPEDB_SCHEMA_LOCK_TIMEOUT_MILLIS
    return TransactionOptions(**kwargs)


@contextmanager
def typedb_transaction(driver, database: str, transaction_type: TransactionType):
    with driver.transaction(
        database,
        transaction_type,
        options=build_typedb_transaction_options(transaction_type),
    ) as tx:
        yield tx

def upsert_to_typedb(
    driver,
    chunks: list[CodeChunk],
    database: str,
) -> None:
    """Insert or update code-entity nodes in the TypeDB hypergraph.

    TypeDB 3.x no longer uses sessions — transactions are opened directly
    from the driver against a named database.
    """
    for chunk in chunks:
        with typedb_transaction(driver, database, TransactionType.READ) as tx:
            existing = list(
                tx.query(f'match $e isa code-entity, has entity-id "{chunk.entity_id}";')
            )

        if existing:
            with typedb_transaction(driver, database, TransactionType.WRITE) as tx:
                tx.query(
                    f"""
                    match
                      $e isa code-entity, has entity-id {_quote_string(chunk.entity_id)},
                           has entity-digest $d, has vector-id $v;
                    delete has entity-digest $d of $e;
                    delete has vector-id $v of $e;
                    insert
                      has entity-digest {_quote_string(chunk.digest)} of $e;
                      has vector-id     {_quote_string(chunk.vector_id)} of $e;
                    """
                )
                tx.commit()
        else:
            with typedb_transaction(driver, database, TransactionType.WRITE) as tx:
                tx.query(build_code_entity_insert(chunk))
                tx.commit()
    logger.info("Upserted %d entities into TypeDB.", len(chunks))


def upsert_chat_session_to_typedb(
    driver,
    session: ChatSessionRecord,
    turns: Sequence[ChatTurnRecord],
    directives: Sequence[DirectiveRecord],
    artifacts: Sequence[ArtifactRecord],
    database: str,
) -> None:
    queries = build_chat_memory_queries(session, turns, directives, artifacts)
    with typedb_transaction(driver, database, TransactionType.WRITE) as tx:
        for query in queries:
            tx.query(query)
        tx.commit()


# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def bootstrap_typedb(driver, database: str, schema_path: Path) -> None:
    """Create database and apply schema if it does not already exist.

    TypeDB 3.x opens schema transactions directly from the driver
    (no separate session concept).
    """
    existing = [db.name for db in driver.databases.all()]
    if database not in existing:
        driver.databases.create(database)
        logger.info("Created TypeDB database '%s'.", database)
        schema_tql = schema_path.read_text()
        with typedb_transaction(driver, database, TransactionType.SCHEMA) as tx:
            tx.query(schema_tql)
            tx.commit()
        logger.info("Applied schema from '%s'.", schema_path)


# ---------------------------------------------------------------------------
# Main ingestion entry-point
# ---------------------------------------------------------------------------

async def ingest(root: Path, language: str) -> None:
    schema_path = Path(__file__).parent / "schema.tql"
    sources = discover_sources(root, language)
    if not sources:
        logger.warning("No %s source files found under '%s'.", language, root)
        return

    logger.info("Discovered %d source files.", len(sources))

    all_chunks: list[CodeChunk] = []
    for src in sources:
        all_chunks.extend(extract_chunks(src, language))
    logger.info("Extracted %d chunks.", len(all_chunks))

    # Embed in batches of 64 to avoid request-size limits
    batch_size = 64
    all_vectors: list[list[float]] = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vecs = await embed_chunks(batch)
        all_vectors.extend(vecs)
        logger.info("Embedded batch %d/%d.", i // batch_size + 1, -(-len(all_chunks) // batch_size))

    # Qdrant
    qdrant = AsyncQdrantClient(url=QDRANT_ADDR)
    await upsert_to_qdrant(qdrant, all_chunks, all_vectors)

    # TypeDB
    creds = build_typedb_credentials()
    if creds is not None:
        with TypeDB.driver(TYPEDB_ADDR, creds, build_typedb_driver_options()) as driver:
            bootstrap_typedb(driver, TYPEDB_DATABASE, schema_path)
            upsert_to_typedb(driver, all_chunks, TYPEDB_DATABASE)
    else:
        with TypeDB.driver(TYPEDB_ADDR, build_typedb_driver_options()) as driver:
            bootstrap_typedb(driver, TYPEDB_DATABASE, schema_path)
            upsert_to_typedb(driver, all_chunks, TYPEDB_DATABASE)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="Kortex memory ingestion pipeline")
    parser.add_argument("--path",     required=True, help="Root directory to ingest")
    parser.add_argument("--language", default="python", help="Source language")
    args = parser.parse_args(argv)

    asyncio.run(ingest(Path(args.path), args.language))


if __name__ == "__main__":
    main()
