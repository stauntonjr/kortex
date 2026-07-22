"""
Kortex Memory Ingestion — Unit Tests
======================================
Tests the chunking, discovery, and entity-construction logic in
``memory/ingest.py`` without requiring live TypeDB, Qdrant, or the gateway.

Run with::

    pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.ingest import (
    CHUNK_SIZE,
    CodeChunk,
    discover_sources,
    extract_chunks,
)


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------


class TestCodeChunk:
    def test_digest_is_sha256_of_content(self):
        chunk = CodeChunk(
            entity_id=str(uuid.uuid4()),
            kind="file",
            name="test.py#0",
            path="/src/test.py",
            language="python",
            content="print('hello')",
        )
        expected = hashlib.sha256(b"print('hello')").hexdigest()
        assert chunk.digest == expected

    def test_vector_id_is_auto_generated_uuid(self):
        chunk = CodeChunk(
            entity_id=str(uuid.uuid4()),
            kind="file",
            name="test.py#0",
            path="/src/test.py",
            language="python",
            content="x = 1",
        )
        # Should be a valid UUID
        uuid.UUID(chunk.vector_id)

    def test_two_chunks_with_same_content_have_same_digest(self):
        content = "# identical content"
        c1 = CodeChunk(str(uuid.uuid4()), "file", "a#0", "/a.py", "python", content)
        c2 = CodeChunk(str(uuid.uuid4()), "file", "b#0", "/b.py", "python", content)
        assert c1.digest == c2.digest

    def test_different_content_gives_different_digest(self):
        c1 = CodeChunk(str(uuid.uuid4()), "file", "a#0", "/a.py", "python", "x = 1")
        c2 = CodeChunk(str(uuid.uuid4()), "file", "a#0", "/a.py", "python", "x = 2")
        assert c1.digest != c2.digest


# ---------------------------------------------------------------------------
# extract_chunks
# ---------------------------------------------------------------------------


class TestExtractChunks:
    def test_empty_file_yields_no_chunks(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        chunks = extract_chunks(f, "python")
        assert chunks == []

    def test_whitespace_only_file_yields_no_chunks(self, tmp_path):
        f = tmp_path / "ws.py"
        f.write_text("   \n\t\n   ")
        chunks = extract_chunks(f, "python")
        assert chunks == []

    def test_small_file_yields_at_least_one_chunk(self, tmp_path):
        f = tmp_path / "small.py"
        f.write_text("x = 1\n")
        chunks = extract_chunks(f, "python")
        assert len(chunks) >= 1

    def test_chunk_content_covers_file(self, tmp_path):
        content = "a" * 10
        f = tmp_path / "tiny.py"
        f.write_text(content)
        chunks = extract_chunks(f, "python")
        # Reconstruct by taking unique non-overlapping content
        combined = "".join(c.content for c in chunks)
        assert content in combined

    def test_large_file_yields_multiple_chunks(self, tmp_path):
        content = "x = 1\n" * 200  # well over CHUNK_SIZE
        f = tmp_path / "large.py"
        f.write_text(content)
        chunks = extract_chunks(f, "python")
        assert len(chunks) > 1

    def test_chunk_language_matches_argument(self, tmp_path):
        f = tmp_path / "mod.go"
        f.write_text("package main\n")
        chunks = extract_chunks(f, "go")
        assert all(c.language == "go" for c in chunks)

    def test_chunk_path_matches_file(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("pass\n")
        chunks = extract_chunks(f, "python")
        assert all(c.path == str(f) for c in chunks)

    def test_chunk_entity_ids_are_unique(self, tmp_path):
        content = "z = 9\n" * 300
        f = tmp_path / "dup.py"
        f.write_text(content)
        chunks = extract_chunks(f, "python")
        ids = [c.entity_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_size_does_not_exceed_limit(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("x" * 5000)
        chunks = extract_chunks(f, "python")
        assert all(len(c.content) <= CHUNK_SIZE for c in chunks)


# ---------------------------------------------------------------------------
# discover_sources
# ---------------------------------------------------------------------------


class TestDiscoverSources:
    def test_finds_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        (tmp_path / "c.go").write_text("package main")
        found = discover_sources(tmp_path, "python")
        names = {p.name for p in found}
        assert "a.py" in names
        assert "b.py" in names
        assert "c.go" not in names

    def test_finds_go_files(self, tmp_path):
        (tmp_path / "main.go").write_text("package main")
        (tmp_path / "main.py").write_text("pass")
        found = discover_sources(tmp_path, "go")
        names = {p.name for p in found}
        assert "main.go" in names
        assert "main.py" not in names

    def test_recurses_into_subdirectories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.py").write_text("pass")
        found = discover_sources(tmp_path, "python")
        assert any(p.name == "inner.py" for p in found)

    def test_unknown_language_uses_dot_extension(self, tmp_path):
        (tmp_path / "script.lua").write_text("print('hi')")
        found = discover_sources(tmp_path, "lua")
        assert any(p.name == "script.lua" for p in found)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert discover_sources(tmp_path, "python") == []
