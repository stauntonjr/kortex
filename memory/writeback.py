from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from kortex.contracts import TranscriptWritebackEvent, transcript_writeback_event_to_dict
from memory.chat_ingest import persist_writeback_event

logger = logging.getLogger(__name__)

WRITEBACK_ENABLED = os.getenv("KORTEX_WRITEBACK_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}
WRITEBACK_PATH = Path(os.getenv("KORTEX_WRITEBACK_PATH", "/tmp/kortex-writeback.jsonl"))
WRITEBACK_PERSIST_ENABLED = os.getenv("KORTEX_WRITEBACK_PERSIST_ENABLED", "0").lower() in {
    "1",
    "true",
    "yes",
}

_writeback_queue: asyncio.Queue[TranscriptWritebackEvent | None] | None = None
_writeback_task: asyncio.Task[None] | None = None


def _append_writeback_event(path: Path, event: TranscriptWritebackEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(transcript_writeback_event_to_dict(event), handle, separators=(",", ":"))
        handle.write("\n")


async def _writeback_worker(path: Path) -> None:
    assert _writeback_queue is not None
    while True:
        event = await _writeback_queue.get()
        if event is None:
            _writeback_queue.task_done()
            break
        try:
            _append_writeback_event(path, event)
            if WRITEBACK_PERSIST_ENABLED:
                await persist_writeback_event(event)
        except Exception:
            logger.exception("Failed to persist transcript writeback event.")
        finally:
            _writeback_queue.task_done()


async def start_writeback_worker(path: Path | None = None) -> None:
    global _writeback_queue, _writeback_task

    if not WRITEBACK_ENABLED or _writeback_task is not None:
        return

    _writeback_queue = asyncio.Queue()
    _writeback_task = asyncio.create_task(_writeback_worker(path or WRITEBACK_PATH))


async def stop_writeback_worker() -> None:
    global _writeback_queue, _writeback_task

    if _writeback_queue is None or _writeback_task is None:
        return

    await _writeback_queue.put(None)
    await _writeback_task
    _writeback_queue = None
    _writeback_task = None


async def enqueue_writeback_event(event: TranscriptWritebackEvent) -> None:
    if not WRITEBACK_ENABLED:
        return
    if _writeback_queue is None:
        await start_writeback_worker()
    assert _writeback_queue is not None
    await _writeback_queue.put(event)
