"""
backend/jobs.py — In-memory async job store.

Jobs are created by POST /api/simplify and polled via GET /api/jobs/{job_id}.
No persistence: jobs vanish on server restart (acceptable for a student project).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStage(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    PARSING = "parsing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    SIMPLIFYING = "simplifying"
    REFINING = "refining"
    VERIFYING = "verifying"
    DONE = "done"


# ── Job model ─────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }


# ── Job store ─────────────────────────────────────────────────────────────────

class JobStore:
    """
    Thread-safe in-memory job registry.
    Uses asyncio.Lock so it can be awaited inside FastAPI async endpoints.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        """Create a new job and return it."""
        job = Job(id=str(uuid.uuid4()))
        async with self._lock:
            self._jobs[job.id] = job
        logger.debug("Job %s created.", job.id)
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        """Return the job or None if not found."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_stage(self, job_id: str, stage: JobStage) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.stage = stage
                job.status = JobStatus.RUNNING
                job.updated_at = time.time()
                logger.debug("Job %s stage → %s", job_id, stage.value)

    async def mark_done(self, job_id: str, result: Dict[str, Any]) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.DONE
                job.stage = JobStage.DONE
                job.result = result
                job.updated_at = time.time()
                logger.info("Job %s done.", job_id)

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                job.updated_at = time.time()
                logger.error("Job %s failed: %s", job_id, error)

    async def prune(self, ttl_seconds: float = 3600.0) -> int:
        """Remove completed/failed jobs older than ttl_seconds. Returns count pruned."""
        now = time.time()
        pruned = 0
        async with self._lock:
            to_delete = [
                jid
                for jid, j in self._jobs.items()
                if j.status in (JobStatus.DONE, JobStatus.FAILED)
                and (now - j.updated_at) > ttl_seconds
            ]
            for jid in to_delete:
                del self._jobs[jid]
                pruned += 1
        if pruned:
            logger.debug("Pruned %d stale jobs.", pruned)
        return pruned


# ── Module singleton (shared across the app via FastAPI app state) ────────────

job_store = JobStore()
