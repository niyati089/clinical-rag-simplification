"""
backend/routers/simplify.py — POST /api/simplify and GET /api/jobs/{job_id}

The simplify endpoint accepts text (or a doc_id from /api/documents) plus a
reading_level. It immediately returns a job_id (202 Accepted) and runs the
A → B → verification chain in a background thread-pool executor.

The jobs endpoint polls for status and returns the final result when done.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from backend.jobs import JobStatus, job_store

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request schema ────────────────────────────────────────────────────────────

class SimplifyRequest(BaseModel):
    """
    At least one of `text` or `doc_id` must be provided.
    If both are provided, `text` takes precedence.
    """
    text: Optional[str] = Field(None, description="Raw clinical text to simplify")
    doc_id: Optional[str] = Field(
        None, description="Document ID returned by POST /api/documents"
    )
    reading_level: Literal["basic", "intermediate", "advanced"] = Field(
        "basic", description="Target reading level"
    )

    @model_validator(mode="after")
    def check_source(self) -> "SimplifyRequest":
        if not self.text and not self.doc_id:
            raise ValueError("Provide either 'text' or 'doc_id'.")
        return self


# ── POST /api/simplify ────────────────────────────────────────────────────────

@router.post("/api/simplify", tags=["Simplify"])
async def simplify_document(
    body: SimplifyRequest,
    request: Request,
) -> JSONResponse:
    """
    Start a background simplification job.

    Returns 202 Accepted immediately with a `job_id`.
    Poll GET /api/jobs/{job_id} for status and results.
    """
    # ── Resolve input text ─────────────────────────────────────────────────
    if body.text:
        clinical_text = body.text.strip()
    else:
        doc = request.app.state.doc_store.get(body.doc_id)
        if doc is None:
            raise HTTPException(
                status_code=404,
                detail=f"doc_id '{body.doc_id}' not found. Upload the document first.",
            )
        clinical_text = doc["text"]

    if not clinical_text:
        raise HTTPException(
            status_code=422,
            detail="Clinical text is empty after resolution.",
        )

    # ── Create job ─────────────────────────────────────────────────────────
    job = await job_store.create()
    logger.info(
        "Job %s queued: reading_level=%s text_length=%d",
        job.id,
        body.reading_level,
        len(clinical_text),
    )

    # ── Dispatch to thread-pool ────────────────────────────────────────────
    loop = asyncio.get_running_loop()

    def _run():
        from backend.pipeline import run_pipeline
        run_pipeline(
            text=clinical_text,
            reading_level=body.reading_level,
            job_id=job.id,
            store=job_store,
            loop=loop,
        )

    loop.run_in_executor(request.app.state.executor, _run)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": job.status.value,
            "message": "Job queued. Poll GET /api/jobs/{job_id} for status.",
        },
    )


# ── GET /api/jobs/{job_id} ────────────────────────────────────────────────────

@router.get("/api/jobs/{job_id}", tags=["Simplify"])
async def get_job_status(job_id: str) -> JSONResponse:
    """
    Poll the status of a simplification job.

    Statuses: queued | running | done | failed
    Stages:   queued | parsing | retrieving | generating | refining | verifying | done
    """
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )

    response_body = job.to_dict()

    # Only include result/error field relevant to current status
    if job.status != JobStatus.DONE:
        response_body.pop("result", None)
    if job.status != JobStatus.FAILED:
        response_body.pop("error", None)

    status_code = 200
    return JSONResponse(status_code=status_code, content=response_body)
