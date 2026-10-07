"""
backend/routers/documents.py — POST /api/documents

Accepts PDF, DOCX, TXT, PNG, JPG. Extracts text, cleans it, and returns
a document_id the client can pass to /api/simplify.

The in-memory doc store is a simple dict on app.state (set in main.py lifespan).
Documents expire on server restart — acceptable for a student project.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
)
from backend.text_extraction import extract_text

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, data: bytes) -> None:
    """Raise HTTPException on invalid type or over-limit size."""
    from pathlib import Path

    # Size check
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{file.filename}' is too large "
                f"({len(data) / 1_048_576:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            ),
        )

    # Extension check
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # MIME type check (client-provided, used as secondary signal)
    ct = (file.content_type or "").split(";")[0].strip()
    if ct and ct not in ALLOWED_MIME_TYPES:
        logger.warning(
            "Unexpected content-type '%s' for file '%s' — proceeding by extension.",
            ct,
            file.filename,
        )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/api/documents", tags=["Documents"])
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT, PNG, or JPG document")],
) -> JSONResponse:
    """
    Upload a clinical document for text extraction.

    Returns a `doc_id` and the extracted text so the user can review and
    optionally edit the text before sending it to /api/simplify.
    """
    # Read bytes
    data = await file.read()

    # Validate
    _validate_upload(file, data)

    # Extract text
    try:
        extracted_text = extract_text(
            data=data,
            filename=file.filename or "upload",
            content_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the uploaded file.",
        )

    # Store in app-level doc registry
    doc_id = str(uuid.uuid4())
    request.app.state.doc_store[doc_id] = {
        "filename": file.filename,
        "text": extracted_text,
    }

    word_count = len(extracted_text.split())
    char_count = len(extracted_text)

    logger.info(
        "Document uploaded: id=%s filename=%s words=%d chars=%d",
        doc_id,
        file.filename,
        word_count,
        char_count,
    )

    return JSONResponse(
        status_code=200,
        content={
            "doc_id": doc_id,
            "filename": file.filename,
            "extracted_text": extracted_text,
            "char_count": char_count,
            "word_count": word_count,
        },
    )
