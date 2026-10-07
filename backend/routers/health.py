"""
backend/routers/health.py — GET /health and GET /api/health

Simple liveness / readiness probe used by the frontend status indicator.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.config import MOCK_MODE

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["Health"])
@router.get("/api/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """
    Liveness probe.  Returns {"status": "ok"} when the server is up.
    The frontend polls this to display the connection indicator.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "mock_mode": MOCK_MODE,
        },
    )
