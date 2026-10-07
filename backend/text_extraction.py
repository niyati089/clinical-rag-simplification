"""
backend/text_extraction.py — Document text extraction layer.

Supported formats:
  - PDF  (.pdf)         → pdfplumber
  - DOCX (.docx)        → python-docx
  - TXT  (.txt)         → plain utf-8 / latin-1 read
  - Image (.png/.jpg)   → EasyOCR (lazy-loaded on first use)

All functions return a plain str. Raises ValueError on unsupported types,
IOError-family exceptions on read failures.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── EasyOCR lazy singleton ────────────────────────────────────────────────────
# Importing easyocr at module level would download models on every cold start.
# We lazy-load it only when an image upload arrives.

_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr  # type: ignore
            logger.info("Initialising EasyOCR reader (English)…")
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader ready.")
        except ImportError:
            raise RuntimeError(
                "EasyOCR is not installed. "
                "Install it with: pip install easyocr"
            )
    return _ocr_reader


# ── Text cleaning helpers ─────────────────────────────────────────────────────

def _clean_extracted_text(text: str) -> str:
    """
    Post-extraction cleaning applied to all document types:
      1. Normalise line endings.
      2. Fix broken hyphenated line-wraps (e.g. "diag-\nnosis" → "diagnosis").
      3. Collapse 3+ blank lines to 2.
      4. Strip trailing spaces per line.
      5. Remove obvious header/footer patterns (page numbers, repeated short lines).
    """
    # 1. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Fix hyphenated line-wraps
    text = re.sub(r"-\n([a-z])", r"\1", text)

    # 3. Merge lines that are broken mid-sentence (single newline between two
    #    lowercase-starting or punctuation-continuing lines) — but keep
    #    paragraph breaks (double newlines) intact.
    text = re.sub(r"(?<!\n)\n(?!\n)(?=[a-z,;])", " ", text)

    # 4. Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]

    # 6. Remove header/footer noise: lines that are purely numeric (page numbers)
    #    or very short repeated lines (≤ 4 chars, e.g. "---")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d+", stripped):       # pure page number
            continue
        if len(stripped) <= 3 and not stripped.isalpha():  # noise like "---"
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise RuntimeError(
            "pdfplumber is not installed. Install with: pip install pdfplumber"
        )

    pages_text: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

    raw = "\n\n".join(pages_text)
    if not raw.strip():
        raise ValueError(
            "PDF appears to be image-based or empty. "
            "No selectable text was found. Try uploading as an image instead."
        )
    return _clean_extracted_text(raw)


# ── DOCX extraction ───────────────────────────────────────────────────────────

def extract_from_docx(data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        import docx  # type: ignore  (python-docx)
    except ImportError:
        raise RuntimeError(
            "python-docx is not installed. Install with: pip install python-docx"
        )

    doc = docx.Document(io.BytesIO(data))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    raw = "\n\n".join(paragraphs)
    return _clean_extracted_text(raw)


# ── TXT extraction ────────────────────────────────────────────────────────────

def extract_from_txt(data: bytes) -> str:
    """Decode plain-text bytes (UTF-8 with Latin-1 fallback)."""
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        raw = data.decode("latin-1")
    return _clean_extracted_text(raw)


# ── Image extraction ──────────────────────────────────────────────────────────

def extract_from_image(data: bytes) -> str:
    """
    Extract text from image bytes (PNG/JPG) using EasyOCR.
    EasyOCR is lazy-loaded on first call.
    """
    reader = _get_ocr_reader()
    import numpy as np  # noqa: F401 — numpy is a hard dep
    from PIL import Image  # type: ignore

    image = Image.open(io.BytesIO(data)).convert("RGB")
    import numpy as np
    image_np = np.array(image)

    results = reader.readtext(image_np, detail=0, paragraph=True)
    raw = "\n".join(results)
    if not raw.strip():
        raise ValueError("No text could be extracted from the image.")
    return _clean_extracted_text(raw)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract_text(
    data: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> str:
    """
    Dispatch to the correct extractor based on file extension.
    Falls back to content_type if extension is ambiguous.

    Parameters
    ----------
    data : bytes
        Raw file bytes.
    filename : str
        Original filename (used to determine extension).
    content_type : str, optional
        MIME type as sent by the client.

    Returns
    -------
    str
        Cleaned extracted text.

    Raises
    ------
    ValueError
        If the file type is unsupported or extraction yields no text.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf" or content_type == "application/pdf":
        logger.info("Extracting text from PDF: %s", filename)
        return extract_from_pdf(data)

    if ext == ".docx" or content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        logger.info("Extracting text from DOCX: %s", filename)
        return extract_from_docx(data)

    if ext == ".txt" or (content_type and content_type.startswith("text/")):
        logger.info("Extracting text from TXT: %s", filename)
        return extract_from_txt(data)

    if ext in (".png", ".jpg", ".jpeg") or (
        content_type and content_type.startswith("image/")
    ):
        logger.info("Extracting text from image via EasyOCR: %s", filename)
        return extract_from_image(data)

    raise ValueError(
        f"Unsupported file type '{ext}' (content-type: {content_type}). "
        "Accepted: .pdf, .docx, .txt, .png, .jpg, .jpeg"
    )
