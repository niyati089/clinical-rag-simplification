"""
chunking.py — Semantic chunking for clinical/medical text.

Splits raw clinical text into meaningful semantic chunks using:
  1. Section/heading detection
  2. Paragraph boundaries
  3. Sentence-level overflow splitting (keeps chunks within token budget)

Does NOT split blindly on character count.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import List

import nltk

from src.config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_SENTENCES

# Download required NLTK data (idempotent)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class ClinicalChunk:
    chunk_id: str
    text: str
    section_title: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── Heading patterns ────────────────────────────────────────────────────────

_HEADING_PATTERNS = [
    # ALLCAPS heading, e.g. "DIAGNOSIS:"
    re.compile(r"^([A-Z][A-Z\s/\-]{3,}):?\s*$", re.MULTILINE),
    # Numbered section, e.g. "1. History" or "1) Assessment"
    re.compile(r"^\d+[\.\)]\s+[A-Z][a-zA-Z\s]+$", re.MULTILINE),
    # Title-case heading followed by colon, e.g. "Chief Complaint:"
    re.compile(r"^([A-Z][a-z]+(?: [A-Za-z]+){0,5}):$", re.MULTILINE),
]


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


# ── Token estimation (fast, no tokenizer dependency) ────────────────────────

def _approx_tokens(text: str) -> int:
    """Approximate GPT-style token count: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── Core splitting logic ─────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def _sentences_to_chunks(
    sentences: List[str],
    section_title: str,
    base_index: int,
    max_tokens: int,
    overlap: int,
) -> List[ClinicalChunk]:
    """Pack sentences into chunks that stay within the token budget."""
    chunks: List[ClinicalChunk] = []
    current: List[str] = []
    current_tokens = 0
    chunk_index = base_index

    for sentence in sentences:
        sentence_tokens = _approx_tokens(sentence)

        # If adding this sentence would exceed budget and we have existing content:
        if current and (current_tokens + sentence_tokens > max_tokens):
            chunk_text = " ".join(current)
            chunks.append(
                ClinicalChunk(
                    chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                    text=chunk_text,
                    section_title=section_title,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            # Retain overlap sentences
            overlap_sentences = current[-overlap:] if overlap else []
            # Safety check: if overlap sentences alone exceed budget, drop overlap
            if sum(_approx_tokens(s) for s in overlap_sentences) >= max_tokens:
                overlap_sentences = []

            current = list(overlap_sentences)
            current_tokens = sum(_approx_tokens(s) for s in current)

            # If adding the new sentence still exceeds (e.g. huge sentence), flush overlap too
            if current and (current_tokens + sentence_tokens > max_tokens):
                current = []
                current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(
            ClinicalChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                text=" ".join(current),
                section_title=section_title,
                chunk_index=chunk_index,
            )
        )

    return chunks


# ── Public API ──────────────────────────────────────────────────────────────

def chunk_clinical_text(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> List[ClinicalChunk]:
    """
    Segment raw clinical text into meaningful semantic chunks.

    Strategy:
      1. Split on detected section headings.
      2. Within each section, split on paragraph boundaries.
      3. If a paragraph is still too long, split on sentence boundaries
         while maintaining a configurable sentence overlap.

    Parameters
    ----------
    text : str
        Raw clinical/medical text.
    max_tokens : int
        Approximate token budget per chunk.
    overlap_sentences : int
        Number of trailing sentences to carry into the next chunk
        for context continuity.

    Returns
    -------
    List[ClinicalChunk]
    """
    if not text or not text.strip():
        raise ValueError("Input clinical text is empty or whitespace-only.")

    text = _clean_text(text)

    # ── Step 1: split on headings ─────────────────────────────────────────
    sections = _split_on_headings(text)

    chunks: List[ClinicalChunk] = []
    global_index = 0

    for title, body in sections:
        if not body.strip():
            continue

        # ── Step 2: split section body on paragraphs ──────────────────────
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]

        for para in paragraphs:
            if _approx_tokens(para) <= max_tokens:
                chunks.append(
                    ClinicalChunk(
                        chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                        text=para,
                        section_title=title,
                        chunk_index=global_index,
                    )
                )
                global_index += 1
            else:
                # ── Step 3: sentence-level splitting ──────────────────────
                sentences = _split_into_sentences(para)
                new_chunks = _sentences_to_chunks(
                    sentences, title, global_index, max_tokens, overlap_sentences
                )
                chunks.extend(new_chunks)
                global_index += len(new_chunks)

    if not chunks:
        # Fallback: treat entire text as one chunk
        chunks.append(
            ClinicalChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                text=text,
                section_title="",
                chunk_index=0,
            )
        )

    return chunks


# ── Helpers ─────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove excessive whitespace and formatting noise."""
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ newlines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _split_on_headings(text: str) -> List[tuple[str, str]]:
    """
    Return list of (heading, body) pairs.
    If no headings are found, returns [("", full_text)].
    """
    lines = text.split("\n")
    sections: List[tuple[str, str]] = []
    current_title = ""
    current_body_lines: List[str] = []

    for line in lines:
        if _is_heading(line):
            if current_body_lines or current_title:
                sections.append((current_title, "\n".join(current_body_lines)))
            current_title = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # flush last section
    sections.append((current_title, "\n".join(current_body_lines)))

    if not sections:
        return [("", text)]

    return sections
