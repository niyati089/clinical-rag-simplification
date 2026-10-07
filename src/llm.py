"""
llm.py — Modular LLM interface for the clinical RAG pipeline.

Supports three providers controlled by LLM_PROVIDER env var:
  - openai    (default) — requires openai package
  - google    — requires google-generativeai package
  - anthropic — requires anthropic package

API keys and model names are read from environment variables / config.
Never hardcode credentials here.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    VALID_READING_LEVELS,
)
from src.prompts import build_simplification_prompt, build_plain_llm_prompt
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Expected output schema ───────────────────────────────────────────────────

_EMPTY_STRUCTURED_OUTPUT: Dict[str, Any] = {
    "simple_explanation": "",
    "important_instructions": [],
    "medication_guidance": [],
    "follow_up": [],
}


# ── JSON extraction helper ───────────────────────────────────────────────────

def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Parse structured JSON from raw LLM output.
    Handles common issues: markdown fences, leading/trailing prose.
    """
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Find the first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:500]}")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON decoding failed: {exc}\nRaw excerpt: {raw[:500]}"
        ) from exc

    # Validate required keys
    required = {"simple_explanation", "important_instructions",
                "medication_guidance", "follow_up"}
    missing = required - set(data.keys())
    if missing:
        logger.warning("LLM output missing keys: %s — filling with defaults.", missing)
        for key in missing:
            data[key] = [] if key != "simple_explanation" else ""

    return data


# ── Provider adapters ────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI provider requires the 'openai' package. "
            "Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},  # structured output
    )
    return response.choices[0].message.content or ""


def _call_google(prompt: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Google provider requires 'google-generativeai'. "
            "Install with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=LLM_API_KEY)
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        ),
    )
    return response.text or ""


def _call_anthropic(prompt: str) -> str:
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Anthropic provider requires the 'anthropic' package. "
            "Install with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=LLM_API_KEY)
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


def _call_groq(prompt: str) -> str:
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=LLM_API_KEY)
    except ImportError:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(
                api_key=LLM_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
        except ImportError as exc:
            raise ImportError(
                "Groq provider requires 'groq' or 'openai' package. "
                "Install with: pip install groq"
            ) from exc

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


_PROVIDER_MAP = {
    "openai": _call_openai,
    "google": _call_google,
    "anthropic": _call_anthropic,
    "groq": _call_groq,
}


def _call_llm(prompt: str, provider: str = LLM_PROVIDER) -> str:
    """
    Dispatch to the correct LLM provider and return raw text response with retry logic.
    """
    import time
    if not LLM_API_KEY:
        raise EnvironmentError(
            "LLM_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    provider_lower = provider.lower()
    if provider_lower not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid options: {list(_PROVIDER_MAP.keys())}"
        )

    logger.info("Calling LLM provider=%s model=%s", provider_lower, LLM_MODEL)
    
    max_retries = 4
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return _PROVIDER_MAP[provider_lower](prompt)
        except Exception as exc:
            err_str = str(exc)
            if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries - 1:
                logger.warning(
                    "Rate limit (429) encountered. Retrying in %.1fs (attempt %d/%d)...",
                    delay, attempt + 1, max_retries
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ── Public API ───────────────────────────────────────────────────────────────

def generate_response(
    clinical_chunk: str,
    retrieved_context: List[RetrievedChunk],
    reading_level: str = "Basic",
    provider: str = LLM_PROVIDER,
) -> Dict[str, Any]:
    """
    Generate a structured simplified response for a clinical text chunk.

    Parameters
    ----------
    clinical_chunk : str
        The raw clinical text segment to simplify.
    retrieved_context : List[RetrievedChunk]
        Trusted references retrieved from the knowledge base (may be empty).
    reading_level : str
        "Basic" | "Intermediate" | "Advanced"
    provider : str
        LLM provider override (default from config).

    Returns
    -------
    dict with keys:
        simple_explanation : str
        important_instructions : List[str]
        medication_guidance : List[str]
        follow_up : List[str]

    Raises
    ------
    ValueError
        For invalid reading level or malformed LLM output.
    EnvironmentError
        If API key is missing.
    """
    if not clinical_chunk or not clinical_chunk.strip():
        raise ValueError("clinical_chunk is empty.")
    if reading_level not in VALID_READING_LEVELS:
        raise ValueError(
            f"Invalid reading_level '{reading_level}'. "
            f"Choose from: {VALID_READING_LEVELS}"
        )

    prompt = build_simplification_prompt(
        clinical_chunk=clinical_chunk,
        references=retrieved_context,
        reading_level=reading_level,
    )

    raw_response = _call_llm(prompt, provider=provider)
    structured = _extract_json(raw_response)
    return structured


def generate_plain_response(
    clinical_chunk: str,
    reading_level: str = "Basic",
    provider: str = LLM_PROVIDER,
) -> Dict[str, Any]:
    """
    Generate a structured response WITHOUT any retrieved context.
    Used for the Plain LLM baseline in evaluation.

    Parameters
    ----------
    clinical_chunk : str
        Raw clinical text.
    reading_level : str
        "Basic" | "Intermediate" | "Advanced"
    provider : str
        LLM provider override.

    Returns
    -------
    Same dict schema as generate_response().
    """
    if not clinical_chunk or not clinical_chunk.strip():
        raise ValueError("clinical_chunk is empty.")
    if reading_level not in VALID_READING_LEVELS:
        raise ValueError(f"Invalid reading_level '{reading_level}'.")

    prompt = build_plain_llm_prompt(
        clinical_chunk=clinical_chunk,
        reading_level=reading_level,
    )

    raw_response = _call_llm(prompt, provider=provider)
    structured = _extract_json(raw_response)
    return structured
