"""
prompts.py — Prompt templates for the clinical RAG pipeline.

Keep all prompt text here so they can be tuned independently
of the pipeline logic.
"""

from __future__ import annotations

from typing import List

from src.config import VALID_READING_LEVELS
from src.retrieval import RetrievedChunk


# ── Reading-level instructions ───────────────────────────────────────────────

_READING_LEVEL_INSTRUCTIONS: dict[str, str] = {
    "Basic": (
        "Use very simple, plain language (Grade 6 level). "
        "Avoid all medical jargon. "
        "Use short sentences. "
        "Explain medical terms in everyday words if they cannot be avoided. "
        "Write as if explaining to someone with no medical background."
    ),
    "Intermediate": (
        "Use clear, accessible language (Grade 10 level). "
        "Minimal jargon; define key medical terms when first used. "
        "Sentences may be moderately complex. "
        "Suitable for someone with basic health literacy."
    ),
    "Advanced": (
        "You may use standard medical terminology. "
        "Maintain clinical precision but remain clear and well-structured. "
        "Suitable for a healthcare student or informed patient."
    ),
}


# ── Context builder ──────────────────────────────────────────────────────────

def _format_references(references: List[RetrievedChunk]) -> str:
    """
    Format retrieved reference chunks as a numbered context block
    for inclusion in the prompt.
    """
    if not references:
        return "No trusted references were retrieved for this section."

    parts = []
    for i, ref in enumerate(references, start=1):
        source_label = ref.source
        if ref.document_name and ref.document_name != ref.source:
            source_label += f" — {ref.document_name}"
        if ref.page_number:
            source_label += f", Page {ref.page_number}"
        if ref.section_title:
            source_label += f" [{ref.section_title}]"

        parts.append(
            f"[Reference {i}] Source: {source_label}\n"
            f"Content: {ref.text}"
        )

    return "\n\n".join(parts)


# ── Main prompt builder ──────────────────────────────────────────────────────

def build_simplification_prompt(
    clinical_chunk: str,
    references: List[RetrievedChunk],
    reading_level: str = "Basic",
) -> str:
    """
    Build the full prompt to send to the LLM.

    Parameters
    ----------
    clinical_chunk : str
        The original clinical text segment to simplify.
    references : List[RetrievedChunk]
        Trusted medical references retrieved from the knowledge base.
    reading_level : str
        One of "Basic", "Intermediate", "Advanced".

    Returns
    -------
    str
        The complete prompt string.

    Raises
    ------
    ValueError
        If reading_level is not one of the valid options.
    """
    if reading_level not in VALID_READING_LEVELS:
        raise ValueError(
            f"Invalid reading level '{reading_level}'. "
            f"Choose from: {VALID_READING_LEVELS}"
        )

    level_instruction = _READING_LEVEL_INSTRUCTIONS[reading_level]
    formatted_refs = _format_references(references)

    prompt = f"""You are a clinical document simplification assistant.
Your task is to simplify a section of a clinical/medical document for a patient.

=== READING LEVEL ===
{reading_level}: {level_instruction}

=== TRUSTED MEDICAL REFERENCES ===
Use ONLY the following trusted references as your factual basis.
Do NOT invent, extrapolate, or assume any medical facts beyond what is stated here.

{formatted_refs}

=== ORIGINAL CLINICAL TEXT ===
{clinical_chunk}

=== STRICT RULES ===
1. Ground every factual statement in the provided references.
2. Do NOT invent medication names, dosages, or treatment plans.
3. Do NOT change medication dosage — reproduce it exactly as stated.
4. Do NOT add treatment recommendations not supported by the references.
5. Preserve ALL warnings, critical instructions, and follow-up actions.
6. Do NOT remove any medication guidance from the original text.
7. If a piece of information is NOT supported by the retrieved references,
   explicitly state: "Note: This information was not found in the retrieved
   references. Please consult your healthcare provider."
8. Do NOT present this output as a medical diagnosis or treatment plan.

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON in exactly this structure — no extra text, no markdown fences:

{{
  "simple_explanation": "<Plain-language summary of what this section says>",
  "important_instructions": [
    "<Instruction 1>",
    "<Instruction 2>"
  ],
  "medication_guidance": [
    "<Exact medication name, dosage, frequency as stated in source>"
  ],
  "follow_up": [
    "<Follow-up action or appointment>"
  ]
}}

If a section (e.g. medication_guidance) is not applicable, return an empty list [].
"""

    return prompt


# ── Plain-LLM prompt (for evaluation comparison) ────────────────────────────

def build_plain_llm_prompt(
    clinical_chunk: str,
    reading_level: str = "Basic",
) -> str:
    """
    Build a prompt for the Plain LLM baseline (no retrieved context).
    Used in the RAG vs. Plain LLM evaluation.
    """
    if reading_level not in VALID_READING_LEVELS:
        raise ValueError(
            f"Invalid reading level '{reading_level}'. "
            f"Choose from: {VALID_READING_LEVELS}"
        )

    level_instruction = _READING_LEVEL_INSTRUCTIONS[reading_level]

    prompt = f"""You are a clinical document simplification assistant.
Your task is to simplify a section of a clinical/medical document for a patient.

=== READING LEVEL ===
{reading_level}: {level_instruction}

=== ORIGINAL CLINICAL TEXT ===
{clinical_chunk}

=== RULES ===
1. Preserve all medication names, dosages, and frequencies exactly.
2. Preserve all warnings and critical instructions.
3. Do NOT invent medical facts.
4. Do NOT add unsupported treatment recommendations.

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON in exactly this structure — no extra text, no markdown fences:

{{
  "simple_explanation": "<Plain-language summary>",
  "important_instructions": ["<Instruction 1>", "<Instruction 2>"],
  "medication_guidance": ["<Medication details>"],
  "follow_up": ["<Follow-up action>"]
}}

If a section is not applicable, return an empty list [].
"""

    return prompt
