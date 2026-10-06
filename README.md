# Clinical RAG Simplification — Person A Module

> **Scope:** RAG Pipeline only.  
> Person B handles: language simplification layer, medical NER, glossary generation, readability optimisation.  
> Person C handles: document parsing, FastAPI backend, frontend, authentication.

---

## Project Structure

```
clinical-rag-simplification/
│
├── data/
│   ├── knowledge_base/
│   │   ├── guidelines/          ← Clinical guidelines (.txt / .md)
│   │   ├── textbooks/           ← Medical textbooks  (.txt / .md)
│   │   └── trusted_sources/     ← Other trusted references
│   └── evaluation_samples/      ← 10 sample clinical documents for evaluation
│
├── vector_store/                 ← Auto-created by ingestion (FAISS index + metadata)
│
├── src/
│   ├── config.py                 ← All configuration (single source of truth)
│   ├── ingestion.py              ← Knowledge base ingestion pipeline
│   ├── chunking.py               ← Semantic chunking module
│   ├── embeddings.py             ← Sentence Transformer embedding module
│   ├── retrieval.py              ← FAISS vector store (create/save/load/search)
│   ├── reranking.py              ← Optional cross-encoder reranking
│   ├── prompts.py                ← Prompt templates (RAG + Plain LLM)
│   ├── llm.py                    ← LLM interface (Groq / OpenAI / Google / Anthropic)
│   └── rag_pipeline.py           ← Complete RAG pipeline (main public API)
│
├── evaluation/
│   ├── evaluate_rag.py           ← RAG vs. Plain LLM evaluation script
│   └── results/                  ← Auto-created; contains per-run outputs
│
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   └── test_retrieval.py
│
├── ingest_knowledge_base.py      ← CLI: build the vector index
├── test_rag.py                   ← CLI: interactive standalone test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` is used. If you have a GPU, replace with `faiss-gpu`.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
LLM_PROVIDER=groq            # or: openai | google | anthropic
LLM_MODEL=qwen/qwen3.8-27b   # or: llama-3.1-8b-instant | gpt-4o-mini | gemini-1.5-flash
LLM_API_KEY=your_actual_api_key_here
```

> ⚠️ **Never commit `.env` to version control.**

### 4. Download NLTK data (first-time only)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Workflow

### Step 1 — Add Knowledge Base Documents

Place trusted medical reference documents (`.txt` or `.md`) in:

```
data/knowledge_base/guidelines/       ← Clinical guidelines
data/knowledge_base/textbooks/        ← Medical textbooks
data/knowledge_base/trusted_sources/  ← Other trusted references
```

> Seed documents are already included. Add your own to improve retrieval quality.

### Step 2 — Build the Vector Index

```bash
python ingest_knowledge_base.py
```

Options:
```
--rebuild    Force full rebuild of the index
--verbose    Show detailed log output
```

### Step 3 — Run the Interactive Test

```bash
python test_rag.py
```

Paste clinical text, choose a reading level, and see the structured output.

### Step 4 — Run the Evaluation

```bash
python evaluation/evaluate_rag.py
python evaluation/evaluate_rag.py --level Intermediate
```

Results are saved to `evaluation/results/run_<timestamp>/`.

### Step 5 — Run Unit Tests

```bash
pytest tests/ -v
```

---

## Configuration Reference

All parameters are in `src/config.py` and controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K` | `5` | Number of retrieved references per chunk |
| `USE_RERANKER` | `false` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RERANKER_TOP_K` | `3` | Chunks to keep after reranking |
| `LLM_PROVIDER` | `openai` | LLM provider: openai \| google \| anthropic |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_API_KEY` | _(required)_ | Your API key |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_MAX_TOKENS` | `1500` | Max response tokens |
| `MAX_CHUNK_TOKENS` | `300` | Approximate token budget per chunk |
| `CHUNK_OVERLAP_SENTENCES` | `1` | Sentences of overlap between chunks |
| `DEFAULT_READING_LEVEL` | `Basic` | Default reading level |

---

## Integration Guide for Person C (FastAPI Backend)

### Import the pipeline

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()           # loads FAISS index on first call
```

### Call process()

```python
result = pipeline.process(
    clinical_text="Patient should take Aspirin 75mg daily...",
    reading_level="Basic"           # "Basic" | "Intermediate" | "Advanced"
)
```

### Response schema (JSON-serialisable dict)

```json
{
  "reading_level": "Basic",
  "total_chunks": 3,
  "disclaimer": "This output is an AI-assisted simplification...",
  "results": [
    {
      "chunk_id": "chunk_a1b2c3d4",
      "original_chunk": "Patient should take...",
      "section_title": "MEDICATIONS",
      "chunk_index": 0,
      "simplified_output": {
        "simple_explanation": "You need to take a small daily aspirin tablet...",
        "important_instructions": ["Take every day at the same time"],
        "medication_guidance": ["Aspirin 75 mg once daily"],
        "follow_up": ["See your GP in 2 weeks"]
      },
      "retrieved_references": [
        {
          "text": "Aspirin 75–100 mg daily is recommended for...",
          "source": "Clinical Guideline",
          "document_name": "Cardiac Medications Reference",
          "page": null,
          "section": "ANTIPLATELET THERAPY",
          "chunk_id": "kb_abc123",
          "score": 0.8742
        }
      ]
    }
  ]
}
```

### FastAPI integration example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline

app = FastAPI()
pipeline = RAGPipeline()

class SimplifyRequest(BaseModel):
    clinical_text: str
    reading_level: str = "Basic"

@app.post("/simplify")
async def simplify(request: SimplifyRequest):
    try:
        result = pipeline.process(
            clinical_text=request.clinical_text,
            reading_level=request.reading_level,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(503, "Vector index not built. Run ingest_knowledge_base.py.")
    except EnvironmentError as e:
        raise HTTPException(500, str(e))
```

---

## Integration Guide for Person B (Language Simplification Layer)

The `simplified_output` dict in each `ChunkResult` is the hook for Person B's layer.

Person B's module can:
1. **Receive** the `simple_explanation` and instruction lists from `simplified_output`.
2. **Post-process** them with medical NER, glossary annotation, and readability scoring.
3. **Return** enhanced output without modifying the RAG pipeline itself.

Suggested interface:

```python
# Person B's module (example)
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
raw_result = pipeline.process(clinical_text, reading_level)

# Person B processes each chunk's simplified_output
for chunk in raw_result["results"]:
    simplified = chunk["simplified_output"]
    enhanced = person_b_enhance(simplified)  # NER, glossary, readability
    chunk["simplified_output"] = enhanced
```

---

## Example Output

```
======================================================================
  RAG PIPELINE OUTPUT  |  Reading Level: Basic  |  Chunks: 2
======================================================================

⚠️  This output is an AI-assisted simplification for educational purposes only.

======================================================================
  CHUNK 1 [MEDICATIONS AT DISCHARGE]
======================================================================

Original Clinical Text:
  Ticagrelor 90 mg twice daily — continue for 12 months;
  do NOT stop without consulting your cardiologist

Retrieved References (2 found):
  1. Clinical Guideline — Anticoagulation Guidelines  [score: 0.891]
  2. Medical Textbook — Cardiac Medications Reference  [score: 0.843]

Simple Explanation:
  You are taking a blood-thinning tablet called Ticagrelor. You must
  take it twice a day for 12 months. This medicine keeps your heart
  stent open and safe.

Important Instructions (2):
  • Take Ticagrelor every day without missing a dose
  • NEVER stop this medicine without talking to your heart doctor first

Medication Guidance (1):
  • Ticagrelor 90 mg — take twice daily (morning and evening)

Follow-up (1):
  • Cardiology outpatient clinic in 4 weeks
```

---

## Medical Safety Disclaimer

> This system is an **academic prototype** for clinical document simplification.  
> It is **NOT a medical diagnosis or treatment system**.  
> All outputs should be reviewed by a qualified healthcare professional before being shared with patients.  
> The RAG pipeline is designed to ground outputs in trusted references, but **cannot guarantee medical accuracy**.

---

## Evaluation Methodology

The `evaluate_rag.py` script:
1. Runs both Plain LLM and RAG pipelines on 10 clinical sample documents.
2. Saves raw JSON outputs for each.
3. Generates a **manual evaluation template** (Markdown) for each document.

**Medical accuracy scores are NOT auto-generated** — they require human review by someone with clinical knowledge. The template guides the reviewer through structured scoring on:
- Factual grounding
- Medical accuracy
- Unsupported claims / hallucinations
- Instruction coverage
- Relevance
- Explanation quality
# Medical Text Simplifier

A Python package for simplifying medical text across different reading levels while preserving critical information like dosages, lab values, and important instructions. Designed as the language simplification layer in a medical RAG pipeline with stable JSON API for FastAPI integration.

## Installation

```bash
# Install package dependencies
pip install -r requirements.txt

# Download required scispaCy model
python -m spacy download en_core_sci_md

# Set up environment variable for LLM API (optional)
export ANTHROPIC_API_KEY="your-api-key-here"
```

## API Reference

### Main Function

```python
def simplify(text_or_sections, level="basic") -> dict
```

**Parameters:**
- `text_or_sections`: Input text in one of two formats:
  - `str`: Plain medical text to be simplified
  - `dict`: Structured sections with keys:
    - `simple_explanation` (str): Main content explanation
    - `important_instructions` (str): Critical patient instructions
    - `medication_guidance` (str): Medication-related information  
    - `follow_up` (str): Follow-up care instructions
- `level`: Target reading level, one of:
  - `"basic"`: 6th grade or below, all terms simplified inline, short sentences
  - `"intermediate"`: 8th-10th grade, key terms annotated in brackets
  - `"advanced"`: Original terminology preserved, definitions in glossary only

**Returns:**
Dictionary with the following JSON-serializable structure:

```python
{
  "level": str,                                    # Applied simplification level
  "sections": {                                    # Text organized by section
    "simple_explanation": str,
    "important_instructions": str, 
    "medication_guidance": str,
    "follow_up": str
  },
  "simplified_text": str,                         # Full markdown-formatted text
  "entities": [                                   # Extracted medical entities
    {
      "text": str,                                 # Entity text
      "type": "DISEASE|DRUG|PROCEDURE|LAB_VALUE|ABBREVIATION",
      "start": int,                              # Position in simplified text
      "end": int
    }
  ],
  "glossary": [                                  # Alphabetical term definitions
    {
      "term": str,
      "type": str,
      "explanation": str
    }
  ],
  "highlights": [                                # Important sentences to emphasize
    {
      "section": str,
      "sentence": str
    }
  ],
  "scores": {                                    # Readability metrics
    "original": {
      "flesch_reading_ease": float,              # 0-100, higher = easier
      "grade_level": float                       # Flesch-Kincaid grade level
    },
    "simplified": {
      "flesch_reading_ease": float,
      "grade_level": float
    }
  },
  "protected_values": [str]                      # Preserved dosages/units/frequencies
}
```

## Usage Examples

### Basic Usage with Plain Text

```python
from simplifier import simplify

# Simple text input
result = simplify(
    "Your HbA1c level is 8.2%. Take metformin 500mg twice daily with meals.",
    level="basic"
)

print(result['simplified_text'])
print(f"Reading level improved from {result['scores']['original']['grade_level']:.1f} to {result['scores']['simplified']['grade_level']:.1f}")
```

### Structured Input with Sections

```python
from simplifier import simplify

# Structured input matching RAG output format
medical_report = {
    "simple_explanation": "Your blood sugar test (HbA1c) shows you have diabetes. The level is 8.2%, which is higher than the target of 7%.",
    "important_instructions": "Take your medication exactly as prescribed. Check your blood sugar daily.",
    "medication_guidance": "Metformin 500mg: Take 2 times per day with breakfast and dinner.", 
    "follow_up": "Return in 3 months for blood work to check your progress."
}

result = simplify(medical_report, level="intermediate")
```

### FastAPI Integration

```python
from fastapi import FastAPI
from simplifier import simplify

app = FastAPI()

@app.post("/simplify")
async def simplify_text(
    text: str,
    level: str = "basic"
):
    """Endpoint for medical text simplification."""
    try:
        result = simplify(text, level)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Simplification failed")
```

## Features

### 🔍 Medical Entity Recognition
- Diseases, drugs, procedures, lab values, and abbreviations
- Uses scispaCy (en_core_sci_md) with custom abbreviation detection
- Regex-based lab value extraction with units

### 🔒 Number Safety
- Automatically detects and preserves dosages, units, frequencies, lab values
- Masks values during rewriting, restores exactly after simplification
- Validation ensures protected values are never altered

### 📚 Term Simplification  
- Curated dictionary of 100+ common medical terms
- LLM fallback for unknown terms (Anthropic API)
- Caching for performance optimization
- Offline rule-based fallback for testing

### 📖 Reading Level Control
- **Basic**: 6th grade, short sentences (<15 words), all terms explained inline
- **Intermediate**: 8th-10th grade, key terms in brackets [like this]
- **Advanced**: Original terminology, glossary definitions only

### 📊 Quality Metrics
- Flesch Reading Ease and Flesch-Kincaid grade level scoring
- Before/after readability comparison
- Entity recognition precision/recall evaluation

### 🎨 Structured Output
- Markdown formatting with section headings
- Bullet points for instructions
- Bold highlighting for critical information
- Mobile-friendly short paragraphs

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=simplifier
```

### Demo Scripts
```bash
# Process a single document
python run_demo.py sample.txt --level basic

# Evaluate all sample documents
python evaluate_B.py
# Generates results.csv with metrics
```

### Project Structure
```
simplifier/
├── __init__.py          # Main API exports
├── simplify.py          # Core simplification logic
├── schema.py            # Pydantic models
├── ner.py              # Named entity recognition
├── masking.py          # Number protection system
├── glossary.py         # Term definition generation
├── readability.py      # Reading level metrics
└── layout.py           # Markdown formatting

data/
├── sample_docs/        # Test medical documents
├── mock_A_outputs/     # Sample RAG pipeline outputs  
├── ner_labels.json     # NER evaluation dataset
└── term_map.json       # Medical term dictionary

tests/
├── test_ner.py         # Entity recognition tests
├── test_masking.py     # Number preservation tests
├── test_readability.py # Reading level tests
└── test_schema.py      # API schema validation
```

## Error Handling

The `simplify()` function raises:
- `ValueError`: Invalid input format or unsupported level
- `RuntimeError`: External dependency failures (scispaCy, API calls)

For production use, wrap calls in try-except blocks and provide fallback responses.

## License

MIT License - see LICENSE file for details.

---

## RAG Pipeline Integration

The `simplify_rag_result()` function is designed to work seamlessly with Person A's RAG pipeline output, adding medical text simplification while preserving all original structure and references.

### RAG Adapter Function

```python
def simplify_rag_result(rag_result: dict, level: str = None) -> dict
```

**Parameters:**
- `rag_result`: Exact output dict from Person A's `RAGPipeline.process()` method
- `level`: Target simplification level ("basic", "intermediate", "advanced"). Defaults to `rag_result["reading_level"]` normalized to lowercase.

**Input Schema** (from RAG Pipeline):
```python
{
  "reading_level": "Basic|Intermediate|Advanced",
  "total_chunks": int,
  "disclaimer": str,
  "results": [
    {
      "chunk_id": str,
      "original_chunk": str,
      "section_title": str,
      "chunk_index": int,
      "simplified_output": {
        "simple_explanation": str,
        "important_instructions": [str],
        "medication_guidance": [str],
        "follow_up": [str]
      },
      "retrieved_references": [
        {
          "text": str,
          "source": str,
          "document_name": str,
          # ... other reference fields
        }
      ]
    }
  ]
}
```

**Output Schema** (Enhanced RAG Result):
```python
{
  # Original top-level fields preserved exactly
  "reading_level": str,
  "total_chunks": int,
  "disclaimer": str,
  
  # Enhanced results with simplification features
  "results": [
    {
      # Original chunk fields preserved exactly
      "chunk_id": str,
      "original_chunk": str,
      "section_title": str,
      "chunk_index": int,
      "simplified_output": {
        "simple_explanation": str,              # Processed through simplification
        "important_instructions": [str],        # Each list item processed
        "medication_guidance": [str],           # Lists preserved as lists
        "follow_up": [str]
      },
      "retrieved_references": [...],            # Preserved exactly as-is
      
      # New fields added by simplify_rag_result
      "entities": [EntityModel],                # Extracted medical entities
      "glossary": [GlossaryEntry],             # Terms defined for this chunk
      "highlights": [Highlight],               # Important sentences to emphasize
      "scores": {                              # Before/after readability
        "original": ReadabilityScore,
        "simplified": ReadabilityScore
      }
    }
  ],
  
  # New top-level aggregated fields
  "glossary": [GlossaryEntry],                # Combined across all chunks
  "scores": ReadabilityScores,                # Averaged across all chunks
  "protected_values_check": {                 # Validation results
    "passed": bool,
    "missing": [str]                          # List of values lost during processing
  }
}
```

### FastAPI Integration Example

```python
from fastapi import FastAPI, HTTPException
from src.rag_pipeline import RAGPipeline
from simplifier import simplify_rag_result

app = FastAPI()
rag_pipeline = RAGPipeline()

class ProcessRequest(BaseModel):
    clinical_text: str
    reading_level: str = "Basic"
    simplification_level: Optional[str] = None  # Override RAG level if needed

@app.post("/process")
async def process_clinical_text(request: ProcessRequest):
    """Complete pipeline: RAG retrieval + medical simplification."""
    try:
        # Step 1: RAG pipeline processing
        rag_result = rag_pipeline.process(
            clinical_text=request.clinical_text,
            reading_level=request.reading_level
        )
        
        # Step 2: Medical simplification layer
        enhanced_result = simplify_rag_result(
            rag_result=rag_result,
            level=request.simplification_level  # Optional override
        )
        
        return enhanced_result
        
    except FileNotFoundError:
        raise HTTPException(503, "Vector index not built")
    except ValueError as e:
        raise HTTPException(400, f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

@app.post("/simplify-rag")
async def simplify_existing_rag_result(rag_result: dict, level: str = "basic"):
    """Process existing RAG result through simplification layer."""
    try:
        result = simplify_rag_result(rag_result, level)
        return result
    except Exception as e:
        raise HTTPException(500, f"Simplification failed: {e}")
```

### Key Features

- **Structure Preservation**: All original chunk metadata (`chunk_id`, `original_chunk`, `section_title`, `chunk_index`, `retrieved_references`) is preserved exactly as provided by the RAG pipeline.

- **List Processing**: The `important_instructions`, `medication_guidance`, and `follow_up` fields remain as lists. Each list item is processed individually through the simplification pipeline while maintaining the list structure.

- **Protected Values**: Dosages, lab values, units, and frequencies are automatically detected and preserved during simplification. The `protected_values_check` field reports whether all values were successfully preserved.

- **Level Normalization**: The function accepts level parameters in any case ("Basic", "INTERMEDIATE", "advanced") and normalizes them to lowercase for internal processing.

- **Offline Support**: Works in offline mode using rule-based simplification when LLM APIs are unavailable.

- **Schema Validation**: Output is validated against Pydantic models to ensure JSON serialization compatibility.

### Usage Example

```python
import json
from src.rag_pipeline import RAGPipeline  
from simplifier import simplify_rag_result

# Initialize RAG pipeline
pipeline = RAGPipeline()

# Clinical text input
clinical_text = """
Patient has diabetes mellitus type 2 with HbA1c of 8.2%. 
Current medication includes metformin 500 mg twice daily.
Blood pressure consistently elevated at 158/94 mmHg.
Started on lisinopril 10 mg once daily.
"""

# Step 1: Process through RAG pipeline
rag_result = pipeline.process(
    clinical_text=clinical_text,
    reading_level="Basic"
)

# Step 2: Enhance with medical simplification
enhanced_result = simplify_rag_result(rag_result)

# The result contains:
print(f"Processed {enhanced_result['total_chunks']} chunks")
print(f"Protected values preserved: {enhanced_result['protected_values_check']['passed']}")
print(f"Aggregate glossary contains {len(enhanced_result['glossary'])} terms")

# Access enhanced chunk data
for chunk in enhanced_result['results']:
    print(f"Chunk {chunk['chunk_index']}: {chunk['section_title']}")
    print(f"  Entities found: {len(chunk['entities'])}")
    print(f"  Grade level: {chunk['scores']['original']['grade_level']:.1f} → {chunk['scores']['simplified']['grade_level']:.1f}")
```