# src/__init__.py
"""
Person A — Clinical RAG Pipeline package.

Public surface exposed to Person C (FastAPI integration):

    from src.rag_pipeline import RAGPipeline
    from src.chunking import chunk_clinical_text
    from src.embeddings import embed_text, embed_documents
    from src.retrieval import FAISSStore, retrieve
    from src.ingestion import ingest_knowledge_base
"""
