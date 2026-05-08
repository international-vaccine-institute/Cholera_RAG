"""CLI entrypoint for the cholera RAG system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.generator import generate_answer
from src.ingestion import (
    DEFAULT_BM25_DOCUMENTS_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_VECTOR_DB_DIR,
    ingest_pdfs,
)
from src.retriever import get_top_reranked_chunks

load_dotenv()


def ensure_gemini_key_loaded() -> None:
    """Validate GEMINI_API_KEY/GOOGLE_API_KEY and normalize env name."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    key = gemini_api_key or google_api_key

    if not key:
        raise ValueError(
            "Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env."
        )

    # LangChain integrations typically read GOOGLE_API_KEY.
    if not google_api_key:
        os.environ["GOOGLE_API_KEY"] = key

def ensure_index_ready(
    vector_db_dir: Path = DEFAULT_VECTOR_DB_DIR,
    bm25_path: Path = DEFAULT_BM25_DOCUMENTS_PATH,
) -> None:
    """Create indexes if vector DB or BM25 docs are missing."""
    has_vector_db = vector_db_dir.exists() and any(vector_db_dir.iterdir())
    has_bm25_docs = bm25_path.exists()

    if has_vector_db and has_bm25_docs:
        return

    print("Index not found. Running PDF ingestion...")
    ingest_pdfs(
        data_dir=DEFAULT_DATA_DIR,
        persist_directory=vector_db_dir,
        bm25_documents_path=bm25_path,
        embedding_provider="huggingface",
    )
    print("Ingestion complete.")


def run_cli() -> None:
    """Run interactive question-answer loop."""
    ensure_gemini_key_loaded()
    ensure_index_ready()

    print("Cholera RAG system is ready. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        docs = get_top_reranked_chunks(
            question=question,
            top_k=5,
            vector_weight=0.7,
            bm25_weight=0.3,
            reranker_provider="flashrank",
            embedding_provider="huggingface",
        )
        answer = generate_answer(question=question, retrieved_docs=docs)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    run_cli()
