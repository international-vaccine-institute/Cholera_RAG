"""PDF ingestion and indexing pipeline for the cholera RAG system.

This module loads PDF files from ``data/``, splits them into retrievable chunks,
stores dense embeddings in Chroma, and saves the split documents for BM25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


EmbeddingProvider = Literal["openai", "huggingface"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
DEFAULT_BM25_DOCUMENTS_PATH = DEFAULT_VECTOR_DB_DIR / "bm25_documents.jsonl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdf_documents(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[Document]:
    """Load all PDFs from ``data_dir`` and normalize source metadata."""
    data_path = Path(data_dir)
    pdf_paths = sorted(data_path.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_path}")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page_index, page in enumerate(pages):
            raw_page = page.metadata.get("page", page_index)
            page_number = int(raw_page) + 1 if isinstance(raw_page, int) else page_index + 1

            page.metadata.update(
                {
                    "filename": pdf_path.name,
                    "page_number": page_number,
                    "source_path": str(pdf_path),
                }
            )
            documents.append(page)

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split loaded PDF pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def get_embedding_model(
    provider: EmbeddingProvider = "huggingface",
    model_name: str | None = None,
) -> Any:
    """Create an embedding model for Chroma.

    OpenAI requires ``OPENAI_API_KEY`` in the environment. HuggingFace defaults
    to a small sentence-transformer model suitable for local development.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name or "text-embedding-3-small")

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def build_chroma_index(
    documents: list[Document],
    persist_directory: Path | str = DEFAULT_VECTOR_DB_DIR,
    embedding_provider: EmbeddingProvider = "huggingface",
    embedding_model_name: str | None = None,
    collection_name: str = "cholera_literature",
) -> Chroma:
    """Embed split documents and persist them to a local Chroma database."""
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model(
        provider=embedding_provider,
        model_name=embedding_model_name,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )

    persist = getattr(vector_store, "persist", None)
    if callable(persist):
        persist()

    return vector_store


def save_documents_for_bm25(
    documents: list[Document],
    output_path: Path | str = DEFAULT_BM25_DOCUMENTS_PATH,
) -> Path:
    """Save split documents locally as JSONL for a later BM25 index build."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return output


def load_documents_for_bm25(
    input_path: Path | str = DEFAULT_BM25_DOCUMENTS_PATH,
) -> list[Document]:
    """Load previously saved split documents for BM25 retrieval."""
    input_file = Path(input_path)

    documents: list[Document] = []
    with input_file.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            documents.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata", {}),
                )
            )

    return documents


def ingest_pdfs(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    persist_directory: Path | str = DEFAULT_VECTOR_DB_DIR,
    embedding_provider: EmbeddingProvider = "huggingface",
    embedding_model_name: str | None = None,
    bm25_documents_path: Path | str = DEFAULT_BM25_DOCUMENTS_PATH,
) -> tuple[Chroma, list[Document]]:
    """Run the full ingestion pipeline and return Chroma plus split documents."""
    pages = load_pdf_documents(data_dir)
    split_docs = split_documents(pages)

    vector_store = build_chroma_index(
        documents=split_docs,
        persist_directory=persist_directory,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
    )
    save_documents_for_bm25(split_docs, bm25_documents_path)

    return vector_store, split_docs


if __name__ == "__main__":
    _, indexed_documents = ingest_pdfs()
    print(f"Indexed {len(indexed_documents)} chunks into {DEFAULT_VECTOR_DB_DIR}")
