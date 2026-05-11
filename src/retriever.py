"""Hybrid retrieval and reranking pipeline for cholera RAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

from langchain_community.retrievers import BM25Retriever

try:
    from .ingestion import (
        DEFAULT_BM25_DOCUMENTS_PATH,
        DEFAULT_VECTOR_DB_DIR,
        get_embedding_model,
        load_documents_for_bm25,
    )
except ImportError:
    from ingestion import (  # type: ignore
        DEFAULT_BM25_DOCUMENTS_PATH,
        DEFAULT_VECTOR_DB_DIR,
        get_embedding_model,
        load_documents_for_bm25,
    )


RerankerProvider = Literal["flashrank", "cohere", "none"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLASHRANK_CACHE_DIR = PROJECT_ROOT / ".cache" / "flashrank"
DEFAULT_FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
DEFAULT_RERANK_CANDIDATE_K = 15

_RERANKER_CACHE: dict[tuple[str, int, str], object] = {}
_HF_TOKEN_CONFIGURED = False

load_dotenv()


def _configure_hf_token_from_env() -> None:
    """Propagate HF_TOKEN to common HuggingFace env var names once."""
    global _HF_TOKEN_CONFIGURED
    if _HF_TOKEN_CONFIGURED:
        return

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    _HF_TOKEN_CONFIGURED = True


def _get_flashrank_compressor(
    top_n: int,
    cache_dir: Path | str,
    model_name: str = DEFAULT_FLASHRANK_MODEL_NAME,
) -> object:
    """Create or reuse a singleton FlashrankRerank compressor."""
    from flashrank import Ranker
    from langchain_community.document_compressors import FlashrankRerank

    _configure_hf_token_from_env()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_key = ("flashrank", top_n, f"{cache_path.resolve()}::{model_name}")
    compressor = _RERANKER_CACHE.get(cache_key)
    if compressor is None:
        ranker = Ranker(model_name=model_name, cache_dir=str(cache_path))
        compressor = FlashrankRerank(
            client=ranker,
            top_n=top_n,
            model=model_name,
        )
        _RERANKER_CACHE[cache_key] = compressor
    return compressor


def _get_cohere_compressor(top_n: int) -> object:
    """Create or reuse a singleton Cohere reranker compressor."""
    cache_key = ("cohere", top_n, "default")
    compressor = _RERANKER_CACHE.get(cache_key)
    if compressor is not None:
        return compressor

    try:
        from langchain_cohere import CohereRerank
    except ImportError:
        from langchain.retrievers.document_compressors import CohereRerank

    compressor = CohereRerank(top_n=top_n)
    _RERANKER_CACHE[cache_key] = compressor
    return compressor


def load_chroma_retriever(
    persist_directory: Path | str = DEFAULT_VECTOR_DB_DIR,
    embedding_provider: Literal["openai", "huggingface"] = "huggingface",
    embedding_model_name: str | None = None,
    collection_name: str = "cholera_literature",
    search_k: int = 20,
) -> BaseRetriever:
    """Load persisted Chroma collection and return a retriever."""
    embeddings = get_embedding_model(
        provider=embedding_provider,
        model_name=embedding_model_name,
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return vector_store.as_retriever(search_kwargs={"k": search_k})


def load_bm25_retriever(
    bm25_documents_path: Path | str = DEFAULT_BM25_DOCUMENTS_PATH,
    search_k: int = 20,
) -> BM25Retriever:
    """Load BM25 retriever from saved split documents."""
    documents = load_documents_for_bm25(bm25_documents_path)
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = search_k
    return bm25


def build_ensemble_retriever(
    chroma_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
) -> EnsembleRetriever:
    """Combine dense and sparse retrievers with weighted fusion."""
    return EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )


def build_rerank_retriever(
    base_retriever: BaseRetriever,
    reranker_provider: RerankerProvider = "flashrank",
    top_n: int = 5,
    flashrank_cache_dir: Path | str = DEFAULT_FLASHRANK_CACHE_DIR,
    flashrank_model_name: str = DEFAULT_FLASHRANK_MODEL_NAME,
    preloaded_compressor: object | None = None,
) -> BaseRetriever:
    """Wrap retriever with optional reranker."""
    provider = reranker_provider.lower()
    if provider == "none":
        return base_retriever

    if preloaded_compressor is not None:
        compressor = preloaded_compressor
    elif provider == "flashrank":
        compressor = _get_flashrank_compressor(
            top_n=top_n,
            cache_dir=flashrank_cache_dir,
            model_name=flashrank_model_name,
        )
    elif provider == "cohere":
        compressor = _get_cohere_compressor(top_n=top_n)
    else:
        raise ValueError(f"Unsupported reranker_provider: {reranker_provider}")

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def get_flashrank_reranker(
    top_n: int = 5,
    flashrank_cache_dir: Path | str = DEFAULT_FLASHRANK_CACHE_DIR,
    flashrank_model_name: str = DEFAULT_FLASHRANK_MODEL_NAME,
) -> object:
    """Expose singleton FlashRank compressor for external caching layers."""
    return _get_flashrank_compressor(
        top_n=top_n,
        cache_dir=flashrank_cache_dir,
        model_name=flashrank_model_name,
    )


def get_top_reranked_chunks(
    question: str,
    top_k: int = 5,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
    reranker_provider: RerankerProvider = "flashrank",
    vector_db_dir: Path | str = DEFAULT_VECTOR_DB_DIR,
    bm25_documents_path: Path | str = DEFAULT_BM25_DOCUMENTS_PATH,
    embedding_provider: Literal["openai", "huggingface"] = "huggingface",
    embedding_model_name: str | None = None,
    flashrank_cache_dir: Path | str = DEFAULT_FLASHRANK_CACHE_DIR,
    flashrank_model_name: str = DEFAULT_FLASHRANK_MODEL_NAME,
    rerank_candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
) -> list[Document]:
    """Return top reranked chunks for a user question."""
    if reranker_provider == "none":
        candidate_k = max(top_k, 5)
    else:
        candidate_k = max(top_k, rerank_candidate_k)

    chroma_retriever = load_chroma_retriever(
        persist_directory=vector_db_dir,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        search_k=candidate_k,
    )
    bm25_retriever = load_bm25_retriever(
        bm25_documents_path=bm25_documents_path,
        search_k=candidate_k,
    )

    ensemble = build_ensemble_retriever(
        chroma_retriever=chroma_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

    retriever = build_rerank_retriever(
        base_retriever=ensemble,
        reranker_provider=reranker_provider,
        top_n=top_k,
        flashrank_cache_dir=flashrank_cache_dir,
        flashrank_model_name=flashrank_model_name,
    )
    return retriever.invoke(question)[:top_k]


if __name__ == "__main__":
    sample_question = "What are the major transmission pathways of cholera?"
    docs = get_top_reranked_chunks(sample_question)
    print(f"Retrieved {len(docs)} reranked chunks.")
