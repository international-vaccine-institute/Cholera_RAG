"""Streamlit web interface for the Ethiopia cholera response RAG system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.generator import generate_answer
from src.ingestion import (
    DEFAULT_BM25_DOCUMENTS_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_VECTOR_DB_DIR,
    ingest_pdfs,
)
from src.retriever import (
    DEFAULT_FLASHRANK_MODEL_NAME,
    build_ensemble_retriever,
    build_rerank_retriever,
    get_flashrank_reranker,
    load_bm25_retriever,
    load_chroma_retriever,
)

load_dotenv()


def _api_key_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _list_reference_documents(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    return sorted(path.name for path in data_dir.glob("*.pdf"))


def _prepare_indexes() -> dict[str, bool]:
    """Ensure vector DB and BM25 index are available."""
    vector_ready = DEFAULT_VECTOR_DB_DIR.exists() and any(DEFAULT_VECTOR_DB_DIR.iterdir())
    bm25_ready = DEFAULT_BM25_DOCUMENTS_PATH.exists()

    if vector_ready and bm25_ready:
        return {"vector_ready": True, "bm25_ready": True}

    if not _list_reference_documents():
        return {"vector_ready": vector_ready, "bm25_ready": bm25_ready}

    try:
        ingest_pdfs(
            data_dir=DEFAULT_DATA_DIR,
            persist_directory=DEFAULT_VECTOR_DB_DIR,
            bm25_documents_path=DEFAULT_BM25_DOCUMENTS_PATH,
            embedding_provider="huggingface",
        )
    except Exception:
        return {"vector_ready": False, "bm25_ready": False}

    return {
        "vector_ready": DEFAULT_VECTOR_DB_DIR.exists() and any(DEFAULT_VECTOR_DB_DIR.iterdir()),
        "bm25_ready": DEFAULT_BM25_DOCUMENTS_PATH.exists(),
    }


@st.cache_resource(show_spinner=False)
def get_system_status() -> dict[str, bool]:
    return _prepare_indexes()


@st.cache_resource(show_spinner=False)
def load_resources() -> dict[str, Any]:
    """Load and cache heavy retrieval/reranking resources once."""
    chroma_retriever = load_chroma_retriever(
        persist_directory=DEFAULT_VECTOR_DB_DIR,
        embedding_provider="huggingface",
        search_k=15,
    )
    bm25_retriever = load_bm25_retriever(
        bm25_documents_path=DEFAULT_BM25_DOCUMENTS_PATH,
        search_k=15,
    )
    cross_encoder = get_flashrank_reranker(
        top_n=7,
        flashrank_model_name=DEFAULT_FLASHRANK_MODEL_NAME,
    )
    return {
        "chroma_retriever": chroma_retriever,
        "bm25_retriever": bm25_retriever,
        "cross_encoder": cross_encoder,
    }


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _doc_to_fragment(doc: Any) -> dict[str, str]:
    filename = str(doc.metadata.get("filename", "unknown"))
    page = str(doc.metadata.get("page_number", doc.metadata.get("page", "N/A")))
    content = doc.page_content.strip()
    return {"filename": filename, "page": page, "content": content}


def _render_fragments(fragments: list[dict[str, str]]) -> None:
    with st.expander("Context Fragments", expanded=False):
        for idx, fragment in enumerate(fragments, start=1):
            st.markdown(
                f"""
<div style="border:1px solid #3a3a3a; border-radius:10px; padding:12px; margin-bottom:10px;">
  <b>Fragment {idx}</b><br/>
  <b>File:</b> {fragment["filename"]} &nbsp; | &nbsp; <b>Page:</b> {fragment["page"]}
  <hr style="margin:8px 0;"/>
  <div style="white-space: pre-wrap;">{fragment["content"]}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def _render_sidebar() -> tuple[float, bool]:
    status = get_system_status()
    references = _list_reference_documents()

    st.sidebar.header("System Status")
    st.sidebar.markdown(
        f"- Vector DB: {'Loaded' if status['vector_ready'] else 'Not loaded'}\n"
        f"- BM25 Index: {'Loaded' if status['bm25_ready'] else 'Not loaded'}"
    )

    st.sidebar.header("Search Balance (Alpha)")
    alpha = st.sidebar.slider(
        "Vector vs BM25",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="alpha=1.0 means vector-only, alpha=0.0 means BM25-only",
    )
    st.sidebar.caption(f"Vector: {alpha:.2f} | BM25: {1-alpha:.2f}")

    st.sidebar.header("Reranker")
    use_reranker = st.sidebar.toggle(
        "Use reranker (FlashRank)",
        value=True,
        help="Turn off to maximize speed when latency is critical.",
    )
    if use_reranker:
        st.sidebar.caption("Enabled: FlashRank lightweight model (top 10 candidates)")
    else:
        st.sidebar.caption("Disabled: faster retrieval-only mode")

    st.sidebar.header("Reference Documents")
    if references:
        for index, name in enumerate(references, start=1):
            st.sidebar.write(f"{index}. {name}")
    else:
        st.sidebar.warning("No PDF files found in data/.")

    return alpha, use_reranker


def main() -> None:
    st.set_page_config(page_title="Cholera RAG System", page_icon=":microscope:", layout="wide")
    st.title("Ethiopia Cholera Response RAG System")

    _init_session_state()
    alpha, use_reranker = _render_sidebar()
    status = get_system_status()

    if not _api_key_available():
        st.error("Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env.")
        st.stop()

    if not (status["vector_ready"] and status["bm25_ready"]):
        st.error("Index is not ready. Add PDFs to data/ and run ingestion.")
        st.stop()

    try:
        resources = load_resources()
    except Exception as exc:
        st.error(f"Failed to load cached resources: {exc}")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("fragments"):
                _render_fragments(message["fragments"])

    user_question = st.chat_input("Ask about cholera prevention, outbreaks, or response strategy...")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                ensemble = build_ensemble_retriever(
                    chroma_retriever=resources["chroma_retriever"],
                    bm25_retriever=resources["bm25_retriever"],
                    vector_weight=alpha,
                    bm25_weight=1 - alpha,
                )

                retriever = build_rerank_retriever(
                    base_retriever=ensemble,
                    reranker_provider="flashrank" if use_reranker else "none",
                    top_n=7,
                    preloaded_compressor=resources["cross_encoder"] if use_reranker else None,
                )
                docs = retriever.invoke(user_question)[:7]
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                return

            if not docs:
                st.warning("No relevant passages were retrieved.")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I could not find relevant evidence in the indexed papers.",
                        "fragments": [],
                    }
                )
                return

            try:
                answer = generate_answer(question=user_question, retrieved_docs=docs)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

        fragments = [_doc_to_fragment(doc) for doc in docs]
        st.markdown(answer)
        _render_fragments(fragments)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "fragments": fragments}
    )


if __name__ == "__main__":
    main()
