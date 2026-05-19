"""Streamlit web interface for the Ethiopia cholera response RAG system."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.generator import build_llm, generate_answer, rewrite_query
from src.ingestion import (
    DEFAULT_BM25_DOCUMENTS_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_VECTOR_DB_DIR,
    ingest_pdfs,
)
from src.retriever import (
    DEFAULT_FLASHRANK_MODEL_NAME,
    _looks_like_synthesis_query,
    build_ensemble_retriever,
    build_multi_query_retriever,
    build_rerank_retriever,
    get_flashrank_reranker,
    get_per_document_chunks,
    load_bm25_retriever,
    load_chroma_retriever,
    load_chroma_store,
)

load_dotenv()


def _api_key_available() -> bool:
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )


def _list_reference_documents(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    def _numeric_prefix(name: str) -> int:
        m = re.match(r"^(\d+)", name)
        return int(m.group(1)) if m else 999

    return sorted(
        (path.name for path in data_dir.glob("*.pdf")),
        key=_numeric_prefix,
    )


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
    chroma_store = load_chroma_store(
        persist_directory=DEFAULT_VECTOR_DB_DIR,
        embedding_provider="huggingface",
    )
    chroma_retriever = chroma_store.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = load_bm25_retriever(
        bm25_documents_path=DEFAULT_BM25_DOCUMENTS_PATH,
        search_k=15,
    )
    cross_encoder = get_flashrank_reranker(
        top_n=7,
        flashrank_model_name=DEFAULT_FLASHRANK_MODEL_NAME,
    )
    return {
        "chroma_store": chroma_store,
        "chroma_retriever": chroma_retriever,
        "bm25_retriever": bm25_retriever,
        "cross_encoder": cross_encoder,
    }


EXAMPLE_QUESTIONS = [
    "What is the case fatality rate (CFR) for cholera in Ethiopia reported in the retrospective analysis?",
    "According to the genomic analysis, how many times has the seventh pandemic lineage been introduced into Africa since 1970, and from which region?",
    "What were the two-dose OCV coverage rates by age group in the Shashemene vaccination campaign?",
    "According to the Ethiopia National Cholera Elimination Plan 2022–2028, what is the total budget and how is it allocated across intervention areas?",
    "What are the recommended rehydration protocols for severe cholera dehydration according to the management guidelines?",
]


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "used_examples" not in st.session_state:
        st.session_state.used_examples = set()


def _doc_to_fragment(doc: Any) -> dict[str, str]:
    filename = str(doc.metadata.get("filename", "unknown"))
    page = str(doc.metadata.get("page_number", doc.metadata.get("page", "N/A")))
    content = doc.page_content.strip()
    return {"filename": filename, "page": page, "content": content}


def _render_answer(text: str, docs: list[Any] | None = None) -> None:
    """Render answer body and deduplicated citation badges.

    If ``docs`` is provided, the SOURCES section is built directly from the
    retrieved document metadata so it is always consistent regardless of which
    [Doc N] tags the LLM chose to emit.
    """
    strip_pattern = re.compile(r'\[Source:[^\]]+\]')
    body = strip_pattern.sub("", text).strip()
    body = body.replace("$", r"\$")
    st.markdown(body)

    # Build source map: filename → sorted unique pages.
    seen: dict[str, list[str]] = {}

    if docs:
        for doc in docs:
            filename = str(doc.metadata.get("filename", "unknown"))
            page = str(doc.metadata.get("page_number", doc.metadata.get("page", "N/A")))
            if filename not in seen:
                seen[filename] = []
            if page not in seen[filename]:
                seen[filename].append(page)
    else:
        citation_pattern = re.compile(r'\[Source:\s*(.*?),\s*p\.([\w/]+)\]')
        for filename, page in citation_pattern.findall(text):
            filename = filename.strip()
            page = page.strip()
            if filename not in seen:
                seen[filename] = []
            if page not in seen[filename]:
                seen[filename].append(page)

    if seen:
        badges = ""
        for filename, pages in seen.items():
            page_label = "pp." + ", ".join(pages) if len(pages) > 1 else "p." + pages[0]
            badges += (
                f'<span style="display:inline-block; background:#1a3a5c; color:#89c4f4; '
                f'padding:3px 10px; border-radius:12px; font-size:0.78em; '
                f'border:1px solid #2d5a8e; margin:2px 4px 2px 0;">'
                f'📄 {filename}, {page_label}</span>'
            )
        st.markdown(
            f'<div style="margin-top:16px; padding:10px 14px; '
            f'border-top:1px solid #2d3a4a; border-radius:0 0 8px 8px; '
            f'background:#0f1e2e;">'
            f'<span style="font-size:0.75em; color:#5a7a9a; '
            f'letter-spacing:0.05em; text-transform:uppercase; font-weight:600;">'
            f'Sources</span>'
            f'<div style="margin-top:6px;">{badges}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


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


def _render_sidebar() -> tuple[float, bool, float, bool]:
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
        relevance_threshold = st.sidebar.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Chunks scoring below this threshold are excluded before generation. "
                 "Raise it to be stricter; lower it to allow more context.",
        )
    else:
        st.sidebar.caption("Disabled: faster retrieval-only mode")
        relevance_threshold = 0.0

    st.sidebar.header("Multi-Query Retrieval")
    use_multi_query = st.sidebar.toggle(
        "Use multi-query retrieval",
        value=False,
        help="Generate 3 query variants and merge results. Improves recall for broad "
             "synthesis questions (e.g. comparisons, overviews). Adds one extra LLM call.",
    )
    if use_multi_query:
        st.sidebar.caption("Enabled: 3 query variants — merged + deduplicated results")
    else:
        st.sidebar.caption("Disabled: single-query mode (faster)")

    st.sidebar.header("Reference Documents")
    if references:
        for index, name in enumerate(references, start=1):
            st.sidebar.write(f"{index}. {name}")
    else:
        st.sidebar.warning("No PDF files found in data/.")

    return alpha, use_reranker, relevance_threshold, use_multi_query


def main() -> None:
    st.set_page_config(page_title="Cholera RAG System", page_icon=":microscope:", layout="wide")
    st.title("Ethiopia Cholera Response RAG System")

    _init_session_state()
    alpha, use_reranker, relevance_threshold, use_multi_query = _render_sidebar()
    status = get_system_status()

    if not _api_key_available():
        st.error(
            "API key is missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env. "
            "GROQ_API_KEY can be used as fallback."
        )
        st.stop()

    if not (status["vector_ready"] and status["bm25_ready"]):
        st.error("Index is not ready. Add PDFs to data/ and run ingestion.")
        st.stop()

    try:
        resources = load_resources()
    except Exception as exc:
        st.error(f"Failed to load cached resources: {exc}")
        st.stop()

    remaining_examples = [q for q in EXAMPLE_QUESTIONS if q not in st.session_state.used_examples]
    if remaining_examples:
        st.markdown(
            '<p style="color:#5a7a9a; font-size:0.85em; margin-bottom:6px;">💡 Example questions — click to ask:</p>',
            unsafe_allow_html=True,
        )
        for q in remaining_examples:
            if st.button(q, key=f"eq_{q[:30]}", use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.used_examples.add(q)
                st.rerun()
        st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer(message["content"], docs=message.get("docs"))
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("fragments"):
                _render_fragments(message["fragments"])

    user_question = st.chat_input("Ask about cholera prevention, outbreaks, or response strategy...")

    if st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None

    if not user_question:
        return

    st.session_state.used_examples.add(user_question)

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Rewrite follow-up questions using conversation history.
            prior_history = st.session_state.messages[:-1]  # exclude the just-added user message
            search_query = rewrite_query(user_question, prior_history) if prior_history else user_question
            if search_query != user_question:
                st.caption(f"🔍 Search query: *{search_query}*")

            try:
                is_synthesis = _looks_like_synthesis_query(search_query)

                if is_synthesis:
                    # Per-document forced retrieval: top-2 chunks from each of
                    # the 10 source documents, guaranteeing full corpus coverage.
                    st.caption(
                        "📚 Synthesis mode: retrieving from all source documents"
                    )
                    doc_names = _list_reference_documents()
                    docs = get_per_document_chunks(
                        question=search_query,
                        chroma_store=resources["chroma_store"],
                        doc_names=doc_names,
                        top_k_per_doc=2,
                    )
                else:
                    # Standard path: hybrid ensemble + optional reranker.
                    ensemble = build_ensemble_retriever(
                        chroma_retriever=resources["chroma_retriever"],
                        bm25_retriever=resources["bm25_retriever"],
                        vector_weight=alpha,
                        bm25_weight=1 - alpha,
                    )

                    if use_multi_query:
                        try:
                            llm = build_llm(temperature=0.0)
                            ensemble = build_multi_query_retriever(ensemble, llm=llm)
                        except Exception:
                            pass

                    retriever = build_rerank_retriever(
                        base_retriever=ensemble,
                        reranker_provider="flashrank" if use_reranker else "none",
                        top_n=7,
                        preloaded_compressor=resources["cross_encoder"] if use_reranker else None,
                    )
                    docs = retriever.invoke(search_query)[:7]

                    if use_reranker and relevance_threshold > 0.0:
                        filtered = [
                            d for d in docs
                            if d.metadata.get("relevance_score", 1.0) >= relevance_threshold
                        ]
                        docs = filtered if filtered else docs[:1]

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
        _render_answer(answer, docs=docs)
        _render_fragments(fragments)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "fragments": fragments, "docs": docs}
    )


if __name__ == "__main__":
    main()
