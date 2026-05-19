"""Answer generation module using Gemini 2.5 Flash."""

from __future__ import annotations

import os
import re
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

load_dotenv()


SYSTEM_PROMPT = """
You are a question-answering assistant specialized in cholera literature.
Follow these rules strictly:
1) Answer only based on the provided context.
2) If the context does not contain enough evidence, say you do not know.
3) Answer in English.
4) At the end of your answer, cite the documents you used with ONLY their document numbers as shown
   in the context headers (e.g., [Doc 1], [Doc 3]). Do NOT write filenames or page numbers yourself —
   they will be filled in automatically. Use only the [Doc N] tags that appear in the context.
5) Always include specific numerical values (percentages, counts, rates, etc.) when they are present
   in the context. Never substitute a vague description (e.g., "high coverage") for an actual number
   that exists in the retrieved text. If multiple figures are available (e.g., by subgroup, region,
   or round), list them all explicitly.
6) Critical Instruction for Table Reading: When extracting numerical data from tables, you must strictly distinguish between 'Total' values and sub-category values (e.g., Sex, Age group, Region). Always double-check the row and column headers to ensure the cited value represents the entire population unless specified otherwise.
7) For any numeric answer from a table, validate all of the following before answering:
   - which row label the value belongs to (e.g., Total vs Male),
   - which column label defines the metric (e.g., CFR vs cases),
   - whether the value is overall population or subgroup-specific.
8) If table structure is ambiguous or headers are incomplete in the retrieved chunk, do not guess. State uncertainty and cite the source.
9) The answer may require data from tables or sections listing numerical values. Carefully analyze not only narrative sentences but also any structured data sections before responding.
10) Chain-of-Verification for Numbers: Before stating ANY numerical value (percentage, count,
    rate, ratio), first quote the EXACT sentence or table cell from the context where you found
    it, using this format:
      > "...exact quote..." [Doc N]
    Then state your interpretation on the next line. Apply this to every number in your answer.
11) If you cannot locate a direct quote in the context that supports a specific number, do not
    state that number. Instead write: "The context mentions [topic] but I cannot confirm the
    exact figure."
12) When referring to a source in your prose, use the short title shown in the context header
    (e.g., "the Retrospective Analysis study" or "the OCV review"), NOT generic labels like
    "Document 1" or "Source 1". Reserve [Doc N] tags only for inline citations at the end of
    your answer.
13) When categorizing factors or findings across multiple categories (e.g., environmental vs.
    behavioral risk factors), each item must appear in exactly ONE category — the most
    appropriate one. Do not repeat the same item under multiple headings.
14) Geographic context is mandatory for every cited result. For each numerical finding or
    study conclusion, state the country, region, or city where the study was conducted
    (e.g., "in a trial conducted in the Philippines", "from a field deployment in Odisha, India",
    "reported for Ethiopia nationally"). Never present results from non-Ethiopian studies as if
    they directly apply to Ethiopia unless the source paper explicitly states so.
15) Distinguish vaccine efficacy from vaccine effectiveness. Efficacy figures come from
    controlled trials (RCTs) under ideal conditions; effectiveness figures come from real-world
    field deployments or observational studies. Label every cited VE figure with its study
    design: write "efficacy (RCT)" or "effectiveness (field study)" immediately after the value.
    Do not average, combine, or present efficacy and effectiveness figures under the same label.
""".strip()


_SYNTHESIS_RE = re.compile(
    r"\b(compare|comparison|across|overall|all papers|summary|summarize|synthesize|"
    r"between|difference|similar|trend|pattern|throughout|multiple|various|"
    r"each study|every study|literature|review|overview|consistently|most|which papers|"
    r"which documents|what are the|common|frequent)\b",
    re.IGNORECASE,
)

SYNTHESIS_PROMPT_SUFFIX = """

Additional instructions for this synthesis question:
- For each factor or finding you list, specify the study context where it appears: include the
  study type (e.g., retrospective analysis, RCT, review), location (region/city in Ethiopia),
  and approximate year if available in the context.
- Indicate whether a factor is mentioned by multiple papers or only one.
- Each factor must appear under exactly one category — do not repeat the same factor across
  multiple lists.
- Prefer citing the actual short paper title (from the context header) over generic "Document N"
  labels when referring to a source in prose.
"""


def _is_synthesis_query(question: str) -> bool:
    return bool(_SYNTHESIS_RE.search(question))


def get_gemini_api_key() -> str:
    """Load Gemini API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment."
        )
    return api_key


def build_llm(temperature: float = 0.0):
    """Return the best available LLM: Gemini first, then Groq fallback.

    Priority: GEMINI_API_KEY / GOOGLE_API_KEY → GROQ_API_KEY
    seed=42 is set for Groq to maximise reproducibility across identical queries.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        return ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL,
            google_api_key=gemini_key,
            temperature=temperature,
        )

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=DEFAULT_GROQ_MODEL,
            api_key=groq_key,
            temperature=temperature,
            seed=42,
        )

    raise ValueError(
        "No LLM API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
        "or set GROQ_API_KEY as fallback."
    )


def _short_title(filename: str) -> str:
    """Extract a brief human-readable label from a PDF filename for use in prose."""
    name = filename
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = re.sub(r"^\([^)]+\)\s*", "", name)  # strip leading "(IVI)", "(WHO)", etc.
    name = name.replace("_", " ").strip()
    if len(name) > 60:
        name = name[:60].rsplit(" ", 1)[0]
    return name


def _score_context_document(doc: Document, question: str) -> int:
    """Prioritise chunks that are most likely to contain the answer.

    Two corpus-agnostic signals:
    1. Structured-data boost: chunks containing tabular/numerical markers get a
       bonus when the question is asking for a quantitative value.
    2. Query-token overlap: long tokens (>=4 chars) shared between the question
       and the chunk text are counted as a relevance signal.
    """
    text = doc.page_content.lower()
    q = question.lower()
    score = 0

    # Boost chunks that look like tables/numbers when the question asks for figures
    numerical_question = any(
        w in q for w in ("how many", "what is the", "rate", "percent",
                         "number", "count", "total", "proportion", "ratio")
    )
    if numerical_question and any(m in text for m in ("[table]", "%", "total")):
        score += 3

    # General token overlap (domain-agnostic)
    q_tokens = set(re.findall(r"\b\w{4,}\b", q))
    text_tokens = set(re.findall(r"\b\w{4,}\b", text))
    score += len(q_tokens & text_tokens)

    return score


def format_context(documents: Iterable[Document], question: str) -> str:
    """Format retrieved chunks into a prompt-ready context block.

    Chunks from the same source file share the same [Doc N] label so the LLM
    never confuses two chunks from one paper as being from different sources.
    """
    doc_list = list(documents)
    doc_list.sort(key=lambda doc: _score_context_document(doc, question), reverse=True)

    # Assign a stable doc number per unique filename
    seen: dict[str, int] = {}
    next_idx = 1
    sections: list[str] = []
    for doc in doc_list:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
        if filename not in seen:
            seen[filename] = next_idx
            next_idx += 1
        idx = seen[filename]
        sections.append(
            f"[Document {idx}] Short title: {_short_title(filename)} | Page: {page}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(sections)


def build_gemini_chain(model_name: str = DEFAULT_GEMINI_MODEL, is_synthesis: bool = False):
    """Create LLM chat chain with a strict RAG prompt (Groq or Gemini)."""
    llm = build_llm(temperature=0.0)
    human_message = (
        "Question:\n{question}\n\n"
        "Context:\n{context}\n\n"
        "Please provide a concise and accurate answer in English.\n"
        "At the end, cite only using [Doc N] tags (e.g., [Doc 1][Doc 2]). Do not write filenames."
    )
    if is_synthesis:
        human_message += SYNTHESIS_PROMPT_SUFFIX
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", human_message),
        ]
    )
    return prompt | llm


def invoke_with_google_genai_sdk(
    question: str,
    context: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
    is_synthesis: bool = False,
) -> str:
    """Fallback path for direct Google GenAI SDK calls."""
    from google import genai

    api_key = get_gemini_api_key()
    client = genai.Client(api_key=api_key)
    human_suffix = SYNTHESIS_PROMPT_SUFFIX if is_synthesis else ""
    composed_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Please provide a concise and accurate answer in English.\n"
        "At the end, cite only using [Doc N] tags (e.g., [Doc 1][Doc 2]). Do not write filenames."
        f"{human_suffix}"
    )
    response = client.models.generate_content(
        model=model_name,
        contents=composed_prompt,
    )
    return (response.text or "").strip()


_FOLLOWUP_SIGNALS = re.compile(
    r"\b(it|its|they|their|them|that|this|those|these|there|"
    r"the same|the campaign|the region|the study|the paper|"
    r"the result|the data|the rate|the number|such|aforementioned)\b",
    re.IGNORECASE,
)


def _looks_like_followup(question: str) -> bool:
    """Cheap heuristic: return True if the question likely needs prior context."""
    q = question.strip()
    if len(q) < 60:
        return True
    if _FOLLOWUP_SIGNALS.search(q):
        return True
    return False


def rewrite_query(
    question: str,
    conversation_history: list[dict[str, str]],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Rewrite a follow-up question as a standalone search query using conversation context.

    Returns the original question unchanged when there is no prior history,
    when the question looks self-contained, or when the rewrite call fails.
    """
    if not conversation_history:
        return question

    if not _looks_like_followup(question):
        return question

    history_lines: list[str] = []
    for msg in conversation_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        snippet = msg["content"][:400].replace("\n", " ")
        history_lines.append(f"{role}: {snippet}")
    history_text = "\n".join(history_lines)

    try:
        llm = build_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a search query rewriter. "
                    "Given a conversation history and a follow-up question, "
                    "rewrite the follow-up question as a complete, self-contained search query "
                    "that includes all necessary context from the conversation. "
                    "If the follow-up question is already self-contained, return it as-is. "
                    "Output only the rewritten query — no explanation, no quotes.",
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\nFollow-up question: {question}\n\nRewritten query:",
                ),
            ]
        )
        chain = prompt | llm
        response = chain.invoke({"history": history_text, "question": question})
        raw = response.content if hasattr(response, "content") else response
        if isinstance(raw, list):
            rewritten = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            ).strip()
        else:
            rewritten = str(raw).strip()
        return rewritten or question
    except Exception:
        return question


def _build_doc_map(documents: list[Document], question: str) -> dict[int, tuple[str, str]]:
    """Return {doc_index: (filename, page)} in the same order as format_context."""
    doc_list = list(documents)
    doc_list.sort(key=lambda d: _score_context_document(d, question), reverse=True)
    result: dict[int, tuple[str, str]] = {}
    for idx, doc in enumerate(doc_list, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = str(doc.metadata.get("page_number", doc.metadata.get("page", "N/A")))
        result[idx] = (filename, page)
    return result


def _resolve_citations(answer: str, doc_map: dict[int, tuple[str, str]]) -> str:
    """Replace [Doc N] tags with verified [Source: filename, p.page] citations."""
    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if n in doc_map:
            filename, page = doc_map[n]
            return f"[Source: {filename}, p.{page}]"
        return match.group(0)

    return re.sub(r"\[Doc\s*(\d+)\]", _replace, answer)


def generate_answer(
    question: str,
    retrieved_docs: list[Document],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate an answer grounded in retrieved context."""
    if not retrieved_docs:
        return "No relevant context was retrieved. [Source: none]"

    context = format_context(retrieved_docs, question=question)
    doc_map = _build_doc_map(retrieved_docs, question=question)
    is_synthesis = _is_synthesis_query(question)

    try:
        chain = build_gemini_chain(model_name=model_name, is_synthesis=is_synthesis)
        response = chain.invoke({"question": question, "context": context})
        raw = response.content if hasattr(response, "content") else response
        if isinstance(raw, list):
            answer = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            ).strip()
        else:
            answer = str(raw).strip()
    except Exception:
        answer = invoke_with_google_genai_sdk(
            question=question,
            context=context,
            model_name=model_name,
            is_synthesis=is_synthesis,
        )

    answer = _resolve_citations(answer, doc_map)

    if "[Source:" not in answer:
        first = retrieved_docs[0]
        filename = first.metadata.get("filename", "unknown")
        page = str(first.metadata.get("page_number", first.metadata.get("page", "N/A")))
        answer = f"{answer.strip()}\n\n[Source: {filename}, p.{page}]"

    return answer.strip()
