---
title: Cholera RAG System
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Ethiopia Cholera Response RAG System

A RAG system built to answer questions about cholera in Ethiopia using 10 research papers and clinical guidelines as the knowledge base. You ask a question, it finds the relevant passages, and Gemini 2.5 Flash writes the answer — always with citations pinned to the exact source file and page number.

---

## Code Structure

```
cholera-rag-system/
├── app.py              # Streamlit UI
├── main.py             # CLI version if you prefer the terminal
├── src/
│   ├── ingestion.py    # PDF loading (PyMuPDF + pdfplumber), chunking, index building
│   ├── retriever.py    # hybrid BM25 + vector search + FlashRank reranking
│   └── generator.py    # prompt engineering, Gemini/Groq call, citation resolution
├── data/               # the 10 PDF source papers
├── vector_db/          # saved Chroma index + BM25 document store
└── requirements.txt
```

The ingestion / retrieval / generation split is intentional — each piece can be swapped out or tested on its own without touching the others.

---

## Setup

You'll need Python 3.10+ and a **Gemini API key** (free at [aistudio.google.com](https://aistudio.google.com/apikey)).
A Groq API key works as fallback — the system uses Groq if no Gemini key is found.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```
GEMINI_API_KEY=your_gemini_key_here    # primary (pay-as-you-go, ~$0.25/1M input tokens)
GROQ_API_KEY=your_groq_key_here        # fallback (1,000 req/day free)
HF_TOKEN=optional_for_gated_hf_models
```

```bash
streamlit run app.py   # opens the web UI
python main.py         # or use the terminal version
```

The first run builds the index from the PDFs in `data/` — takes a minute or two. After that it reuses the saved index so startup is fast.

---

## Design Justification

### Why 1,000-character chunks with 200-character overlap?

The papers have a mix of narrative paragraphs and dense tables, so the chunk size needed to be big enough to keep a table's header and at least a few rows together, but not so big that one chunk covers two completely different topics.

1,000 characters (roughly 150 words) turned out to be a reasonable middle ground. With 400-character chunks, sentences and table rows kept getting cut in half and the model didn't have enough context to make sense of numbers. With 2,000-character chunks, a single chunk would cover multiple topics and retrieval scores became less meaningful.

The 200-character overlap is mostly to avoid the edge case where a sentence gets split right at a chunk boundary, so nothing falls through the cracks between chunks.

### Why PyMuPDF + pdfplumber for PDF parsing?

The source documents are a mix of single-column reports and multi-column academic papers. `pdfplumber` alone handles single-column layouts well but collapses word spaces in two-column PDFs — words like "introduced at least 11 times since 1970" get extracted as "introducedatleast11timessince1970", which breaks both BM25 and vector search entirely.

PyMuPDF resolves this: it uses character-level coordinates to reconstruct word boundaries correctly, regardless of column layout. So the pipeline now uses PyMuPDF for body text extraction (accurate spacing), then pdfplumber on top to detect and format tables as pipe-delimited rows (`Region | Cases | Deaths | CFR`). Tables that span pages still get split at the page boundary, but column alignment within a page is reliable.

### Why `BAAI/bge-large-en-v1.5` for embeddings?

Practical reasons mainly: it's 80 MB, runs on CPU without issues, and indexes all ~835 chunks in under a minute on a regular laptop. For a project that needs to run without a GPU, that matters.

It's not the most powerful embedding model — a biomedical-focused model would probably do better on specialized terminology. But for matching the general intent of a question to the right passages, it's good enough that the reranker handles the fine-grained sorting.

The current index holds ~835 chunks across 10 documents.

### Why hybrid retrieval + FlashRank reranking?

BM25 alone misses questions where the wording doesn't exactly match the document (e.g., "death rate" vs. "case fatality ratio"). Vector search alone misses exact terms — region names like Oromia or Shashemene, acronyms like OCV or AWD, specific numeric values. Running both and combining the results catches what either one would miss on its own.

The FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) then re-scores the top candidates by reading the query and each passage together, which is much more accurate than the initial retrieval scores. The reranker was upgraded from `TinyBERT-L-2` (2 layers) to `MiniLM-L-12` (12 layers) for meaningfully better reranking quality with acceptable latency.

### Why Gemini 2.5 Flash as the primary LLM?

Input tokens are the bottleneck for RAG workloads — each query sends 5–7 retrieved chunks (typically 3,000–6,000 tokens) to the model. Gemini 2.5 Flash costs $0.25/1M input tokens, which is roughly half the cost of alternatives. For a system where 90% of token spend is on input, that difference adds up quickly.

Groq (Llama 3.3 70B) is kept as an automatic fallback when no Gemini key is configured. Its lower output token price makes it preferable for queries that generate long answers.

### How citations work

The model is instructed to cite using `[Doc 1]`, `[Doc 2]` etc. rather than writing filenames directly. After generation, the code replaces each tag with the actual filename and page from the retrieved document metadata. This eliminates a whole class of citation errors where the model would confuse filenames or page numbers from memory.

Chunks from the same source file are assigned the same `[Doc N]` label in the context window, so the LLM never describes two chunks from one paper as being from different documents.

### How context ordering works

Before passing retrieved chunks to the LLM, `format_context` sorts them by a corpus-agnostic relevance signal:

- Chunks containing tabular or numerical markers (`[TABLE]`, `%`, `total`) get a bonus when the question asks for a quantitative value.
- Query–document token overlap (shared words of 4+ characters) adds to the score.

This replaces an earlier version that used hardcoded domain keywords (`shashemene`, `age group`, `campaign`, etc.), which was not generalisable across different corpora.

### How synthesis questions are handled

When a question is detected as requiring cross-document synthesis (keywords: `compare`, `across`, `all papers`, `overview`, `summarize`, etc.), the retrieval path switches from the standard ensemble to a **per-document forced retrieval**: the system queries the vector index separately for each of the 10 source documents and collects the top-2 most relevant chunks from each. This guarantees that all documents contribute to the context, preventing answers from being drawn from only one or two papers.

For standard factual questions, the normal hybrid ensemble + FlashRank path is used unchanged.

### How follow-up questions are handled

If a follow-up question is detected (short query, pronouns like "it" or "they", missing explicit subject), the system rewrites it into a standalone query using the conversation history before running retrieval. This prevents follow-ups like "How effective was it?" from returning irrelevant results.

---

## Error Analysis

### Broad synthesis questions — partially addressed

**Original problem:**
> "Across the reviewed literature, what are the most consistently cited risk factors for cholera transmission in Ethiopia?"

Previously the system answered using only one or two of the ten papers, missing evidence distributed across the collection.

**Current behaviour:**
Synthesis questions now trigger per-document retrieval, which collects the top-2 chunks from each of the 10 documents before passing them to the LLM. In practice this means all 10 source papers appear in the Sources section. The quality of the final synthesis still depends on how well the LLM can combine 20 chunks of varying relevance — a MapReduce approach (summarise each paper separately, then synthesise) would be more accurate but costs 10× more LLM calls.

---

### A risk that hasn't caused a failure yet: table row misidentification

When asked *"What is the CFR for cholera in Ethiopia reported in the retrospective analysis?"*, the system returns the correct answer. But the same question could go wrong if the value only appeared inside a stratified table (CFR by region, sex, age group). The character-based splitter doesn't know where tables start and end, so a table can get split with the header in one chunk and the data rows in another. If that happens, the model sees numbers without their column labels and might cite the wrong sub-group.

System prompt rules tell the model to flag this ambiguity rather than guess. But it doesn't fully eliminate the risk — treating each table as an atomic unit during chunking would be the real fix.

---
