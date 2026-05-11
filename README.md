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

A RAG system built to answer questions about cholera in Ethiopia using 10 research papers and clinical guidelines as the knowledge base. You ask a question, it finds the relevant passages, and Gemini 3.1 Flash-Lite writes the answer — always with citations pinned to the exact source file and page number.

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
GEMINI_API_KEY=your_gemini_key_here    # primary (pay-as-you-go, ~$0.30/1M input tokens)
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

### Why `all-MiniLM-L6-v2` for embeddings?

Practical reasons mainly: it's 80MB, runs on CPU without issues, and indexes all ~750 chunks in under a minute on a regular laptop. For a project that needs to run without a GPU, that matters.

It's not the most powerful embedding model — a biomedical-focused model would probably do better on specialized terminology. But for matching the general intent of a question to the right passages, it's good enough that the reranker handles the fine-grained sorting.

The current index holds ~835 chunks across 10 documents.

### Why hybrid retrieval + FlashRank reranking?

BM25 alone misses questions where the wording doesn't exactly match the document (e.g., "death rate" vs. "case fatality ratio"). Vector search alone misses exact terms — region names like Oromia or Shashemene, acronyms like OCV or AWD, specific numeric values. Running both and combining the results catches what either one would miss on its own.

The FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`) then re-scores the top 15 candidates by reading the query and each passage together, which is much more accurate than the initial retrieval scores. We upgraded from `TinyBERT-L-2` (2 layers) to `MiniLM-L-12` (12 layers) for meaningfully better reranking quality with acceptable latency.

### Why Gemini 3.1 Flash-Lite as the primary LLM?

Input tokens are the bottleneck for RAG workloads — each query sends 5–7 retrieved chunks (typically 3,000–6,000 tokens) to the model. Gemini 3.1 Flash-Lite costs $0.25/1M input tokens, which is roughly half the cost of Groq's Llama 3.3 70B ($0.59/1M). For a system where 90% of token spend is on input, that difference adds up quickly.

Groq (Llama 3.3 70B) is kept as an automatic fallback when no Gemini key is configured. Its lower output token price ($0.79/1M vs $1.50/1M) makes it preferable for queries that generate long answers.

### How citations work

The model is instructed to cite using `[Doc 1]`, `[Doc 2]` etc. rather than writing filenames directly. After generation, the code replaces each tag with the actual filename and page from the retrieved document metadata. This eliminates a whole class of citation errors where the model would confuse filenames or page numbers from memory.

---

## Error Analysis

### What actually failed: broad synthesis questions

**Question:**
> "Across the reviewed literature, what are the most consistently cited risk factors for cholera transmission in Ethiopia, and which interventions show the strongest evidence of effectiveness?"

The system gave a reasonable answer mentioning WASH deficits and OCV, but it basically just pulled from one or two papers rather than looking across all ten. It missed the risk factor discussion in the healthcare-seeking behaviour paper and didn't compare effect sizes between studies.

The underlying problem is that a single embedding query tends to pull chunks from whichever documents score highest — usually one or two papers. The other documents don't make it into the top results even when they contain relevant information.

A proper fix would be to break the question into smaller sub-queries (one per paper/topic), retrieve separately for each, then synthesize — but that's not implemented yet.

---

### A risk that hasn't caused a failure yet: table row misidentification

When asked *"What is the CFR for cholera in Ethiopia reported in the retrospective analysis?"*, the system returned the correct answer: **1.10% (95% CI 1.092–1.095)**. But it got lucky — that number was written out as a sentence in the abstract, so retrieval was straightforward.

The same question could go wrong if the value only appeared inside a stratified table (CFR by region, sex, age group). The character-based splitter doesn't know where tables start and end, so a table can get split with the header in one chunk and the data rows in another. If that happens, the model sees numbers without their column labels and might cite the wrong sub-group.

System prompt rules 6–8 tell the model to flag this ambiguity rather than guess. But it doesn't fully eliminate the risk — treating each table as an atomic unit during chunking would be the real fix.

---