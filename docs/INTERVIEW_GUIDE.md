# Interview Guide — End-to-End RAG (LangChain)

This document explains **exactly how data flows** through this project, which files are responsible for what, and what you should highlight in interviews.

It is written to help you answer:
- “Explain your RAG pipeline end-to-end.”
- “How does retrieval work?”
- “How do you prevent hallucinations?”
- “What are the trade-offs in chunking and $k$?”
- “How would you evaluate / improve it?”

---

## 0) The one-sentence summary (what you should say first)

This project builds a **persistent vector index** from your documents (ingestion), then at query time it **retrieves the top-$k$ most similar chunks** and asks an LLM to answer using **only that retrieved context**.

---

## 1) What is RAG? (interview-ready)

**Retrieval-Augmented Generation** is a pattern:
1. Convert documents into an index optimized for search (usually via embeddings).
2. For a question, retrieve relevant text from the index.
3. Give that retrieved text to an LLM as context so it can answer grounded in your data.

Why it exists:
- LLMs don’t “know” your private documents.
- LLMs can hallucinate; retrieval provides grounding.
- It’s cheaper and more controllable than fine-tuning for many knowledge tasks.

---

## 2) Two phases: ingestion vs query

### A) Ingestion (offline)

Goal: turn raw files into a searchable vector index.

Pipeline:
- Load documents from `data/raw/`
- Split into chunks
- Embed chunks (vectorize)
- Store vectors + text + metadata in Chroma (persisted to disk)

Command:
- `rag ingest`

Key files:
- docs loading: `src/rag_project/loaders.py`
- splitting + storing: `src/rag_project/ingest.py`
- embeddings + vector store: `src/rag_project/rag.py`

### B) Query (online)

Goal: answer a question using retrieved chunks.

Pipeline:
- Embed the question
- Similarity search to fetch top-$k$ chunks
- Construct a prompt with: system instruction + question + retrieved context
- Generate an answer

Commands:
- `rag query "..."`
- `rag chat`

Key files:
- RAG chain assembly: `src/rag_project/rag.py`
- CLI: `src/rag_project/cli.py`

---

## 3) Data flow: where the “data comes from”

### 3.1 Raw data location

Documents live in:
- `data/raw/`

Supported formats in this project:
- `.md`, `.txt`, `.pdf`

Loader logic:
- `iter_files(root)` finds all supported files recursively.
- `load_file(path)` chooses the right loader based on file extension.
- `load_documents(raw_dir)` returns a list of LangChain `Document` objects.

Important interview point:
- Each `Document` includes **page_content** (text) and **metadata** (like `source`, `page`). Metadata is how you later cite sources.

### 3.2 Vector store persistence

The vector database is persisted to:
- `data/chroma/`

Meaning:
- After you ingest once, querying reuses the saved index.
- If you change embedding model or chunking, you should re-ingest (`rag ingest --reset`).

---

## 4) Chunking (one of the most important interview topics)

Chunking happens in `src/rag_project/ingest.py` using `RecursiveCharacterTextSplitter`:
- `chunk_size=1000`
- `chunk_overlap=150`

Why chunking is necessary:
- Vector search works best on smaller text pieces.
- LLM context windows are limited.

Trade-offs:
- Bigger chunks:
  - Pros: more context per chunk, fewer boundary issues
  - Cons: less precise retrieval; more irrelevant text in context
- Smaller chunks:
  - Pros: more precise retrieval
  - Cons: can miss cross-sentence meaning; more chunks to store

Why overlap exists:
- Prevents losing meaning when the “best” answer spans a boundary.

How to explain in interviews:
- “I tuned chunk size and overlap based on document structure and retrieval quality. I use overlap to preserve continuity across chunk boundaries.”

---

## 5) Embeddings (how retrieval actually works)

Embeddings are numeric vectors representing semantic meaning.

This project supports multiple providers. For learning (free hosted), default is:
- `RAG_PROVIDER=huggingface`

### 5.1 Embedding provider selection

The settings are loaded from environment variables in:
- `src/rag_project/settings.py`

Key env vars:
- `RAG_PROVIDER` (defaults to `huggingface`)
- `HF_EMBEDDING_MODEL`
- `HUGGINGFACEHUB_API_TOKEN`

### 5.2 What happens during embedding

During ingestion:
1. For each chunk’s text, the embedding model produces a vector.
2. Chroma stores that vector + chunk text + metadata.

During querying:
1. The question is embedded into the same vector space.
2. Chroma returns the nearest chunk vectors (similarity search).

Critical interview point:
- You must use the **same embedding model** at ingestion and query time. If you change it, you must rebuild the index.

---

## 6) Vector database (Chroma) and retrieval

This project uses:
- `langchain_chroma.Chroma`

It stores:
- vectors (embeddings)
- documents (chunk text)
- metadata (`source`, `page`, etc.)

Retrieval steps:
1. Convert question → embedding
2. Similarity search for top-$k$
3. Return `Document` chunks

Important knob:
- `RAG_TOP_K` (defaults to 4)

Trade-offs of $k$:
- Higher $k$ improves recall but may add noise and increase prompt length.
- Lower $k$ is faster but may miss key evidence.

---

## 7) Prompting and hallucination control

The “grounding” instruction is built in `src/rag_project/rag.py`:

System rule (conceptually):
- “Answer using ONLY the provided context. If insufficient, say you don’t know.”

Why this matters:
- The retriever can return irrelevant chunks.
- The LLM can hallucinate if prompted loosely.

What this project does:
- Keeps a strict system instruction.
- Includes only retrieved context in the prompt.
- Optionally returns sources (`--show-sources`) so answers can be validated.

Interview-ready line:
- “I constrain the model to answer only from retrieved context and I surface sources so users can verify the grounding.”

---

## 8) Free hosted provider: Hugging Face Inference API

This project uses Hugging Face Inference for:
- Embeddings (feature extraction)
- Text generation

Implementation:
- `src/rag_project/hf_inference.py`

Key setup requirement:
- You must set `HUGGINGFACEHUB_API_TOKEN` in `.env` or you will get `401 Unauthorized`.

Interview caveat:
- Free tiers can have rate limits / cold starts. In production you’d plan for caching, retries, and possibly a dedicated endpoint.

---

## 9) What each important file does

### `src/rag_project/cli.py`
- Defines CLI commands: `ingest`, `query`, `chat`
- Loads `.env`
- Calls the ingestion function and the RAG chain

### `src/rag_project/settings.py`
- Reads environment variables using Pydantic Settings
- Central place for provider choice and model names

### `src/rag_project/loaders.py`
- Finds files in `data/raw/`
- Loads `.pdf` / `.md` / `.txt`
- Adds metadata like `source`

### `src/rag_project/ingest.py`
- Loads documents
- Splits into chunks
- Creates vector store and adds chunks
- Resets index if `--reset` is passed

### `src/rag_project/rag.py`
- Creates embeddings
- Creates vector store/retriever
- Builds the prompt
- Creates a runnable chain (retrieve → format → prompt → LLM → string)
- Optional sources output

### `src/rag_project/hf_inference.py`
- Hugging Face inference wrapper:
  - `HFInferenceEmbeddings` for embeddings
  - `hf_generate()` for text generation

---

## 10) Common interview questions and strong answers

### Q1: “How does your RAG system work?”
Answer:
- “I ingest documents by chunking them, embedding each chunk, and persisting them in a vector database. At query time I embed the question, retrieve the top-k similar chunks, and prompt the LLM to answer strictly from that retrieved context.”

### Q2: “How do you reduce hallucinations?”
Answer:
- “I constrain the prompt to only use retrieved context, I surface sources, and I tune retrieval so the model sees high-quality evidence. If evidence is missing, I instruct the model to say it doesn’t know.”

### Q3: “How do you choose chunk size and overlap?”
Answer:
- “I start with a reasonable chunk size like ~800–1200 chars and add overlap to preserve continuity. Then I evaluate retrieval relevance and adjust based on doc type and question patterns.”

### Q4: “How do you evaluate RAG quality?”
Answer:
- “I evaluate retrieval (are the top-k chunks relevant?), grounding (does the answer match the retrieved text?), and end-to-end latency and cost. I also do manual spot-checks and can build a small labeled set of Q/A pairs for regression testing.”

### Q5: “What are typical failure modes?”
Answer:
- “Bad chunking, weak embeddings, noisy documents, or top-k too low/high causing missing evidence or too much noise. Also prompt too permissive can cause hallucinations.”

---

## 11) Practical debugging checklist (what you do when it fails)

1. **Ingest fails with 401**
   - Token missing: set `HUGGINGFACEHUB_API_TOKEN` in `.env`.
2. **Answers are wrong**
   - Inspect sources (`--show-sources`) to see if retrieval was relevant.
   - Increase `RAG_TOP_K` slightly.
   - Adjust chunk size/overlap.
3. **Slow responses**
   - Reduce $k$, reduce chunk size, or cache embeddings.
   - Consider batching embeddings in ingestion (advanced improvement).

---

## 12) How to present this project in 60 seconds

Script:
- “I built a LangChain-based RAG system. During ingestion it loads PDFs/Markdown, splits into overlapping chunks, generates embeddings, and stores them in a persistent Chroma vector database. At query time it embeds the user question, retrieves the top-k chunks, and prompts the LLM to answer strictly from that context. I can show sources to validate grounding and I tune chunking and top-k for retrieval quality and latency.”
