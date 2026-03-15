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
- Ensures that important information at the end of one chunk is also present at the start of the next, so context isn’t lost if a relevant answer crosses chunk boundaries.
- In interviews, you can say: “Overlap helps preserve context for the model. If a key sentence or fact is split between two chunks, overlap makes sure both chunks contain the full information, improving retrieval accuracy and reducing the chance of missing evidence.”
- This is especially important for documents where ideas or facts often run across paragraph or section breaks.

How to explain in interviews:

- “I experimented with different chunk sizes and overlaps to find what works best for my documents. For example, I started with a chunk size of 1000 characters and 150 overlap, then checked if the retrieved chunks actually contained the information needed to answer typical questions. If answers were missing or context was cut off, I adjusted the chunk size or overlap. Overlap is important because it helps preserve meaning when information spans across chunk boundaries, so the model doesn’t lose context. I also considered the structure of my documents—if they have lots of short sections, I might use smaller chunks; for longer, continuous text, larger chunks with overlap work better. I validated my choices by running test queries and seeing if the retrieval step returned relevant, complete information.”

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

<!--
This section highlights a crucial aspect of working with embedding-based retrieval systems: consistency in the embedding model. When building a vector index (for example, for semantic search or retrieval-augmented generation), the embeddings generated from your data at ingestion time must use the same model as the one used to encode queries at search time. 

If you change the embedding model after building the index, the vector representations of your data and your queries will no longer be compatible, leading to poor or incorrect retrieval results. Therefore, any change in the embedding model requires you to re-embed your data and rebuild the index to maintain retrieval accuracy.

For interviews, be prepared to explain:
- Why embedding model consistency is important (vector space alignment).
- What could go wrong if different models are used (semantic mismatch, poor recall/precision).
- The process of re-indexing when updating or switching embedding models.
-->
<!--
This section highlights a critical point for retrieval-augmented generation (RAG) systems: the embedding model used to convert text into vector representations must remain consistent between the data ingestion phase (when documents are indexed) and the query phase (when user queries are processed). Changing the embedding model after indexing requires rebuilding the index to ensure accurate similarity search and retrieval.

Popular embedding models include:
- OpenAI's `text-embedding-ada-002`
- Sentence Transformers (e.g., `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L12-v2`)
- Cohere's embedding models
- Google's Universal Sentence Encoder (USE)
- Hugging Face Transformers models (e.g., `distilbert-base-nli-stsb-mean-tokens`)

Choosing a widely adopted and well-supported embedding model is recommended for compatibility and performance.
-->
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

### 6.1 Querying the Vector Database: Step-by-Step

When you query the vector database (Chroma), the process is as follows:

**1. Embed the Question**
- The user’s question is converted into an embedding (a vector) using the *same* embedding model as used during ingestion.
- This ensures the question and document chunks are represented in the same vector space.

**2. Similarity Search**
- The vector database (Chroma) performs a similarity search between the question embedding and all stored chunk embeddings.
- It uses a distance metric (typically cosine similarity or Euclidean distance) to find the most similar vectors.
- The top-$k$ most similar chunks are selected (where $k$ is configurable, e.g., `RAG_TOP_K=4`).

**3. Retrieve Document Chunks**
- The database returns the top-$k$ `Document` objects, each containing:
  - The chunk’s text (`page_content`)
  - Metadata (such as `source`, `page`, etc.)
- These chunks are the most relevant pieces of your ingested documents for the given question.

**4. Construct the Prompt**
- The retrieved chunks are combined with the system instruction and the user’s question to form the prompt for the LLM.
- Example prompt structure:
  ```
  [System instruction]
  [User question]
  [Retrieved context: chunk 1, chunk 2, ...]
  ```

**5. Generate the Answer**
- The LLM receives the prompt and generates an answer, ideally using only the retrieved context.
- Optionally, the system can display the sources of the retrieved chunks for transparency.

---

#### Types of Queries Supported

There are two main ways to query the vector database in this project:

1. **Direct Question Query (`rag query "...")`**
   - You provide a single question.
   - The system retrieves relevant chunks and generates an answer.

2. **Conversational Query (`rag chat`)**
   - Supports multi-turn conversations.
   - Maintains chat history and context across turns.
   - Each user message is embedded and processed as above, but the prompt may include previous Q&A pairs for continuity.

**Advanced Query Types (optional, for interviews):**
- **Hybrid Search:** Combine vector similarity with keyword or metadata filters (e.g., only retrieve from certain sources or dates).
- **Filtered Retrieval:** Use metadata filters (e.g., only retrieve chunks from a specific document or section).
- **Re-ranking:** Retrieve more than $k$ chunks, then use another model to re-rank for relevance.

---

#### Interview-Ready Explanation

- “At query time, I embed the user’s question and perform a similarity search in the vector database to retrieve the top-$k$ most relevant document chunks. These chunks, along with the question and a strict system instruction, are used to construct the prompt for the LLM. The LLM then generates an answer grounded in the retrieved context. I can query the system either with single questions or in a conversational mode, and I can also filter or re-rank results for more advanced use cases.”


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
