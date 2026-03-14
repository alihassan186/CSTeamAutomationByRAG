# LangChain RAG Project (Interview Learning)

This is a complete, runnable Retrieval-Augmented Generation (RAG) starter project built with **LangChain**.

It includes:
- Document ingestion (load → split → embed)
- Persistent vector store (Chroma)
- A CLI to query your knowledge base (single question or interactive chat)
- Provider switch: **OpenAI** or **local Ollama**

## 1) Setup

### Option A (recommended): free hosted — Hugging Face Inference API

1. Create a free Hugging Face account: https://huggingface.co
2. Create an access token (read): https://huggingface.co/settings/tokens
3. Copy the env template and paste your token:

```bash
cp .env.example .env
```

Then edit `.env` and set `HUGGINGFACEHUB_API_TOKEN=...`

### Python env

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

## 2) Add documents

Put your files into `data/raw/`.

Supported:
- `.txt`
- `.md`
- `.pdf`

There is already a small example doc in `data/raw/example.md`.

## 3) Ingest (build the vector index)

```bash
python -m rag_project ingest
```

To rebuild from scratch:

```bash
python -m rag_project ingest --reset
```

## 4) Ask questions

Single question:

```bash
python -m rag_project query "What is RAG and why use it?"
```

Interactive chat:

```bash
python -m rag_project chat
```

Show sources:

```bash
python -m rag_project query "Summarize the example doc" --show-sources
```

## 5) Switch to OpenAI

In `.env`:

```env
RAG_PROVIDER=openai
OPENAI_API_KEY=...
```

Then re-ingest:

```bash
python -m rag_project ingest --reset
```

## Optional: Switch to Ollama (local)

In `.env`:

```env
RAG_PROVIDER=ollama
```

Then install/start Ollama (https://ollama.com) and pull models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## How this project is structured

- `src/rag_project/settings.py`: env-based configuration
- `src/rag_project/ingest.py`: load/split/index documents
- `src/rag_project/rag.py`: build the retriever + RAG chain
- `src/rag_project/cli.py`: CLI entrypoints

For a detailed interview-focused walkthrough, see [docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md).

## Interview notes (what to explain)

When asked to “explain your RAG pipeline”, keep it crisp:

1. **Ingestion**: load docs → chunk them (e.g. 800–1200 chars with overlap) → compute embeddings → store in vector DB.
2. **Retrieval**: at query time, embed question → similarity search → top-$k$ chunks.
3. **Generation**: give LLM the question + retrieved context → answer grounded in the context.
4. **Evaluation**: measure retrieval quality (recall / relevance), hallucinations, latency, and cost.

If you want, tell me which interview role (ML Eng vs Backend vs Data) and I can tailor a practice script + common questions.

