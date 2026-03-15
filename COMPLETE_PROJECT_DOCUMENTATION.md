# Complete RAG Project Documentation for Interview Preparation

## Project Overview

This is a complete, runnable **Retrieval-Augmented Generation (RAG)** starter project built with **LangChain**. The project demonstrates end-to-end RAG implementation with multiple LLM providers, persistent vector storage, and a CLI interface for document ingestion and querying.

**Key Features:**
- Document ingestion pipeline (load → split → embed → store)
- Persistent vector database using Chroma
- CLI interface for single questions and interactive chat
- Multi-provider support: Hugging Face (free hosted), OpenAI (paid), Ollama (local)
- Source attribution and hallucination control
- Production-ready configuration management

## Project Structure

```
RAGProject/
├── pyproject.toml          # Project configuration and dependencies
├── requirements.txt        # Python dependencies
├── README.md              # User documentation
├── .env.example           # Environment variables template
├── data/
│   ├── raw/               # Raw documents (.md, .txt, .pdf)
│   └── chroma/            # Persistent vector database
├── docs/
│   └── INTERVIEW_GUIDE.md # Detailed interview preparation guide
└── src/rag_project/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py             # Command-line interface
    ├── hf_inference.py    # Hugging Face API integration
    ├── ingest.py          # Document ingestion pipeline
    ├── loaders.py         # Document loading utilities
    ├── rag.py             # Core RAG chain implementation
    └── settings.py        # Configuration management
```

## Technical Architecture

### 1. Configuration Management (`settings.py`)

The project uses **Pydantic Settings** for robust configuration management with environment variable support.

**Key Configuration Options:**
- **Provider Selection:** `RAG_PROVIDER` (huggingface/ollama/openai)
- **Data Directories:** Raw documents and vector store locations
- **Retrieval Parameters:** Top-k results, collection name
- **Model Configurations:** Separate settings for each provider

**Environment Variables:**
```bash
# Provider
RAG_PROVIDER=huggingface

# Data paths
RAG_RAW_DIR=data/raw
RAG_CHROMA_DIR=data/chroma
RAG_COLLECTION=rag_docs
RAG_TOP_K=4

# Hugging Face (default provider)
HUGGINGFACEHUB_API_TOKEN=your_token
HF_MODEL=HuggingFaceH4/zephyr-7b-beta
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# OpenAI
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 2. Document Loading (`loaders.py`)

**Supported Formats:**
- Markdown (.md)
- Plain text (.txt)
- PDF documents (.pdf)

**Loading Process:**
1. **File Discovery:** Recursively scans `data/raw/` for supported files
2. **Format-Specific Loading:** Uses appropriate LangChain loaders
3. **Metadata Enrichment:** Adds source file path and page information
4. **Document Objects:** Returns list of LangChain `Document` objects

**Key Functions:**
- `iter_files(root)`: Finds all supported files recursively
- `load_file(path)`: Loads single file based on extension
- `load_documents(raw_dir)`: Loads all documents with metadata

### 3. Document Ingestion (`ingest.py`)

**Ingestion Pipeline:**
1. **Load Documents:** From `data/raw/` directory
2. **Text Splitting:** Chunk documents with overlap
3. **Embedding Generation:** Convert chunks to vectors
4. **Vector Storage:** Persist in Chroma database

**Chunking Strategy:**
- **Chunk Size:** 1000 characters
- **Overlap:** 150 characters
- **Splitter:** `RecursiveCharacterTextSplitter`

**Why Chunking Matters:**
- Vector search works better on smaller text pieces
- LLM context windows have limits
- Overlap preserves meaning across boundaries

### 4. RAG Chain Implementation (`rag.py`)

**Core Components:**

#### LLM Selection
```python
def get_llm(settings: Settings):
    if settings.rag_provider == "huggingface":
        return RunnableLambda(lambda prompt: hf_generate(...))
    elif settings.rag_provider == "openai":
        return ChatOpenAI(model=settings.openai_model)
    else:  # ollama
        return ChatOllama(model=settings.ollama_model)
```

#### Embeddings Selection
```python
def get_embeddings(settings: Settings) -> Embeddings:
    if settings.rag_provider == "huggingface":
        return HFInferenceEmbeddings(...)
    elif settings.rag_provider == "openai":
        return OpenAIEmbeddings(...)
    else:  # ollama
        return OllamaEmbeddings(...)
```

#### Vector Store
- **Database:** Chroma (persistent, file-based)
- **Storage:** Vectors + text + metadata
- **Retrieval:** Similarity search with configurable top-k

#### Prompt Engineering
**System Instruction:**
```
"You are a helpful assistant. Answer using ONLY the provided context. 
If the context is insufficient, say you don't know and ask for the missing info."
```

**Prompt Template:**
```
System: {system_instruction}

Question: {question}

Context:
{context}

Answer:
```

#### Chain Construction
```python
def build_rag_chain(settings: Settings, *, show_sources: bool = False):
    embeddings = get_embeddings(settings)
    vectorstore = get_vectorstore(settings, embeddings, create_if_missing=False)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.rag_top_k})
    llm = get_llm(settings)
    
    # Retrieval → Format → Prompt → LLM → Parse
    chain = ({"question": RunnablePassthrough(), "context": context_runnable} 
             | prompt | llm | StrOutputParser())
    
    return chain
```

### 5. Hugging Face Integration (`hf_inference.py`)

**Custom Embeddings Class:**
```python
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, *, token: str, model: str):
        self._client = InferenceClient(token=token)
        self._model = model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch embedding with mean pooling for token embeddings
        return [self._to_vector(self._client.feature_extraction(text, model=self._model)) 
                for text in texts]
    
    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(self._client.feature_extraction(text, model=self._model))
```

**Text Generation:**
```python
def hf_generate(*, token: str, model: str, prompt: str, 
                max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    client = InferenceClient(token=token)
    return client.text_generation(prompt, model=model, 
                                  max_new_tokens=max_new_tokens, 
                                  temperature=temperature, 
                                  return_full_text=False).strip()
```

### 6. CLI Interface (`cli.py`)

**Commands:**
- `rag ingest [--reset]`: Build/update the vector index
- `rag query "question" [--show-sources]`: Single question
- `rag chat [--show-sources]`: Interactive chat mode

**Features:**
- Environment loading with `python-dotenv`
- JSON output for structured data
- Interactive chat with exit commands
- Source attribution when requested

## Data Flow and Processing

### Ingestion Phase (Offline)
1. **Input:** Raw documents in `data/raw/`
2. **Processing:** Load → Split → Embed → Store
3. **Output:** Persistent Chroma database in `data/chroma/`
4. **Reusability:** Index persists across sessions

### Query Phase (Online)
1. **Input:** User question (string)
2. **Processing:** Embed question → Retrieve → Format → Generate
3. **Output:** Answer with optional sources

### Key Data Structures

**Document Object:**
```python
class Document:
    page_content: str      # The actual text content
    metadata: dict         # Source file, page number, etc.
```

**Vector Store Contents:**
- **Vectors:** Numeric embeddings (384-dim for MiniLM)
- **Documents:** Original text chunks
- **Metadata:** Source attribution, page info

## Provider Comparison

| Provider | Cost | Setup | Performance | Use Case |
|----------|------|-------|-------------|----------|
| Hugging Face | Free tier | API token | Good, some latency | Learning/Development |
| OpenAI | Paid | API key | Excellent | Production |
| Ollama | Free | Local install | Variable | Privacy/Offline |

## Interview-Ready Explanations

### "Explain your RAG pipeline end-to-end"

**Concise Answer:**
"This project implements a complete RAG system. During ingestion, it loads documents from a directory, splits them into overlapping chunks, generates embeddings using a transformer model, and stores everything in a persistent Chroma vector database. At query time, it embeds the user's question, retrieves the top-k most similar chunks via similarity search, and prompts an LLM to answer using only that retrieved context."

### "How does retrieval work?"

**Technical Answer:**
"Retrieval uses semantic similarity search. Both document chunks and queries are converted to high-dimensional vectors using an embedding model. The system then finds the k nearest neighbors in vector space using cosine similarity. This allows finding relevant information even when the query uses different words than the source documents."

### "How do you prevent hallucinations?"

**Multi-layered Approach:**
1. **Strict Prompting:** System instruction requires answers to come "ONLY from provided context"
2. **Source Attribution:** Optional `--show-sources` flag lets users verify grounding
3. **Retrieval Quality:** Tune chunk size, overlap, and top-k for relevant results
4. **Fallback Behavior:** Model instructed to say "I don't know" when context insufficient

### "What are the trade-offs in chunking?"

**Chunk Size:**
- **Smaller chunks (500-800 chars):** More precise retrieval, better for specific questions, but may miss context
- **Larger chunks (1000-2000 chars):** Better context preservation, fewer boundary issues, but potentially less precise

**Overlap:**
- **Benefits:** Preserves meaning across chunk boundaries
- **Costs:** Increased storage, potential redundancy
- **Typical:** 10-20% of chunk size (150 chars for 1000-char chunks)

### "How would you evaluate RAG quality?"

**Metrics:**
1. **Retrieval Quality:** Precision@K, Recall@K, Mean Reciprocal Rank
2. **Generation Quality:** ROUGE/BLEU scores, human evaluation
3. **End-to-End:** Latency, cost, user satisfaction
4. **Safety:** Hallucination rate, source attribution accuracy

**Evaluation Methods:**
- **Automated:** Create Q/A pairs, measure retrieval relevance
- **Manual:** Spot-check answers, verify source grounding
- **A/B Testing:** Compare different configurations

## Setup and Usage

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Copy environment template
cp .env.example .env
```

### Configuration (Hugging Face - Recommended for Learning)
```bash
# Get free token from https://huggingface.co/settings/tokens
# Edit .env file:
RAG_PROVIDER=huggingface
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### Basic Usage
```bash
# Add documents to data/raw/
# Ingest documents
python -m rag_project ingest

# Ask questions
python -m rag_project query "What is RAG?"

# Interactive chat
python -m rag_project chat

# Show sources
python -m rag_project query "Explain chunking" --show-sources
```

## Production Considerations

### Scalability
- **Batch Processing:** Process documents in batches during ingestion
- **Index Optimization:** Consider index compression, quantization
- **Caching:** Cache embeddings to reduce API calls
- **Distributed Storage:** Move from file-based Chroma to distributed vector DB

### Reliability
- **Error Handling:** Robust error handling for API failures
- **Rate Limiting:** Implement backoff strategies for API limits
- **Monitoring:** Track latency, cost, and quality metrics
- **Fallbacks:** Graceful degradation when services unavailable

### Security
- **API Key Management:** Secure storage of API tokens
- **Input Validation:** Sanitize user inputs and document content
- **Access Control:** Implement user permissions for document access
- **Audit Logging:** Track queries and sources for compliance

## Common Interview Questions & Answers

### Q: "How does this differ from fine-tuning an LLM?"
A: "Fine-tuning requires significant compute resources and can be costly. RAG provides up-to-date knowledge without retraining, offers better explainability through source attribution, and allows easy updates by refreshing the index."

### Q: "What happens if the retrieved context doesn't contain the answer?"
A: "The system is designed to handle this gracefully. The prompt instructs the LLM to only use provided context and admit when it doesn't know sufficient information, preventing hallucinations."

### Q: "How do you handle different document types?"
A: "The loader system automatically detects file extensions and uses appropriate parsers. PDFs use specialized PDF loaders that extract text while preserving structure, while markdown and text files use simpler text loaders."

### Q: "What's your chunking strategy and why?"
A: "I use 1000-character chunks with 150-character overlap. This balances context preservation with retrieval precision. The overlap ensures that important information spanning chunk boundaries isn't lost."

### Q: "How do you debug poor retrieval results?"
A: "First, inspect the retrieved sources using --show-sources. Then adjust top-k, chunk size, or try different embedding models. I also manually evaluate whether the retrieved chunks actually contain relevant information."

## Project Strengths for Interviews

1. **Complete Implementation:** End-to-end working system
2. **Multi-Provider Support:** Shows understanding of different deployment scenarios
3. **Production-Ready Code:** Proper configuration, error handling, CLI
4. **Educational Value:** Clear separation of concerns, well-documented
5. **Cost-Effective:** Free tier options for learning
6. **Extensible Design:** Easy to add new providers or features

## Technologies Used

- **LangChain:** Core RAG framework and integrations
- **Chroma:** Vector database for persistence
- **Hugging Face:** Free hosted inference API
- **Pydantic:** Settings management and validation
- **Typer:** CLI framework
- **Python-dotenv:** Environment variable loading

This project demonstrates practical ML engineering skills, from data processing pipelines to production-ready deployment considerations, making it an excellent showcase for machine learning and backend engineering interviews.