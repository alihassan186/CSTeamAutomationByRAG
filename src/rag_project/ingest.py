from __future__ import annotations

import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_project.loaders import load_documents
from rag_project.rag import get_embeddings, get_vectorstore
from rag_project.settings import Settings

# TODO: Add logging and error handling
def ingest(settings: Settings, *, reset: bool = False) -> dict:
    raw_dir = settings.rag_raw_dir
    chroma_dir = settings.rag_chroma_dir

    if reset and chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(raw_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings(settings)
    vectorstore = get_vectorstore(settings, embeddings, create_if_missing=True)
    if chunks:
        vectorstore.add_documents(chunks)
        persist = getattr(vectorstore, "persist", None)
        if callable(persist):
            persist()

    print(f"Ingestion complete: {len(docs)} documents loaded, {len(chunks)} chunks created.")
    return {
        "raw_dir": str(raw_dir),
        "chroma_dir": str(chroma_dir),
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "provider": settings.rag_provider,
        "collection": settings.rag_collection,
    }
