from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma

from rag_project.hf_inference import HFInferenceEmbeddings, hf_generate
from rag_project.settings import Settings


def get_llm(settings: Settings):
    if settings.rag_provider == "huggingface":
        try:
            from langchain_community.llms import HuggingFaceHub
        except ImportError as e:
            raise RuntimeError("Install HuggingFace support: pip install langchain-community") from e

        if not settings.huggingfacehub_api_token:
            raise RuntimeError(
                "Missing HUGGINGFACEHUB_API_TOKEN. Set it in .env (free token from https://huggingface.co/settings/tokens)."
            )

        return HuggingFaceHub(
            repo_id=settings.hf_model,
            huggingfacehub_api_token=settings.huggingfacehub_api_token,
            model_kwargs={"temperature": 0.2, "max_length": 512},
        )

    if settings.rag_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("Install OpenAI support: pip install langchain-openai") from e

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when RAG_PROVIDER=openai")

        return ChatOpenAI(model=settings.openai_model)

    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise RuntimeError("Install Ollama support: pip install langchain-ollama") from e

    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.rag_provider == "huggingface":
        if not settings.huggingfacehub_api_token:
            raise RuntimeError(
                "Missing HUGGINGFACEHUB_API_TOKEN. Set it in .env (free token from https://huggingface.co/settings/tokens)."
            )
        return HFInferenceEmbeddings(token=settings.huggingfacehub_api_token, model=settings.hf_embedding_model)

    if settings.rag_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise RuntimeError("Install OpenAI support: pip install langchain-openai") from e

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when RAG_PROVIDER=openai")

        # Rely on OPENAI_API_KEY in the environment for compatibility across versions.
        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as e:
        raise RuntimeError("Install Ollama support: pip install langchain-ollama") from e

    return OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)


def get_vectorstore(settings: Settings, embeddings: Embeddings, *, create_if_missing: bool) -> Chroma:
    persist_directory = str(settings.rag_chroma_dir)
    collection_name = settings.rag_collection
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def _format_docs(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(settings: Settings, *, show_sources: bool = False):
    embeddings = get_embeddings(settings)
    vectorstore = get_vectorstore(settings, embeddings, create_if_missing=False)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.rag_top_k})
    llm = get_llm(settings)

    system = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the context is insufficient, say you don't know and ask for the missing info."
    )

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            f"{system}\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
    )

    def retrieve(question: str) -> list[Document]:
        return retriever.invoke(question)

    context_runnable = RunnableLambda(retrieve) | RunnableLambda(_format_docs)

    chain = ({"question": RunnablePassthrough(), "context": context_runnable} | prompt | llm | StrOutputParser())

    if not show_sources:
        return chain

    def with_sources(question: str) -> dict[str, Any]:
        docs = retriever.invoke(question)
        answer = chain.invoke(question)
        sources = []
        for doc in docs:
            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
            )
        return {"answer": answer, "sources": sources}

    return RunnableLambda(with_sources)
