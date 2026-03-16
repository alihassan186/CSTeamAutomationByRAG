from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rag_provider: Literal["huggingface", "ollama", "openai"] = Field(default="huggingface", alias="RAG_PROVIDER")

    rag_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw", alias="RAG_RAW_DIR")
    rag_chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="RAG_CHROMA_DIR")
    rag_collection: str = Field(default="rag_docs", alias="RAG_COLLECTION")
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")

    # Hugging Face Inference API
    huggingfacehub_api_token: str | None = Field(default=None, alias="HUGGINGFACEHUB_API_TOKEN")
    hf_model: str = Field(default="microsoft/phi-1.5", alias="HF_MODEL")
    hf_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="HF_EMBEDDING_MODEL")

    # OpenAI
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")


def get_settings() -> Settings:
    return Settings()
