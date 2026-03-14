from __future__ import annotations

from typing import Any

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings


def _mean_pool(token_embeddings: list[list[float]]) -> list[float]:
    if not token_embeddings:
        return []
    dim = len(token_embeddings[0])
    sums = [0.0] * dim
    for row in token_embeddings:
        for i, v in enumerate(row):
            sums[i] += float(v)
    n = float(len(token_embeddings))
    return [v / n for v in sums]


def _to_vector(obj: Any) -> list[float]:
    # HF feature-extraction can return:
    # - list[float] (sentence embedding)
    # - list[list[float]] (token embeddings) -> mean pool
    if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
        return [float(x) for x in obj]
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return _mean_pool(obj)
    raise TypeError(f"Unexpected embeddings response type: {type(obj)}")


class HFInferenceEmbeddings(Embeddings):
    def __init__(self, *, token: str, model: str):
        self._client = InferenceClient(token=token)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            obj = self._client.feature_extraction(text, model=self._model)
            vectors.append(_to_vector(obj))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        obj = self._client.feature_extraction(text, model=self._model)
        return _to_vector(obj)


def hf_generate(
    *,
    token: str,
    model: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    client = InferenceClient(token=token)
    out = client.text_generation(
        prompt,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_full_text=False,
    )
    return out.strip()
