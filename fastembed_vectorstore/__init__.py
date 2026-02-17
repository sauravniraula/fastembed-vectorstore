from __future__ import annotations

import inspect
import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from fastembed import TextEmbedding


class FastembedEmbeddingModel(Enum):
    """Enumeration of available embedding models."""
    AllMiniLML6V2 = "AllMiniLML6V2"
    AllMiniLML6V2Q = "AllMiniLML6V2Q"
    AllMiniLML12V2 = "AllMiniLML12V2"
    AllMiniLML12V2Q = "AllMiniLML12V2Q"
    BGEBaseENV15 = "BGEBaseENV15"
    BGEBaseENV15Q = "BGEBaseENV15Q"
    BGELargeENV15 = "BGELargeENV15"
    BGELargeENV15Q = "BGELargeENV15Q"
    BGESmallENV15 = "BGESmallENV15"
    BGESmallENV15Q = "BGESmallENV15Q"
    NomicEmbedTextV1 = "NomicEmbedTextV1"
    NomicEmbedTextV15 = "NomicEmbedTextV15"
    NomicEmbedTextV15Q = "NomicEmbedTextV15Q"
    ParaphraseMLMiniLML12V2 = "ParaphraseMLMiniLML12V2"
    ParaphraseMLMiniLML12V2Q = "ParaphraseMLMiniLML12V2Q"
    ParaphraseMLMpnetBaseV2 = "ParaphraseMLMpnetBaseV2"
    BGESmallZHV15 = "BGESmallZHV15"
    BGELargeZHV15 = "BGELargeZHV15"
    ModernBertEmbedLarge = "ModernBertEmbedLarge"
    MultilingualE5Small = "MultilingualE5Small"
    MultilingualE5Base = "MultilingualE5Base"
    MultilingualE5Large = "MultilingualE5Large"
    MxbaiEmbedLargeV1 = "MxbaiEmbedLargeV1"
    MxbaiEmbedLargeV1Q = "MxbaiEmbedLargeV1Q"
    GTEBaseENV15 = "GTEBaseENV15"
    GTEBaseENV15Q = "GTEBaseENV15Q"
    GTELargeENV15 = "GTELargeENV15"
    GTELargeENV15Q = "GTELargeENV15Q"
    ClipVitB32 = "ClipVitB32"
    JinaEmbeddingsV2BaseCode = "JinaEmbeddingsV2BaseCode"


def _normalize_model_key(model_name: str) -> str:
    return "".join(ch for ch in model_name.lower() if ch.isalnum())


def _extract_model_name(model: object) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        for key in ("model", "model_name", "name"):
            value = model.get(key)
            if isinstance(value, str):
                return value
    for attr in ("model", "model_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value
    return str(model)


def _build_model_lookup() -> Dict[str, str]:
    models: Iterable[object] = TextEmbedding.list_supported_models()
    lookup: Dict[str, str] = {}
    for model in models:
        model_name = _extract_model_name(model)
        key_source = model_name.split("/")[-1]
        lookup[_normalize_model_key(key_source)] = model_name
    return lookup


def _resolve_model_name(model: FastembedEmbeddingModel) -> str:
    lookup = _build_model_lookup()
    key = _normalize_model_key(model.name)
    resolved = lookup.get(key)
    if resolved is None:
        available = ", ".join(sorted(lookup.values()))
        raise ValueError(
            f"Unsupported model: {model.name}. Supported models: {available}"
        )
    return resolved


def _filter_supported_kwargs(callable_obj: object, kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class FastembedVectorstore:
    def __init__(
        self,
        model: FastembedEmbeddingModel,
        show_download_progress: Optional[bool] = None,
        cache_directory: Union[str, os.PathLike[str], None] = None,
    ) -> None:
        """Initialize a vector store with the specified embedding model."""
        model_name = _resolve_model_name(model)
        cache_dir = Path(cache_directory) if cache_directory is not None else Path("fastembed_cache")
        init_kwargs: Dict[str, object] = {
            "model_name": model_name,
            "cache_dir": str(cache_dir),
        }
        if show_download_progress is not None:
            init_kwargs["show_progress"] = show_download_progress
        init_kwargs = _filter_supported_kwargs(TextEmbedding, init_kwargs)
        self._embedder = TextEmbedding(**init_kwargs)
        self._embeddings: Dict[str, np.ndarray] = {}

    @classmethod
    def load(
        cls,
        model: FastembedEmbeddingModel,
        path: str,
        show_download_progress: Optional[bool] = None,
        cache_directory: Union[str, os.PathLike[str], None] = None,
    ) -> "FastembedVectorstore":
        """Load a vector store from a JSON file."""
        if not Path(path).exists():
            raise FileNotFoundError("File doesn't exist")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        instance = cls(
            model=model,
            show_download_progress=show_download_progress,
            cache_directory=cache_directory,
        )
        instance._embeddings = {
            doc: np.asarray(embedding, dtype=np.float32)
            for doc, embedding in payload.items()
        }
        return instance

    def embed_documents(self, documents: List[str]) -> bool:
        """Embed documents and store them in memory."""
        if not documents:
            return True
        embeddings = list(self._embedder.embed(documents))
        for document, embedding in zip(documents, embeddings):
            self._embeddings[document] = np.asarray(embedding, dtype=np.float32)
        return True

    def search(self, query: str, n: int) -> List[Tuple[str, float]]:
        """Search for similar documents using cosine similarity."""
        if n <= 0 or not self._embeddings:
            return []
        query_embedding = np.asarray(
            next(iter(self._embedder.embed([query]))), dtype=np.float32
        )
        documents = list(self._embeddings.keys())
        matrix = np.stack([self._embeddings[doc] for doc in documents])
        doc_norms = np.linalg.norm(matrix, axis=1)
        query_norm = float(np.linalg.norm(query_embedding))
        dot_products = matrix @ query_embedding
        denom = doc_norms * query_norm
        similarities = np.divide(
            dot_products,
            denom,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=denom != 0,
        )
        top_indices = np.argsort(-similarities)[: min(n, len(documents))]
        return [(documents[index], float(similarities[index])) for index in top_indices]

    def save(self, path: str) -> bool:
        """Save embeddings to a JSON file."""
        payload = {
            document: embedding.tolist()
            for document, embedding in self._embeddings.items()
        }
        file_path = Path(path)
        if file_path.parent:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return True


__all__ = ["FastembedEmbeddingModel", "FastembedVectorstore"]
