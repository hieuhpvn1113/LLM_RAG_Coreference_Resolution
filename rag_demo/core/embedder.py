# core/embedder.py - Tao vector embedding

import hashlib
import random
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
    _EMBED_BACKEND_OK = True
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]
    _EMBED_BACKEND_OK = False

from config import EMBED_MODEL, VECTOR_DIM

_model: Any = None


def is_embed_backend_ok() -> bool:
    return _EMBED_BACKEND_OK


def _fallback_vector(text: str) -> list[float]:
    # Deterministic hash-based vector de giu pipeline hoat dong khi torch DLL loi.
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(VECTOR_DIM)]
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def _get_model():
    global _model
    if not _EMBED_BACKEND_OK:
        return None
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


def embed_text(text: str) -> list[float]:
    model = _get_model()
    if model is None:
        return _fallback_vector(f"passage: {text}")
    vector = model.encode(f"passage: {text}", normalize_embeddings=True)
    return vector.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    if model is None:
        return _fallback_vector(f"query: {text}")
    vector = model.encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    if model is None:
        return [_fallback_vector(f"passage: {t}") for t in texts]
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()
