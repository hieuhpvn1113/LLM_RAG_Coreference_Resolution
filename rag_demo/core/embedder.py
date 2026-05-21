# core/embedder.py — Tạo vector embedding (local, miễn phí)
# Model: intfloat/multilingual-e5-large — 1024 dims
# Chạy hoàn toàn local, không cần API key
#
# E5 yêu cầu prefix theo vai trò:
#   "passage: <text>"  → khi encode nội dung tài liệu (chunk, câu để chunking)
#   "query: <text>"    → khi encode câu hỏi tìm kiếm
# Không dùng đúng prefix → accuracy giảm ~10-15%.

from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load model lần đầu gọi, cache lại cho các lần sau."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Encode 1 đoạn text dạng PASSAGE (chunk nội dung, câu trong chunking).
    Dùng cho: lưu vector vào Qdrant, embed_batch trong semantic chunking.
    """
    model = _get_model()
    vector = model.encode(f"passage: {text}", normalize_embeddings=True)
    return vector.tolist()


def embed_query(text: str) -> list[float]:
    """
    Encode 1 câu hỏi tìm kiếm dạng QUERY.
    Dùng cho: search trong retriever.py thay vì embed_text.
    """
    model = _get_model()
    vector = model.encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Encode nhiều đoạn text dạng PASSAGE cùng lúc.
    Dùng cho: semantic chunking (_semantic_units), ingest batch.
    """
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()
