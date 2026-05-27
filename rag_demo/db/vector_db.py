# db/vector_db.py — Qdrant Hybrid DB (Dense + Sparse BM25)
"""
Mỗi point chứa:
  - vector["dense"]  : 1024 chiều multilingual-e5-large (semantic)
  - vector["sparse"] : BM25 sparse vector từ fastembed (keyword)

Payload lưu ĐẦY ĐỦ để:
  1. Self-Querying filter (keywords, entities, entity_types, source_file)
  2. Trả về raw_text trực tiếp từ Qdrant — không cần query thêm PostgreSQL
     cho bước hiển thị kết quả tìm kiếm (search preview)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector,
    Filter, FieldCondition, MatchValue, MatchAny,
    FilterSelector,
    Prefetch, FusionQuery, Fusion,
)
from fastembed import SparseTextEmbedding
from config import QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM

_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def _make_sparse(text: str) -> SparseVector:
    result = list(_get_sparse_model().embed([text]))[0]
    return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())


def _make_sparse_batch(texts: list[str]) -> list[SparseVector]:
    results = list(_get_sparse_model().embed(texts))
    return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) for r in results]


class VectorDB:

    def __init__(self):
        self.client: QdrantClient | None = None

    def connect(self):
        self.client = QdrantClient(url=QDRANT_URL)
        print(f"✅ Qdrant connected: {QDRANT_URL}")

    def ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"  Created Qdrant collection: '{QDRANT_COLLECTION}' "
                  f"(dense={VECTOR_DIM}d + sparse BM25 + full payload)")
        else:
            print(f"  Qdrant collection '{QDRANT_COLLECTION}' already exists")

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_batch(self, chunks: list, dense_vectors: list):
        """
        Upsert dense + sparse + payload đầy đủ (bao gồm raw_text và clean_text).
        Payload đủ để:
          - Self-Querying lọc cứng (keywords, entities, source_file)
          - Hiển thị kết quả search preview không cần query PostgreSQL
          - Resolve L1 vẫn dùng PostgreSQL (raw_text L1 không lưu ở đây)
        """
        texts          = [c.get("clean_text", "") for c in chunks]
        sparse_vectors = _make_sparse_batch(texts)

        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector={
                    "dense":  dense_vec,
                    "sparse": sparse_vec,
                },
                payload={
                    # ── Core IDs ───────────────────────────────────────────
                    "chunk_id":    chunk["chunk_id"],
                    "doc_id":      chunk["doc_id"],
                    "parent_id":   chunk.get("parent_id", ""),
                    "source_file": chunk.get("source_file", ""),
                    "level":       chunk.get("level", 2),
                    "seq_no":      chunk.get("seq_no", "0"),

                    # ── Full text (dùng cho search preview & fallback) ──────
                    "clean_text":  chunk.get("clean_text", ""),   # L2 clean text
                    "raw_text":    chunk.get("raw_text",   ""),   # L2 raw text (nếu có)

                    # ── Semantic metadata ──────────────────────────────────
                    "title":   chunk.get("title",   ""),
                    "summary": chunk.get("summary", ""),
                    "hypothetical_questions": chunk.get("hypothetical_questions", []),

                    # ── Self-Querying filter fields ────────────────────────
                    # keywords: list[str] — dùng MatchAny filter
                    "keywords": chunk.get("keywords", []),

                    # entities: list[str] tên — dùng MatchAny filter
                    "entities": [
                        e["name"] if isinstance(e, dict) else str(e)
                        for e in chunk.get("entities", [])
                    ],

                    # entity_types: list[str] loại — PERSON/ORG/CONCEPT/LOCATION
                    "entity_types": [
                        e["type"] if isinstance(e, dict) else "CONCEPT"
                        for e in chunk.get("entities", [])
                    ],

                    # ── Stats ──────────────────────────────────────────────
                    "token_count":      chunk.get("token_count", 0),
                    "character_length": len(chunk.get("clean_text", "")),
                },
            )
            for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors)
        ]

        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  Qdrant: upserted {len(points)} chunks "
              f"(dense + sparse BM25 + full payload)")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vector: list, query_text: str, top_k: int = 9,
               qdrant_filter: Filter | None = None) -> list:
        """
        Hybrid search: dense prefetch + sparse BM25 prefetch → RRF fusion.
        qdrant_filter từ Self-Querying lọc cứng TRƯỚC khi rank vector.
        Trả về đầy đủ payload (bao gồm clean_text, title, summary...).
        """
        sparse_vec = _make_sparse(query_text)

        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                # ── IDs ──────────────────────────────────────────────────
                "chunk_id":    r.payload.get("chunk_id", str(r.id)),
                "doc_id":      r.payload.get("doc_id", ""),
                "parent_id":   r.payload.get("parent_id", ""),
                "source_file": r.payload.get("source_file", ""),
                "seq_no":      r.payload.get("seq_no", ""),
                # ── Score ─────────────────────────────────────────────────
                "score":       r.score,
                "source":      "qdrant_hybrid",
                # ── Text (dùng cho preview / fallback) ────────────────────
                "title":       r.payload.get("title", ""),
                "summary":     r.payload.get("summary", ""),
                "clean_text":  r.payload.get("clean_text", ""),
                "raw_text":    r.payload.get("raw_text", ""),
                # ── Filter metadata (dùng cho debug) ──────────────────────
                "keywords":    r.payload.get("keywords", []),
                "entities":    r.payload.get("entities", []),
            }
            for r in response.points
        ]

    def search_dense_only(self, query_vector: list, top_k: int = 9,
                          qdrant_filter: Filter | None = None) -> list:
        """Dense-only fallback."""
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "chunk_id":   r.payload.get("chunk_id", str(r.id)),
                "score":      r.score,
                "title":      r.payload.get("title", ""),
                "clean_text": r.payload.get("clean_text", ""),
                "source":     "qdrant_dense",
            }
            for r in response.points
        ]

    def delete_by_doc(self, doc_id: str):
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )
