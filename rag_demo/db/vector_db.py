# db/vector_db.py — Qdrant Vector DB client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from config import QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM


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
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            print(f"  Created Qdrant collection: '{QDRANT_COLLECTION}' (dim={VECTOR_DIM})")
        else:
            print(f"  Qdrant collection '{QDRANT_COLLECTION}' already exists")

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_batch(self, chunks: list, vectors: list):
        """
        Chỉ lưu metadata nhỏ để filter/sort — KHÔNG lưu full text.
        Full text là trách nhiệm của PostgreSQL (source of truth).
        """
        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={
                    "chunk_id":    chunk["chunk_id"],
                    "doc_id":      chunk["doc_id"],
                    "level":       chunk.get("level", 2),
                    "title":       chunk.get("title", ""),
                    "source_file": chunk.get("source_file", ""),
                    "seq_no":      chunk.get("seq_no", "0"),
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  Qdrant: upserted {len(points)} chunks")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vector: list, top_k: int = 3,
               doc_id: str | None = None) -> list:
        """
        Dense vector search — trả về chunk_id + score + title.
        Full text KHÔNG trả về ở đây — fetch từ PostgreSQL sau khi có chunk_id.
        """
        query_filter = None
        if doc_id:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )

        try:
            response = self.client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            results = response.points

        except AttributeError:
            results = self.client.search(          # type: ignore[attr-defined]
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        return [
            {
                "chunk_id": r.payload.get("chunk_id", str(r.id)),
                "score":    r.score,
                "title":    r.payload.get("title", ""),
                "source":   "qdrant",
            }
            for r in results
        ]

    def delete_by_doc(self, doc_id: str):
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
