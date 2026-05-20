# db/keyword_db.py — Elasticsearch Keyword DB client
"""
Lưu trữ:
  - clean_text, title, summary    : full-text search (BM25)
  - keywords                      : keyword match
  - hypothetical_questions        : boosted x2 — lợi thế lớn nhất của keyword search
  - level, seq_no                 : filter / sort

Chỉ index Level 2 (paragraph).
"""

from elasticsearch import Elasticsearch, helpers
from config import ES_URL, ES_INDEX


# Mapping chi tiết — boost hypothetical_questions khi search
_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id":               {"type": "keyword"},
            "doc_id":                 {"type": "keyword"},
            "clean_text":             {"type": "text", "analyzer": "standard"},
            "title":                  {"type": "text", "analyzer": "standard"},
            "summary":                {"type": "text", "analyzer": "standard"},
            "keywords":               {"type": "keyword"},
            "hypothetical_questions": {
                "type": "text",
                "analyzer": "standard",
            },
            "level":                  {"type": "integer"},
            "seq_no":                 {"type": "keyword"},
            "source_file":            {"type": "keyword"},
        }
    },
    "settings": {
        "index": {
            "number_of_shards":   1,
            "number_of_replicas": 0,
        }
    },
}

# Headers tương thích với ES v8 khi dùng client v9
_COMPAT_HEADERS = {
    "Accept":       "application/json",
    "Content-Type": "application/json",
}


class KeywordDB:

    def __init__(self):
        self.client: Elasticsearch | None = None

    def connect(self):
        # headers override để tránh lỗi version mismatch (client v9 vs server v8)
        self.client = Elasticsearch(ES_URL, headers=_COMPAT_HEADERS)
        info = self.client.info()
        print(f"✅ Elasticsearch connected: {ES_URL} (v{info['version']['number']})")

    def ensure_index(self):
        """Tạo index với mapping nếu chưa có."""
        if not self.client.indices.exists(index=ES_INDEX):
            self.client.indices.create(index=ES_INDEX, body=_INDEX_MAPPING)
            print(f"  Created ES index: '{ES_INDEX}'")
        else:
            print(f"  ES index '{ES_INDEX}' already exists")

    # ── Write ─────────────────────────────────────────────────────────────────

    def index_chunk(self, chunk: dict):
        """Index 1 chunk vào Elasticsearch (upsert by chunk_id)."""
        self.client.index(
            index=ES_INDEX,
            id=chunk["chunk_id"],
            document={
                "chunk_id":               chunk["chunk_id"],
                "doc_id":                 chunk["doc_id"],
                "clean_text":             chunk.get("clean_text", ""),
                "title":                  chunk.get("title", ""),
                "summary":                chunk.get("summary", ""),
                "keywords":               chunk.get("keywords", []),
                "hypothetical_questions": chunk.get("hypothetical_questions", []),
                "level":                  chunk.get("level", 2),
                "seq_no":                 chunk.get("seq_no", "0"),
                "source_file":            chunk.get("source_file", ""),
            },
        )

    def index_batch(self, chunks: list):
        """Bulk index nhiều chunks cùng lúc."""
        actions = [
            {
                "_index": ES_INDEX,
                "_id":    chunk["chunk_id"],
                "_source": {
                    "chunk_id":               chunk["chunk_id"],
                    "doc_id":                 chunk["doc_id"],
                    "clean_text":             chunk.get("clean_text", ""),
                    "title":                  chunk.get("title", ""),
                    "summary":                chunk.get("summary", ""),
                    "keywords":               chunk.get("keywords", []),
                    "hypothetical_questions": chunk.get("hypothetical_questions", []),
                    "level":                  chunk.get("level", 2),
                    "seq_no":                 chunk.get("seq_no", "0"),
                    "source_file":            chunk.get("source_file", ""),
                },
            }
            for chunk in chunks
        ]
        success, errors = helpers.bulk(self.client, actions, raise_on_error=False)
        if errors:
            print(f"  ⚠️  ES bulk index errors: {len(errors)}")
        else:
            print(f"  ES: indexed {success} chunks")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, doc_id: str | None = None) -> list:
        """BM25 multi-field search với hypothetical_questions boost x2."""
        must_clauses = [
            {
                "multi_match": {
                    "query":  query,
                    "fields": [
                        "clean_text",
                        "title^1.5",
                        "summary",
                        "keywords^1.5",
                        "hypothetical_questions^2",
                    ],
                    "type":      "best_fields",
                    "fuzziness": "AUTO",
                }
            }
        ]
        filter_clauses = [{"term": {"level": 2}}]
        if doc_id:
            filter_clauses.append({"term": {"doc_id": doc_id}})

        body = {
            "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
            "size":  top_k,
        }
        resp = self.client.search(index=ES_INDEX, body=body)
        return [
            {
                "chunk_id":   h["_source"]["chunk_id"],
                "score":      h["_score"],
                "title":      h["_source"].get("title", ""),
                "clean_text": h["_source"].get("clean_text", ""),
                "source":     "elasticsearch",
            }
            for h in resp["hits"]["hits"]
        ]

    def delete_by_doc(self, doc_id: str):
        """Xóa tất cả chunks của 1 document."""
        self.client.delete_by_query(
            index=ES_INDEX,
            body={"query": {"term": {"doc_id": doc_id}}},
        )
