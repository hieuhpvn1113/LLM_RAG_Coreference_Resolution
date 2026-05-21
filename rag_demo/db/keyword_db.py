# db/keyword_db.py — Elasticsearch Keyword DB client
"""
Lưu trữ để search (inverted index):
  - clean_text, title, summary    : full-text BM25
  - keywords                      : keyword match
  - hypothetical_questions        : boosted x2
  - level, seq_no                 : filter / sort

Sau khi tìm được chunk_id, full text lấy từ PostgreSQL — không trả về ở đây.
Chỉ index Level 2 (paragraph).
"""

from elasticsearch import Elasticsearch, helpers
from config import ES_URL, ES_INDEX


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

_COMPAT_HEADERS = {
    "Accept":       "application/json",
    "Content-Type": "application/json",
}


class KeywordDB:

    def __init__(self):
        self.client: Elasticsearch | None = None

    def connect(self):
        self.client = Elasticsearch(ES_URL, headers=_COMPAT_HEADERS)
        info = self.client.info()
        print(f"✅ Elasticsearch connected: {ES_URL} (v{info['version']['number']})")

    def ensure_index(self):
        if not self.client.indices.exists(index=ES_INDEX):
            self.client.indices.create(index=ES_INDEX, body=_INDEX_MAPPING)
            print(f"  Created ES index: '{ES_INDEX}'")
        else:
            print(f"  ES index '{ES_INDEX}' already exists")

    # ── Write ─────────────────────────────────────────────────────────────────

    def index_chunk(self, chunk: dict):
        """Index 1 chunk — lưu full text để ES xây inverted index cho BM25."""
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
        """Bulk index nhiều chunks."""
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
        """
        BM25 multi-field search — trả về chunk_id + score + title.
        Full text KHÔNG trả về ở đây — fetch từ PostgreSQL sau khi có chunk_id.
        """
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
                "chunk_id": h["_source"]["chunk_id"],
                "score":    h["_score"],
                "title":    h["_source"].get("title", ""),
                "source":   "elasticsearch",
            }
            for h in resp["hits"]["hits"]
        ]

    def delete_by_doc(self, doc_id: str):
        self.client.delete_by_query(
            index=ES_INDEX,
            body={"query": {"term": {"doc_id": doc_id}}},
        )
