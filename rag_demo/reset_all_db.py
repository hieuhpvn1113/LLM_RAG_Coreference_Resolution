"""
reset_all_db.py — Xóa sạch toàn bộ dữ liệu trong 4 DB và tạo lại schema từ đầu.

Chạy: python reset_all_db.py
Tùy chọn:
  --yes       : Bỏ qua bước confirm (dùng khi chạy script tự động)
  --pg-only   : Chỉ reset PostgreSQL
  --qdrant-only
  --es-only
  --neo4j-only
"""

import asyncio
import argparse
import sys
import os

# Đảm bảo import đúng config từ thư mục rag_demo
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    PG_DSN,
    QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM,
    ES_URL, ES_INDEX,
    NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD,
)


# ═══════════════════════════════════════════════════════════════════════════
# PostgreSQL — TRUNCATE CASCADE rồi để ON DELETE CASCADE tự dọn chunks
# ═══════════════════════════════════════════════════════════════════════════

async def reset_postgres():
    import asyncpg
    print("\n🔄 [PostgreSQL] Đang reset...")
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=3)
    async with pool.acquire() as conn:
        # TRUNCATE CASCADE — xóa documents kéo theo chunks (ON DELETE CASCADE)
        # RESTART IDENTITY reset sequence (nếu có)
        await conn.execute("TRUNCATE TABLE search_logs RESTART IDENTITY CASCADE;")
        await conn.execute("TRUNCATE TABLE chunks     RESTART IDENTITY CASCADE;")
        await conn.execute("TRUNCATE TABLE documents  RESTART IDENTITY CASCADE;")
        print("  ✅ Đã TRUNCATE: documents, chunks, search_logs")
    await pool.close()


# ═══════════════════════════════════════════════════════════════════════════
# Qdrant — Xóa collection rồi tạo lại
# ═══════════════════════════════════════════════════════════════════════════

def reset_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    print("\n🔄 [Qdrant] Đang reset...")
    client = QdrantClient(url=QDRANT_URL)

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"  🗑️  Đã xóa collection '{QDRANT_COLLECTION}'")
    else:
        print(f"  ℹ️  Collection '{QDRANT_COLLECTION}' chưa tồn tại, bỏ qua bước xóa")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"  ✅ Đã tạo lại collection '{QDRANT_COLLECTION}' (dim={VECTOR_DIM})")


# ═══════════════════════════════════════════════════════════════════════════
# Elasticsearch — Xóa index rồi tạo lại với mapping đầy đủ
# ═══════════════════════════════════════════════════════════════════════════

def reset_elasticsearch():
    from elasticsearch import Elasticsearch

    _COMPAT_HEADERS = {
        "Accept":       "application/json",
        "Content-Type": "application/json",
    }
    _INDEX_MAPPING = {
        "mappings": {
            "properties": {
                "chunk_id":               {"type": "keyword"},
                "doc_id":                 {"type": "keyword"},
                "clean_text":             {"type": "text", "analyzer": "standard"},
                "title":                  {"type": "text", "analyzer": "standard"},
                "summary":                {"type": "text", "analyzer": "standard"},
                "keywords":               {"type": "keyword"},
                "hypothetical_questions": {"type": "text", "analyzer": "standard"},
                "level":                  {"type": "integer"},
                "seq_no":                 {"type": "integer"},
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

    print("\n🔄 [Elasticsearch] Đang reset...")
    client = Elasticsearch(ES_URL, headers=_COMPAT_HEADERS)

    if client.indices.exists(index=ES_INDEX):
        client.indices.delete(index=ES_INDEX)
        print(f"  🗑️  Đã xóa index '{ES_INDEX}'")
    else:
        print(f"  ℹ️  Index '{ES_INDEX}' chưa tồn tại, bỏ qua bước xóa")

    client.indices.create(index=ES_INDEX, body=_INDEX_MAPPING)
    print(f"  ✅ Đã tạo lại index '{ES_INDEX}'")


# ═══════════════════════════════════════════════════════════════════════════
# Neo4j — Xóa toàn bộ nodes + relationships, giữ lại constraints
# ═══════════════════════════════════════════════════════════════════════════

def reset_neo4j():
    from neo4j import GraphDatabase

    print("\n🔄 [Neo4j] Đang reset...")
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        # Xóa toàn bộ nodes + relationships
        result = session.run("MATCH (n) DETACH DELETE n")
        summary = result.consume()
        deleted = summary.counters.nodes_deleted
        rels    = summary.counters.relationships_deleted
        print(f"  🗑️  Đã xóa {deleted} nodes, {rels} relationships")

        # Đảm bảo constraints vẫn tồn tại
        session.run("CREATE CONSTRAINT chunk_id    IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.chunk_id IS UNIQUE")
        session.run("CREATE CONSTRAINT doc_id      IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id   IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity)   REQUIRE e.name     IS UNIQUE")
        print("  ✅ Constraints đã được đảm bảo")

    driver.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Reset toàn bộ dữ liệu RAG DB")
    parser.add_argument("--yes",        action="store_true", help="Bỏ qua confirm")
    parser.add_argument("--pg-only",    action="store_true")
    parser.add_argument("--qdrant-only",action="store_true")
    parser.add_argument("--es-only",    action="store_true")
    parser.add_argument("--neo4j-only", action="store_true")
    args = parser.parse_args()

    # Xác định DB nào sẽ reset
    specific = args.pg_only or args.qdrant_only or args.es_only or args.neo4j_only
    do_pg     = args.pg_only     or not specific
    do_qdrant = args.qdrant_only or not specific
    do_es     = args.es_only     or not specific
    do_neo4j  = args.neo4j_only  or not specific

    targets = []
    if do_pg:     targets.append("PostgreSQL (documents, chunks, search_logs)")
    if do_qdrant: targets.append(f"Qdrant     (collection: {QDRANT_COLLECTION})")
    if do_es:     targets.append(f"Elasticsearch (index: {ES_INDEX})")
    if do_neo4j:  targets.append("Neo4j      (toàn bộ nodes + relationships)")

    print("=" * 60)
    print("⚠️  CẢNH BÁO: Thao tác này sẽ XÓA SẠCH dữ liệu của:")
    for t in targets:
        print(f"   • {t}")
    print("=" * 60)

    if not args.yes:
        confirm = input("\nBạn có chắc chắn muốn tiếp tục? (gõ 'yes' để xác nhận): ")
        if confirm.strip().lower() != "yes":
            print("❌ Hủy thao tác.")
            return

    errors = []

    if do_pg:
        try:
            await reset_postgres()
        except Exception as e:
            print(f"  ❌ PostgreSQL lỗi: {e}")
            errors.append(("PostgreSQL", e))

    if do_qdrant:
        try:
            reset_qdrant()
        except Exception as e:
            print(f"  ❌ Qdrant lỗi: {e}")
            errors.append(("Qdrant", e))

    if do_es:
        try:
            reset_elasticsearch()
        except Exception as e:
            print(f"  ❌ Elasticsearch lỗi: {e}")
            errors.append(("Elasticsearch", e))

    if do_neo4j:
        try:
            reset_neo4j()
        except Exception as e:
            print(f"  ❌ Neo4j lỗi: {e}")
            errors.append(("Neo4j", e))

    print("\n" + "=" * 60)
    if errors:
        print(f"⚠️  Hoàn thành với {len(errors)} lỗi:")
        for db, err in errors:
            print(f"   • {db}: {err}")
    else:
        print("✅ Reset hoàn tất! Tất cả DB đã sạch và sẵn sàng insert lại.")
    print("=" * 60)
    print("\n💡 Bước tiếp theo: chạy pipeline ingest lại từ đầu")
    print("   python main.py ingest <đường_dẫn_file>")


if __name__ == "__main__":
    asyncio.run(main())
