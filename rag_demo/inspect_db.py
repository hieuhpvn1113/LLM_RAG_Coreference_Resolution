#!/usr/bin/env python3
"""
inspect_db.py — Kiểm tra dữ liệu trong tất cả 4 DB của hệ thống RAG
Usage:
    cd rag_demo
    python inspect_db.py            # Xem tổng quan tất cả DB
    python inspect_db.py --pg       # Chỉ PostgreSQL
    python inspect_db.py --qdrant   # Chỉ Qdrant
    python inspect_db.py --es       # Chỉ Elasticsearch
    python inspect_db.py --neo4j    # Chỉ Neo4j
    python inspect_db.py --full     # Xem full text (không truncate)
    python inspect_db.py --doc <doc_id>  # Lọc theo document cụ thể
"""

import asyncio
import argparse
import json
import sys

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def hdr(title: str):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

def sub(title: str):
    print(f"\n{YELLOW}{'─'*50}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'─'*50}{RESET}")

def ok(msg):  print(f"{GREEN}✅ {msg}{RESET}")
def err(msg): print(f"{RED}❌ {msg}{RESET}")
def dim(msg): print(f"{DIM}{msg}{RESET}")

def trunc(text: str, n: int = 120) -> str:
    if not text:
        return "(empty)"
    return text[:n] + "…" if len(text) > n else text


# ─── PostgreSQL ───────────────────────────────────────────────────────────────
async def inspect_postgres(doc_id: str | None = None, full: bool = False):
    hdr("PostgreSQL — Documents & Chunks")
    try:
        import asyncpg
        from config import PG_DSN
        conn = await asyncpg.connect(PG_DSN)

        # ── Documents
        sub("📄 Documents table")
        docs = await conn.fetch(
            "SELECT doc_id, file_name, status, total_chunks, created_at FROM documents ORDER BY created_at DESC LIMIT 20"
        )
        if not docs:
            dim("  (no documents found)")
        else:
            print(f"  {'doc_id':<38} {'file_name':<25} {'status':<12} {'chunks':>7}")
            print(f"  {'-'*38} {'-'*25} {'-'*12} {'-'*7}")
            for d in docs:
                print(f"  {str(d['doc_id']):<38} {d['file_name']:<25} {d['status']:<12} {str(d['total_chunks'] or 0):>7}")

        # ── Chunks summary
        sub("🧩 Chunks table — Summary by Level")
        counts = await conn.fetch(
            "SELECT level, COUNT(*) as cnt, AVG(token_count)::INT as avg_tokens FROM chunks GROUP BY level ORDER BY level"
        )
        for row in counts:
            print(f"  Level {row['level']}: {row['cnt']} chunks | avg tokens = {row['avg_tokens']}")

        # ── embed_status
        sub("📦 Embed Status")
        embed_stats = await conn.fetch(
            "SELECT embed_status, COUNT(*) as cnt FROM chunks GROUP BY embed_status"
        )
        for row in embed_stats:
            status_icon = "✅" if row["embed_status"] == "done" else "⏳"
            print(f"  {status_icon} {row['embed_status']}: {row['cnt']} chunks")

        # ── Sample chunks
        filter_sql = "WHERE doc_id = $1::UUID" if doc_id else ""
        filter_val = [doc_id] if doc_id else []

        sub(f"🔍 Sample Chunks (level=2, {'doc_id=' + doc_id if doc_id else 'all docs'})")
        rows = await conn.fetch(
            f"""
            SELECT chunk_id, doc_id, level, seq_no, token_count,
                   clean_text, title, summary,
                   keywords, hypothetical_questions, entities,
                   embed_status, prev_id, next_id, parent_id
            FROM chunks
            {filter_sql}
            WHERE level = 2
            ORDER BY seq_no
            LIMIT 5
            """,
            *filter_val
        )

        for i, r in enumerate(rows):
            print(f"\n  {BOLD}── Chunk #{i+1} ──{RESET}")
            print(f"  chunk_id   : {r['chunk_id']}")
            print(f"  doc_id     : {r['doc_id']}")
            print(f"  seq_no     : {r['seq_no']}  |  level: {r['level']}  |  tokens: {r['token_count']}")
            print(f"  embed      : {r['embed_status']}")
            print(f"  prev_id    : {r['prev_id']}")
            print(f"  next_id    : {r['next_id']}")
            print(f"  parent_id  : {r['parent_id']}")
            print(f"  title      : {r['title'] or '(empty)'}")
            print(f"  summary    : {trunc(r['summary'] or '', 200 if full else 120)}")
            print(f"  clean_text : {trunc(r['clean_text'] or '', 300 if full else 150)}")

            # JSON fields
            for field in ["keywords", "hypothetical_questions", "entities"]:
                val = r[field]
                if val:
                    parsed = json.loads(val) if isinstance(val, str) else val
                    if parsed:
                        print(f"  {field}: {json.dumps(parsed, ensure_ascii=False, indent=None)[:200]}")
                    else:
                        print(f"  {field}: []")
                else:
                    print(f"  {field}: (null)")

        # ── search_logs
        sub("📊 Search Logs (last 5)")
        logs = await conn.fetch(
            "SELECT query_original, latency_ms, created_at FROM search_logs ORDER BY created_at DESC LIMIT 5"
        )
        if not logs:
            dim("  (no search logs yet)")
        else:
            for lg in logs:
                print(f"  [{lg['created_at'].strftime('%H:%M:%S')}] {trunc(lg['query_original'], 60)} — {lg['latency_ms']}ms")

        await conn.close()
        ok("PostgreSQL inspection done")

    except Exception as e:
        err(f"PostgreSQL error: {e}")


# ─── Qdrant ───────────────────────────────────────────────────────────────────
def inspect_qdrant(doc_id: str | None = None, full: bool = False):
    hdr("Qdrant — Vector DB")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
        from config import QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM

        client = QdrantClient(url=QDRANT_URL)

        # ── Collection info
        sub("📦 Collection Info")
        info = client.get_collection(QDRANT_COLLECTION)
        print(f"  collection   : {QDRANT_COLLECTION}")
        print(f"  vector dim   : {info.config.params.vectors.size}")
        print(f"  distance     : {info.config.params.vectors.distance}")
        print(f"  total points : {info.points_count}")
        print(f"  indexed      : {info.indexed_vectors_count}")

        if info.points_count == 0:
            dim("  (collection is empty)")
            return

        # ── Sample points
        sub(f"🔍 Sample Points ({'doc_id=' + doc_id if doc_id else 'all'})")
        scroll_filter = None
        if doc_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )

        results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=5,
            with_payload=True,
            with_vectors=False,   # không cần vector bytes
        )

        for i, pt in enumerate(results):
            p = pt.payload
            print(f"\n  {BOLD}── Point #{i+1} ──{RESET}")
            print(f"  id (chunk_id): {pt.id}")
            print(f"  doc_id       : {p.get('doc_id', '?')}")
            print(f"  level        : {p.get('level', '?')} | seq_no: {p.get('seq_no', '?')}")
            print(f"  source_file  : {p.get('source_file', '?')}")
            print(f"  title        : {p.get('title') or '(empty)'}")
            print(f"  summary      : {trunc(p.get('summary', ''), 120 if not full else 250)}")
            print(f"  clean_text   : {trunc(p.get('clean_text', ''), 150 if not full else 350)}")

        ok("Qdrant inspection done")

    except Exception as e:
        err(f"Qdrant error: {e}")


# ─── Elasticsearch ────────────────────────────────────────────────────────────
def inspect_elasticsearch(doc_id: str | None = None, full: bool = False):
    hdr("Elasticsearch — Keyword DB")
    try:
        from elasticsearch import Elasticsearch
        from config import ES_URL, ES_INDEX

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        client  = Elasticsearch(ES_URL, headers=headers)

        # ── Index stats
        sub("📦 Index Stats")
        stats = client.indices.stats(index=ES_INDEX)
        count = stats["indices"][ES_INDEX]["total"]["docs"]["count"]
        size  = stats["indices"][ES_INDEX]["total"]["store"]["size_in_bytes"]
        print(f"  index        : {ES_INDEX}")
        print(f"  total docs   : {count}")
        print(f"  store size   : {size / 1024:.1f} KB")

        if count == 0:
            dim("  (index is empty)")
            return

        # ── Mapping check
        sub("📋 Index Mapping Fields")
        mapping = client.indices.get_mapping(index=ES_INDEX)
        props   = mapping[ES_INDEX]["mappings"].get("properties", {})
        for field, meta in props.items():
            print(f"  {field:<30} type={meta.get('type', '?')}")

        # ── Sample docs
        sub(f"🔍 Sample Documents ({'doc_id=' + doc_id if doc_id else 'all'})")
        query_body = {"query": {"match_all": {}}, "size": 5}
        if doc_id:
            query_body = {"query": {"term": {"doc_id": doc_id}}, "size": 5}

        resp = client.search(index=ES_INDEX, body=query_body)
        hits = resp["hits"]["hits"]

        for i, h in enumerate(hits):
            s = h["_source"]
            print(f"\n  {BOLD}── Doc #{i+1} ──{RESET}")
            print(f"  _id (chunk_id)  : {h['_id']}")
            print(f"  doc_id          : {s.get('doc_id', '?')}")
            print(f"  level           : {s.get('level', '?')} | seq_no: {s.get('seq_no', '?')}")
            print(f"  title           : {s.get('title') or '(empty)'}")
            print(f"  summary         : {trunc(s.get('summary', ''), 120 if not full else 250)}")
            print(f"  clean_text      : {trunc(s.get('clean_text', ''), 150 if not full else 350)}")
            kws = s.get('keywords', [])
            hqs = s.get('hypothetical_questions', [])
            print(f"  keywords        : {kws[:5]}")
            if hqs:
                print(f"  hypo_questions  : {hqs[:2]}")

        ok("Elasticsearch inspection done")

    except Exception as e:
        err(f"Elasticsearch error: {e}")


# ─── Neo4j ────────────────────────────────────────────────────────────────────
def inspect_neo4j(doc_id: str | None = None):
    hdr("Neo4j — Graph DB")
    try:
        from neo4j import GraphDatabase
        from config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD

        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()

        with driver.session() as session:

            # ── Node counts
            sub("📦 Node & Relationship Counts")
            for label in ["Document", "Chunk", "Entity"]:
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()
                print(f"  {label:<12}: {r['cnt']} nodes")

            for rel in ["BELONGS_TO", "PARENT_OF", "NEXT", "MENTIONS", "RELATES_TO"]:
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt").single()
                print(f"  [{rel:<14}]: {r['cnt']} edges")

            # ── Documents
            sub("📄 Documents in Graph")
            docs = session.run("MATCH (d:Document) RETURN d.doc_id AS id, d.file_name AS name LIMIT 10").data()
            if not docs:
                dim("  (no documents)")
            else:
                for d in docs:
                    print(f"  {d['id']} — {d['name']}")

            # ── Sample chunks
            sub(f"🔍 Sample Chunks ({'doc_id=' + doc_id if doc_id else 'all'})")
            filter_clause = "WHERE c.doc_id = $doc_id" if doc_id else ""
            rows = session.run(
                f"""
                MATCH (c:Chunk) {filter_clause}
                WHERE c.level = 2
                RETURN c.chunk_id AS id, c.title AS title,
                       c.summary AS summary, c.seq_no AS seq
                ORDER BY c.seq_no LIMIT 5
                """,
                doc_id=doc_id,
            ).data()
            for r in rows:
                print(f"\n  chunk_id : {r['id']}")
                print(f"  seq_no   : {r['seq']}")
                print(f"  title    : {r['title'] or '(empty)'}")
                print(f"  summary  : {trunc(r['summary'] or '', 120)}")

            # ── Top entities
            sub("🏷️  Top 10 Entities (by mention count)")
            entities = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WITH e, count(c) AS cnt
                ORDER BY cnt DESC LIMIT 10
                RETURN e.name AS name, e.type AS type, cnt
                """
            ).data()
            if not entities:
                dim("  (no entities found)")
            else:
                for e in entities:
                    print(f"  [{e['type']:<12}] {e['name']:<30} — {e['cnt']} mentions")

            # ── Sample relations
            sub("🔗 Sample Entity Relations (5)")
            rels = session.run(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                RETURN a.name AS from, r.relation_type AS rel, b.name AS to
                LIMIT 5
                """
            ).data()
            if not rels:
                dim("  (no relations found)")
            else:
                for r in rels:
                    print(f"  {r['from']} —[{r['rel']}]→ {r['to']}")

        driver.close()
        ok("Neo4j inspection done")

    except Exception as e:
        err(f"Neo4j error: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Inspect RAG databases")
    parser.add_argument("--pg",     action="store_true", help="PostgreSQL only")
    parser.add_argument("--qdrant", action="store_true", help="Qdrant only")
    parser.add_argument("--es",     action="store_true", help="Elasticsearch only")
    parser.add_argument("--neo4j",  action="store_true", help="Neo4j only")
    parser.add_argument("--full",   action="store_true", help="Show full text (no truncation)")
    parser.add_argument("--doc",    type=str, default=None, help="Filter by doc_id UUID")
    args = parser.parse_args()

    all_dbs = not any([args.pg, args.qdrant, args.es, args.neo4j])

    print(f"\n{BOLD}🔍 RAG Database Inspector{RESET}")
    if args.doc:
        print(f"   Filter: doc_id = {args.doc}")
    if args.full:
        print(f"   Mode: FULL text (no truncation)")

    if all_dbs or args.pg:
        await inspect_postgres(args.doc, args.full)

    if all_dbs or args.qdrant:
        inspect_qdrant(args.doc, args.full)

    if all_dbs or args.es:
        inspect_elasticsearch(args.doc, args.full)

    if all_dbs or args.neo4j:
        inspect_neo4j(args.doc)

    print(f"\n{BOLD}{GREEN}{'═'*60}")
    print(f"  ✅ Inspection complete!")
    print(f"{'═'*60}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
