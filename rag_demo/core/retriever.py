# core/retriever.py — Search Pipeline
"""
Pipeline:
  1. Query Rewrite  — LLM viết lại thành 3 phiên bản
  2. Parallel Search — Qdrant + ES + Neo4j (top_k mỗi DB)
  3. RRF Merge      — Gộp → top FINAL_TOP_K bằng Reciprocal Rank Fusion
  4. Context Expand — Lấy L2 full text từ PostgreSQL (dùng cho LLM)
  5. Parent Sources — Gom L2 → L1 chunk cha (dùng cho hiển thị)
  6. Generate       — LLM sinh câu trả lời
  7. Log            — Ghi search_logs
"""

import asyncio
import json
import re
import time

from db.meta_db    import MetaDB
from db.vector_db  import VectorDB
from db.keyword_db import KeywordDB
from db.graph_db   import GraphDB
from core.embedder import embed_query
from llm.client    import AsyncLLMClient
from llm.prompts   import (QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER,
                            ANSWER_SYSTEM, ANSWER_USER)
from config        import SEARCH_TOP_K, FINAL_TOP_K, RRF_K, SOURCE_MIN_RRF

SOURCE_DISPLAY_MIN_RRF = max(SOURCE_MIN_RRF, 0.02)


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: Query Rewrite
# ─────────────────────────────────────────────────────────────────────────────
async def rewrite_query(query: str, llm: AsyncLLMClient) -> dict:
    try:
        raw = await llm.complete(
            system=QUERY_REWRITE_SYSTEM,
            user=QUERY_REWRITE_USER.format(query=query),
            max_tokens=300,
        )
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r'\s*```$', '', raw.strip())
        data = json.loads(raw)
        return {
            "original":  data.get("original",  query),
            "technical": data.get("technical", query),
            "keywords":  data.get("keywords",  query),
        }
    except Exception:
        return {"original": query, "technical": query, "keywords": query}


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: Parallel Search
# ─────────────────────────────────────────────────────────────────────────────
def search_qdrant(vector_db: VectorDB, query_text: str, top_k: int) -> list:
    return vector_db.search(embed_query(query_text), top_k=top_k)

def search_es(kw_db: KeywordDB, query_text: str, top_k: int) -> list:
    return kw_db.search(query_text, top_k=top_k)

def search_neo4j(graph_db: GraphDB, query_text: str, top_k: int) -> list:
    entities = graph_db.extract_entities_simple(query_text)
    return graph_db.search(entities, top_k=top_k) if entities else []


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: RRF Merge
# ─────────────────────────────────────────────────────────────────────────────
def rrf_merge(results_per_db: list, k: int = RRF_K, top_n: int = FINAL_TOP_K) -> list:
    """
    Gộp kết quả từ nhiều DB bằng Reciprocal Rank Fusion.
    Chỉ giữ chunk_id + title để display — full text fetch từ PostgreSQL sau.
    """
    scores: dict[str, float] = {}
    meta:   dict[str, dict]  = {}

    for db_results in results_per_db:
        for rank, item in enumerate(db_results, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in meta:
                meta[cid] = {
                    "chunk_id": cid,
                    "title":    item.get("title", ""),
                    "sources":  [],
                }
            meta[cid]["sources"].append(item.get("source", "?"))

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        {**meta[cid], "rrf_score": round(scores[cid], 6)}
        for cid in ranked[:top_n]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: Context Expand  (fetch full text từ PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────
async def expand_context(top_chunks: list, meta_db: MetaDB) -> list:
    chunk_ids = [c["chunk_id"] for c in top_chunks]
    pg_rows   = await meta_db.get_context(chunk_ids)
    pg_map    = {r["chunk_id"]: r for r in pg_rows}

    extra_ids = set()
    for row in pg_rows:
        if (row.get("token_count") or 999) < 150:
            for fld in ("prev_id", "next_id"):
                if row.get(fld):
                    extra_ids.add(row[fld])
    new_ids = [eid for eid in extra_ids if eid not in pg_map]
    if new_ids:
        for r in await meta_db.get_context(new_ids):
            pg_map[r["chunk_id"]] = r

    final, seen = [], set()
    for chunk in top_chunks:
        cid = chunk["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        pg = pg_map.get(cid, {})
        entry = {
            "chunk_id":     cid,
            "title":        pg.get("title")      or chunk.get("title", ""),
            "clean_text":   pg.get("clean_text", ""),
            "summary":      pg.get("summary", ""),
            "parent_id":    pg.get("parent_id", ""),
            "parent_title": pg.get("parent_title", ""),
            "seq_no":       pg.get("seq_no", ""),
            "source_file":  pg.get("source_file", ""),
            "rrf_score":    chunk.get("rrf_score", 0),
            "sources":      chunk.get("sources", []),
        }
        if (pg.get("token_count") or 999) < 150:
            for fld in ("prev_id", "next_id"):
                extra_id = pg.get(fld)
                if extra_id and extra_id not in seen:
                    extra_pg = pg_map.get(extra_id, {})
                    if extra_pg.get("clean_text"):
                        entry["clean_text"] += "\n\n" + extra_pg["clean_text"]
                        seen.add(extra_id)
        final.append(entry)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: Parent Sources  (gom L2 → L1, dùng cho hiển thị)
# ─────────────────────────────────────────────────────────────────────────────
def _heading_from_text(text: str) -> str:
    if not text:
        return ""
    first = text.strip().split('\n')[0].strip()
    return first[:120]


async def get_parent_sources(top_chunks: list, expanded: list,
                              meta_db: MetaDB) -> list:
    expanded_map = {c["chunk_id"]: c for c in expanded}
    parent_groups: dict[str, dict] = {}

    for chunk in top_chunks:
        cid  = chunk["chunk_id"]
        info = expanded_map.get(cid, chunk)
        pid  = info.get("parent_id")

        if not pid:
            continue

        if pid not in parent_groups:
            parent_groups[pid] = {
                "parent_id":   pid,
                "matched_l2":  [],
                "sources":     set(),
                "best_rrf":    0.0,
                "source_file": info.get("source_file", ""),
            }

        l2_title = (info.get("title") or chunk.get("title", "")).strip()
        if l2_title:
            parent_groups[pid]["matched_l2"].append({
                "title":  l2_title,
                "seq_no": info.get("seq_no", ""),
            })
        parent_groups[pid]["sources"].update(chunk.get("sources", []))
        parent_groups[pid]["best_rrf"] = max(
            parent_groups[pid]["best_rrf"],
            chunk.get("rrf_score", 0.0),
        )

    top1_rrf = max((g["best_rrf"] for g in parent_groups.values()), default=0)

    parent_rows = await meta_db.get_parent_chunks(list(parent_groups.keys()))
    parent_map  = {r["chunk_id"]: r for r in parent_rows}

    result = []
    for pid, group in parent_groups.items():
        rrf = group["best_rrf"]

        if rrf < SOURCE_DISPLAY_MIN_RRF and rrf < top1_rrf:
            continue

        p = parent_map.get(pid, {})

        content = (p.get("clean_text") or "").strip()
        if not content:
            continue

        title = p.get("title") or _heading_from_text(content)
        if not title:
            continue

        seen_t: set[str] = set()
        unique_l2 = []
        for m in group["matched_l2"]:
            t = m.get("title", "").strip()
            if t and t not in seen_t:
                seen_t.add(t)
                unique_l2.append(m)

        result.append({
            "parent_id":   pid,
            "title":       title,
            "clean_text":  content,
            "seq_no":      p.get("seq_no", ""),
            "source_file": group["source_file"] or p.get("source_file", ""),
            "sources":     sorted(group["sources"]),
            "best_rrf":    rrf,
            "matched_l2":  unique_l2,
        })

    def _seq_key(e):
        try:
            return int(e["seq_no"])
        except (ValueError, TypeError):
            return 9999

    result.sort(key=_seq_key)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: Generate Answer
# ─────────────────────────────────────────────────────────────────────────────
async def generate_answer(query: str, context_chunks: list,
                           llm: AsyncLLMClient) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        title       = chunk.get("title")       or f"Đoạn {i}"
        parent      = chunk.get("parent_title") or ""
        source_file = chunk.get("source_file")  or ""
        text        = chunk.get("clean_text", "").strip()

        meta_line = f"[{i}] {title}"
        if parent:
            meta_line += f" | Phần: {parent}"
        if source_file:
            meta_line += f" | File: {source_file}"

        context_parts.append(f"{meta_line}\n{text}")

    return await llm.complete(
        system=ANSWER_SYSTEM,
        user=ANSWER_USER.format(query=query, context="\n\n---\n\n".join(context_parts)),
        max_tokens=1200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hiển thị nguồn
# ─────────────────────────────────────────────────────────────────────────────
def _print_sources(parent_sources: list):
    n = len(parent_sources)
    print(f"\n{'─'*60}")
    print(f"  📚 NGUỒN DỮ LIỆU ({n} chương):")
    print(f"{'─'*60}")

    for i, s in enumerate(parent_sources, 1):
        title       = s["title"]
        source_file = s.get("source_file", "")
        dbs         = " + ".join(s.get("sources", []))
        rrf         = s.get("best_rrf", 0)
        matched     = s.get("matched_l2", [])
        text        = s.get("clean_text", "").strip()

        print(f"\n  [{i}] {title}")
        if source_file:
            print(f"       📄 File     : {source_file}")
        if dbs:
            print(f"       🗄  Tìm qua  : {dbs}  (rrf={rrf:.5f})")
        if matched:
            display = "  |  ".join(f'"{m["title"][:50]}"' for m in matched[:4])
            print(f"       📌 Đoạn khớp: {display}")

        preview_lines = [ln.strip() for ln in text.split('\n') if ln.strip()][:2]
        preview = "  /  ".join(preview_lines)[:180]
        if len(text) > 200:
            preview += "..."
        print(f"       📖 Nội dung : \"{preview}\"")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def search(query: str, verbose: bool = True) -> dict:
    t0 = time.time()

    meta_db   = MetaDB()
    vector_db = VectorDB()
    kw_db     = KeywordDB()
    graph_db  = GraphDB()
    llm       = AsyncLLMClient()

    await meta_db.connect()
    vector_db.connect()
    kw_db.connect()
    graph_db.connect()

    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  🔍 Query: {query}")
            print(f"{'='*60}")

        if verbose:
            print("\n[1/5] Query rewrite...")
        variants = await rewrite_query(query, llm)
        if verbose:
            print(f"  original  : {variants['original']}")
            print(f"  technical : {variants['technical']}")
            print(f"  keywords  : {variants['keywords']}")

        if verbose:
            print("\n[2/5] Search song song (Qdrant + ES + Neo4j)...")
        qdrant_res, es_res, neo4j_res = await asyncio.gather(
            asyncio.to_thread(search_qdrant, vector_db, variants["technical"], SEARCH_TOP_K),
            asyncio.to_thread(search_es,     kw_db,     variants["original"],  SEARCH_TOP_K),
            asyncio.to_thread(search_neo4j,  graph_db,  variants["keywords"],  SEARCH_TOP_K),
        )
        if verbose:
            print(f"  Qdrant: {len(qdrant_res)}  ES: {len(es_res)}  Neo4j: {len(neo4j_res)}")

        if verbose:
            print("\n[3/5] RRF merge...")
        top_chunks = rrf_merge([qdrant_res, es_res, neo4j_res])
        if verbose:
            print(f"  Top {len(top_chunks)} chunks sau RRF:")
            for i, c in enumerate(top_chunks, 1):
                srcs = "+".join(c["sources"])
                print(f"  {i}. rrf={c['rrf_score']:.5f} [{srcs}]  {c['title'][:50]!r}")

        if verbose:
            print("\n[4/5] Context expand + generate...")
        expanded       = await expand_context(top_chunks, meta_db)
        parent_sources = await get_parent_sources(top_chunks, expanded, meta_db)
        llm_context    = parent_sources if parent_sources else expanded
        answer         = await generate_answer(query, llm_context, llm)

        latency_ms = int((time.time() - t0) * 1000)

        await meta_db.log_search({
            "query_original":   query,
            "query_rewritten":  variants,
            "chunks_retrieved": [
                {"chunk_id": c["chunk_id"], "rrf_score": c["rrf_score"],
                 "sources":  c["sources"]}
                for c in top_chunks
            ],
            "llm_response": answer,
            "latency_ms":   latency_ms,
        })

        if verbose:
            print(f"\n{'='*60}")
            print(f"  💬 CÂU TRẢ LỜI ({latency_ms}ms):")
            print(f"{'='*60}")
            print(answer)
            _print_sources(parent_sources)

        return {
            "answer":         answer,
            "query_variants": variants,
            "top_chunks":     expanded,
            "parent_sources": parent_sources,
            "latency_ms":     latency_ms,
        }

    finally:
        await meta_db.close()
        graph_db.close()
