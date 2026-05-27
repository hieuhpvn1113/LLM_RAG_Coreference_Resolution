# core/retriever.py — Search Pipeline
"""
Pipeline:
  1. Query Rewrite  — LLM viết lại thành 3 phiên bản
  2. Parallel Search — Qdrant + ES + Neo4j (top_k mỗi DB)
  3. RRF Merge      — Gộp → top FINAL_TOP_K bằng Reciprocal Rank Fusion
  4. Resolve L1     — Lấy parent_id từ L2 chunks → fetch L1 raw_text từ PostgreSQL
  5. Generate       — LLM sinh câu trả lời từ L1 raw_text (full, không trim)
  6. Log            — Ghi search_logs
"""

import asyncio
import json
import re
import time
import unicodedata

from db.meta_db    import MetaDB
from db.vector_db  import VectorDB
from db.graph_db   import GraphDB
from core.embedder import embed_query
from core.self_query import self_query
from llm.client    import AsyncLLMClient
from llm.prompts   import (QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER,
                            ANSWER_SYSTEM, ANSWER_USER)
from config        import SEARCH_TOP_K, RRF_TOP_K, FINAL_TOP_K, RRF_K, SOURCE_MIN_RRF


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _score_parent_for_query(query: str, entry: dict) -> float:
    q = _normalize_text(query)
    title = _normalize_text(entry.get("title", ""))
    text = _normalize_text(entry.get("raw_text", ""))
    matched_titles = " ".join(m.get("title", "") for m in entry.get("matched_l2", []))
    matched_titles = _normalize_text(matched_titles)

    strong_phrases = [
        "yeu to",
        "hai yeu to",
        "chi phi hoat dong",
        "doanh thu tang truong",
        "bien loi nhuan",
    ]
    score = 0.0
    for phrase in strong_phrases:
        if phrase in text:
            score += 5.0
        if phrase in title:
            score += 3.0
        if phrase in matched_titles:
            score += 2.0

    for token in set(q.split()):
        if len(token) < 3:
            continue
        if token in text:
            score += 0.15
        if token in title:
            score += 0.30
        if token in matched_titles:
            score += 0.20

    score += float(entry.get("best_rrf", 0.0)) * 20.0
    return score


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
def search_qdrant(vector_db: VectorDB, query_text: str, top_k: int, qdrant_filter=None) -> list:
    return vector_db.search(
        query_vector=embed_query(query_text),
        query_text=query_text,
        top_k=top_k,
        qdrant_filter=qdrant_filter,
    )

def search_neo4j(graph_db: GraphDB, query_text: str, top_k: int) -> list:
    entities = graph_db.extract_entities_simple(query_text)
    return graph_db.search(entities, top_k=top_k) if entities else []


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: RRF Merge
# ─────────────────────────────────────────────────────────────────────────────
def rrf_merge(results_per_db: list, k: int = RRF_K, top_n: int = RRF_TOP_K) -> list:
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


def merge_with_qdrant_guard(rrf_items: list, qdrant_items: list, top_n: int) -> list:
    """
    Giữ ổn định chất lượng Qdrant:
    - Lấy danh sách RRF trước
    - Nếu thiếu item thì bơm thêm từ top Qdrant theo thứ tự gốc
    """
    out = list(rrf_items)
    seen = {x.get("chunk_id") for x in out}
    for q in qdrant_items:
        if len(out) >= top_n:
            break
        cid = q.get("chunk_id")
        if not cid or cid in seen:
            continue
        out.append({
            "chunk_id": cid,
            "title": q.get("title", ""),
            "sources": [q.get("source", "qdrant_hybrid")],
            "rrf_score": 0.0,
            "parent_id": q.get("parent_id", ""),
        })
        seen.add(cid)
    return out[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: Resolve L1 — lấy parent_id từ L2, fetch L1 raw_text từ PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────
async def resolve_l1_context(top_chunks: list, meta_db: MetaDB) -> tuple[list, list]:
    """
    Nhận top L2 chunks sau RRF.
    1. Fetch metadata L2 (chỉ lấy parent_id, không lấy text L2).
    2. Suy ra danh sách parent_id (L1).
    3. Fetch L1 raw_text từ PostgreSQL.
    4. Trả về (l2_meta_list, parent_sources).
    """
    chunk_ids = [c["chunk_id"] for c in top_chunks]
    l2_rows   = await meta_db.get_context(chunk_ids)
    l2_map    = {r["chunk_id"]: r for r in l2_rows}

    # Gom nhóm L2 theo parent_id
    parent_groups: dict[str, dict] = {}
    for chunk in top_chunks:
        cid  = chunk["chunk_id"]
        info = l2_map.get(cid, {})
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

    # Fetch L1 raw_text
    parent_rows = await meta_db.get_parent_chunks(list(parent_groups.keys()))
    parent_map  = {r["chunk_id"]: r for r in parent_rows}

    def _heading_from_text(text: str) -> str:
        if not text:
            return ""
        return text.strip().split('\n')[0].strip()[:120]

    result = []
    for pid, group in parent_groups.items():
        rrf = group["best_rrf"]
        p = parent_map.get(pid, {})
        content = (p.get("raw_text") or "").strip()
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
            "raw_text":    content,            # ← dùng raw_text, không phải clean_text
            "seq_no":      p.get("seq_no", ""),
            "source_file": group["source_file"] or p.get("source_file", ""),
            "sources":     sorted(group["sources"]),
            "best_rrf":    rrf,
            "matched_l2":  unique_l2,
        })

    return l2_rows, result


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: Generate Answer — dùng L1 raw_text full, không trim
# ─────────────────────────────────────────────────────────────────────────────
async def generate_answer(query: str, context_chunks: list,
                           llm: AsyncLLMClient) -> str:
    """
    Đưa toàn bộ raw_text của L1 vào context LLM, không cắt bớt.
    LLM đọc đúng văn bản gốc → hiểu ngữ cảnh chính xác nhất.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        title       = chunk.get("title")       or f"Đoạn {i}"
        source_file = chunk.get("source_file") or ""
        text        = chunk.get("raw_text", "").strip()   # ← raw_text full, không trim

        meta_line = f"[{i}] {title}"
        if source_file:
            meta_line += f" | File: {source_file}"

        context_parts.append(f"{meta_line}\n{text}")

    return await llm.complete(
        system=ANSWER_SYSTEM,
        user=ANSWER_USER.format(query=query, context="\n\n---\n\n".join(context_parts)),
        max_tokens=1200,
    )


def _is_metric_query(query: str) -> bool:
    q = _normalize_text(query)
    keys = ["roic", "roe", "roa", "ebitda", "margin", "ttm", "12 thang", "31 03 2025", "%"]
    return any(k in q for k in keys) or bool(re.search(r"\d", q))


def _print_db_results(name: str, items: list):
    print(f"  {name} top {len(items)}:")
    for i, item in enumerate(items, 1):
        cid = item.get("chunk_id", "")
        title = (item.get("title", "") or "")[:50]
        score = item.get("score", 0.0)
        try:
            score_str = f"{float(score):.5f}"
        except Exception:
            score_str = str(score)
        print(f"  {i}. score={score_str} chunk_id={cid}  {title!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Hiển thị nguồn
# ─────────────────────────────────────────────────────────────────────────────
def _print_sources(parent_sources: list):
    n = len(parent_sources)
    print(f"\n{'─'*60}")
    print(f"  📚 NGUỒN DỮ LIỆU ({n} chương L1):")
    print(f"{'─'*60}")

    for i, s in enumerate(parent_sources, 1):
        title       = s["title"]
        source_file = s.get("source_file", "")
        dbs         = " + ".join(s.get("sources", []))
        rrf         = s.get("best_rrf", 0)
        matched     = s.get("matched_l2", [])
        text        = s.get("raw_text", "").strip()

        print(f"\n  [{i}] {title}")
        if source_file:
            print(f"       📄 File     : {source_file}")
        if dbs:
            print(f"       🗄  Tìm qua  : {dbs}  (rrf={rrf:.5f})")
        if matched:
            display = "  |  ".join(f'"{m["title"][:50]}"' for m in matched[:4])
            print(f"       📌 L2 khớp  : {display}")

        preview_lines = [ln.strip() for ln in text.split('\n') if ln.strip()][:2]
        preview = "  /  ".join(preview_lines)[:180]
        if len(text) > 200:
            preview += "..."
        print(f"       📖 raw_text : \"{preview}\"")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def search(query: str, verbose: bool = True) -> dict:
    t0 = time.time()

    meta_db   = MetaDB()
    vector_db = VectorDB()
    graph_db  = GraphDB()
    llm       = AsyncLLMClient()

    await meta_db.connect()
    vector_db.connect()
    graph_db.connect()

    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  🔍 Query: {query}")
            print(f"{'='*60}")

        # ── Bước 1: Query Rewrite ──────────────────────────────────────────
        if verbose:
            print("\n[1/5] Query rewrite...")
        variants = await rewrite_query(query, llm)
        if verbose:
            print(f"  original  : {variants['original']}")
            print(f"  technical : {variants['technical']}")
            print(f"  keywords  : {variants['keywords']}")

        # ── Bước 2: Self-query + Parallel Search ───────────────────────────
        if verbose:
            print("\n[2/5] Self-query + Search song song (Qdrant + Neo4j)...")
        sq = await self_query(query, llm)
        qdrant_query = sq.get("search_query") or variants["technical"]
        qdrant_filter = sq.get("qdrant_filter")
        if verbose:
            print(f"  self_query.search_query: {qdrant_query}")
            print(f"  self_query.filter_mode : {sq.get('filter_mode', 'none')}")
            print(f"  self_query.filter_json : {json.dumps(sq.get('raw_parsed', {}), ensure_ascii=False)}")
        qdrant_res, neo4j_res = await asyncio.gather(
            asyncio.to_thread(search_qdrant, vector_db, qdrant_query, SEARCH_TOP_K, qdrant_filter),
            asyncio.to_thread(search_neo4j,  graph_db,  variants["keywords"],  SEARCH_TOP_K),
        )
        if _is_metric_query(query):
            for item in neo4j_res:
                item["score"] = float(item.get("score", 0.0)) * 0.6
                item["source"] = "neo4j_low_weight"
        if verbose:
            print(f"  Qdrant: {len(qdrant_res)}  Neo4j: {len(neo4j_res)}")
            _print_db_results("Qdrant", qdrant_res)
            _print_db_results("Neo4j", neo4j_res)

        # ── Bước 3: RRF Merge ──────────────────────────────────────────────
        if verbose:
            print("\n[3/5] RRF merge...")
        top_chunks = rrf_merge([qdrant_res, neo4j_res])
        top_chunks = merge_with_qdrant_guard(top_chunks, qdrant_res, RRF_TOP_K)
        # Enrich parent_id cho L2 chunks từ MetaDB nếu payload search chưa có.
        l2_rrf_meta = await meta_db.get_context(
            [c.get("chunk_id", "") for c in top_chunks if c.get("chunk_id")]
        )
        l2_rrf_map = {r.get("chunk_id"): r for r in l2_rrf_meta}
        for c in top_chunks:
            if not c.get("parent_id"):
                c["parent_id"] = l2_rrf_map.get(c.get("chunk_id"), {}).get("parent_id", "")
        if verbose:
            print(f"  Top {len(top_chunks)} L2 chunks sau RRF:")
            for i, c in enumerate(top_chunks, 1):
                print(
                    f"  {i}. rrf={c.get('rrf_score', 0):.5f}  "
                    f"chunk_id={c.get('chunk_id', '')}  "
                    f"parent_id={c.get('parent_id', '')}"
                )

        # ── Bước 4: Resolve L1 ─────────────────────────────────────────────
        if verbose:
            print("\n[4/5] Resolve L1 từ parent_id...")
        l2_meta, parent_sources = await resolve_l1_context(top_chunks, meta_db)

        if verbose:
            print("  Map L2 chunk_id → parent_id (L1):")
            l2_map = {r["chunk_id"]: r for r in l2_meta}
            for i, c in enumerate(top_chunks, 1):
                info = l2_map.get(c.get("chunk_id"), {})
                pid  = info.get("parent_id", "")
                print(f"  {i}. L2={c.get('chunk_id')}  →  L1={pid or 'N/A'}")

        # Rank L1 theo query, lấy top FINAL_TOP_K
        ranked_parents = sorted(
            parent_sources,
            key=lambda e: _score_parent_for_query(query, e),
            reverse=True,
        )
        llm_context    = ranked_parents[:FINAL_TOP_K]
        parent_sources = ranked_parents   # full list dùng cho hiển thị

        if verbose:
            print(f"\n  -> {len(parent_sources)} L1 chunks sau dedupe theo parent_id:")
            for i, ctx in enumerate(parent_sources, 1):
                t   = (ctx.get("title") or "").strip()
                sf  = (ctx.get("source_file") or "").strip()
                txt = (ctx.get("raw_text") or "").strip().replace("\n", " ")
                print(f"  [{i}] {t[:120]!r} | file={sf}")
                print(f"      preview: {txt[:280]}")
            print(f"  -> Dung top {len(llm_context)} L1 cho LLM (FINAL_TOP_K={FINAL_TOP_K})")

        # ── Bước 5: Generate ───────────────────────────────────────────────
        if verbose:
            print("\n[5/5] Generate answer...")
        answer     = await generate_answer(query, llm_context, llm)

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
            "top_chunks":     top_chunks,
            "parent_sources": parent_sources,
            "latency_ms":     latency_ms,
        }

    finally:
        await meta_db.close()
        graph_db.close()
