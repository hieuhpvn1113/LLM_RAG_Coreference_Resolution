# core/ingestor.py — Orchestrate toàn bộ pipeline ingest
"""
Pipeline:
  1.  Đọc + clean file text (auto-detect encoding)
  2.  Check duplicate
  3.  Insert Document (status='processing')
  4.  Hierarchical Split → Level 1 Sections
  5.  Insert L1 chunks (null links trước, update links sau) + Neo4j
  6.  Semantic Split (embedding) → Level 2 Paragraphs
  7.  Insert L2 chunks (null links trước, update links sau)
  8.  LLM Enrichment: title, summary, keywords, entities, hypo_questions
  9.  Update enrichment vào PostgreSQL
  10. Embed Level 2 chunks (intfloat/multilingual-e5-base, batch)
  11. Write → Qdrant + Elasticsearch + Neo4j
  12. Finalize (status='ready')
"""

import asyncio
import time
from pathlib import Path

from core.chunker  import hierarchical_split, semantic_split, clean_text
from core.enricher import enrich_chunk
from core.embedder import embed_batch
from db.meta_db    import MetaDB
from db.vector_db  import VectorDB
from db.keyword_db import KeywordDB
from db.graph_db   import GraphDB
from llm.client    import AsyncLLMClient
from config        import EMBED_MODEL


def _read_text_smart(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ('utf-8-sig', 'utf-8'):
        try:
            text = raw.decode(enc)
            print(f"  Encoding detected: {enc}")
            return text
        except UnicodeDecodeError:
            pass
    try:
        import chardet
        result = chardet.detect(raw)
        detected = result.get('encoding') or ''
        confidence = result.get('confidence', 0)
        print(f"  chardet: {detected} (confidence={confidence:.0%})")
        if detected and confidence > 0.7:
            try:
                text = raw.decode(detected)
                print(f"  Encoding detected: {detected}")
                return text
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        print("  chardet chưa cài — thử các encoding phổ biến")
    for enc in ('cp1258', 'cp1252', 'latin-1'):
        try:
            text = raw.decode(enc)
            print(f"  Encoding detected (fallback): {enc}")
            return text
        except (UnicodeDecodeError, LookupError):
            pass
    print("  ⚠️  WARNING: Không xác định được encoding — dùng utf-8 errors=replace")
    return raw.decode('utf-8', errors='replace')


async def _insert_chunks_then_link(meta_db: MetaDB, chunks: list):
    """
    Insert tất cả chunks với prev_id=None / next_id=None trước,
    sau đó mới update_chunk_links — tránh ForeignKeyViolationError.
    """
    for chunk in chunks:
        await meta_db.insert_chunk({**chunk, 'prev_id': None, 'next_id': None})
    for chunk in chunks:
        if chunk.get('prev_id') or chunk.get('next_id'):
            await meta_db.update_chunk_links(
                chunk['chunk_id'],
                chunk.get('prev_id'),
                chunk.get('next_id'),
            )


async def ingest_file(file_path: str, force: bool = False) -> str:
    start_time = time.time()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    print(f"\n{'='*60}")
    print(f"  📄 Ingest: {path.name}")
    print(f"{'='*60}")

    meta_db   = MetaDB()
    vector_db = VectorDB()
    kw_db     = KeywordDB()
    graph_db  = GraphDB()
    llm       = AsyncLLMClient()

    await meta_db.connect()
    vector_db.connect();   vector_db.ensure_collection()
    kw_db.connect();       kw_db.ensure_index()
    graph_db.connect();    graph_db.ensure_constraints()

    try:
        # ── 1. Đọc file ───────────────────────────────────────────────────────
        print("\n[1/7] Đọc file...")
        raw_text = _read_text_smart(path)
        text     = clean_text(raw_text)
        print(f"  File size : {path.stat().st_size:,} bytes")
        print(f"  Text len  : {len(text):,} chars")
        if text.count('\ufffd') > 0:
            print(f"  ⚠️  Có {text.count(chr(65533))} ký tự lỗi sau decode!")

        # ── 2. Check duplicate ────────────────────────────────────────────────
        if not force:
            existing = await meta_db.get_document_by_filename(path.name)
            if existing:
                print(f"\n⚠️  File '{path.name}' đã được ingest rồi!")
                print(f"   doc_id      : {existing['doc_id']}")
                print(f"   total_chunks: {existing['total_chunks']}")
                print(f"   created_at  : {existing['created_at']}")
                print(f"\n   Dùng --force để ingest lại.")
                return existing['doc_id']

        # ── 3. Document record ────────────────────────────────────────────────
        print("\n[2/7] Tạo document record...")
        doc_id = await meta_db.create_document(path.name, str(path.resolve()))
        graph_db.upsert_document(doc_id, path.name)
        print(f"  doc_id: {doc_id}")

        # ── 4+5. Hierarchical Split L1, insert + link ─────────────────────────
        print("\n[3/7] Hierarchical split (Level 1 — Sections)...")
        l1_chunks = hierarchical_split(text, doc_id, path.name)
        print(f"  → {len(l1_chunks)} sections")

        for chunk in l1_chunks:
            await meta_db.insert_chunk({**chunk, 'prev_id': None, 'next_id': None})
            graph_db.upsert_chunk_node(chunk)
        for chunk in l1_chunks:
            if chunk.get('prev_id') or chunk.get('next_id'):
                await meta_db.update_chunk_links(
                    chunk['chunk_id'],
                    chunk.get('prev_id'),
                    chunk.get('next_id'),
                )

        # ── 6+7. Semantic Split L2 (embedding), insert + link ─────────────────
        print("\n[4/7] Semantic split (Level 2 — Paragraphs, embedding)...")
        all_l2_chunks = []
        for i, section in enumerate(l1_chunks, 1):
            l2 = semantic_split(section, doc_id)
            print(f"  Section {i:2d}/{len(l1_chunks)}: "
                  f"{section['token_count']:4d} tokens → {len(l2)} paragraphs")
            all_l2_chunks.extend(l2)
        print(f"  → Tổng: {len(all_l2_chunks)} paragraphs (Level 2)")

        await _insert_chunks_then_link(meta_db, all_l2_chunks)

        # ── 8+9. LLM Enrichment ───────────────────────────────────────────────
        # Groq free tier: 30 req/phút.
        # Delay tối thiểu (2.2s) được xử lý bên trong enrich_chunk().
        # Không cần sleep thủ công ở đây nữa — tránh cộng dồn delay không cần thiết.
        print(f"\n[5/7] LLM Enrichment ({len(all_l2_chunks)} chunks)...")
        enriched_chunks = []
        for i, chunk in enumerate(all_l2_chunks, 1):
            print(f"  [{i:2d}/{len(all_l2_chunks)}] {chunk['token_count']:4d} tok...",
                  end='', flush=True)
            enrichment = await enrich_chunk(chunk['clean_text'], llm)
            chunk.update({
                'title':                  enrichment['title'],
                'summary':                enrichment['summary'],
                'keywords':               enrichment['keywords'],
                'entities':               enrichment['entities'],
                'relations':              enrichment['relations'],
                'hypothetical_questions': enrichment['hypothetical_questions'],
            })
            enriched_chunks.append(chunk)
            await meta_db.update_enrichment(chunk['chunk_id'], enrichment)
            print(f" ✓  \"{enrichment['title'][:45]}\"")

        # ── 10. Embedding ─────────────────────────────────────────────────────
        print(f"\n[6/7] Embedding {len(enriched_chunks)} chunks ({EMBED_MODEL})...")
        vectors = embed_batch([c['clean_text'] for c in enriched_chunks])
        print(f"  → {len(vectors)} vectors, dim={len(vectors[0])}")
        for chunk in enriched_chunks:
            await meta_db.mark_embedded(chunk['chunk_id'], EMBED_MODEL)

        # ── 11. Write → Qdrant + ES + Neo4j ──────────────────────────────────
        print(f"\n[7/7] Write → Qdrant + Elasticsearch + Neo4j...")
        vector_db.upsert_batch(enriched_chunks, vectors)
        kw_db.index_batch(enriched_chunks)
        print(f"  Neo4j: {len(enriched_chunks)} chunks + entities...")
        for chunk in enriched_chunks:
            graph_db.write_chunk_full(chunk)
        print(f"  Neo4j: ✓")

        # ── 12. Finalize ──────────────────────────────────────────────────────
        await meta_db.finalize_document(doc_id, len(enriched_chunks))

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  ✅ Ingest hoàn thành!")
        print(f"  doc_id     : {doc_id}")
        print(f"  Sections   : {len(l1_chunks)}")
        print(f"  Paragraphs : {len(enriched_chunks)}")
        print(f"  Thời gian  : {elapsed:.1f}s")
        print(f"{'='*60}\n")
        return doc_id

    finally:
        await meta_db.close()
        graph_db.close()


async def ingest_directory(dir_path: str, force: bool = False) -> list:
    txt_files = list(Path(dir_path).glob('*.txt'))
    if not txt_files:
        print(f"Không tìm thấy file .txt trong {dir_path}")
        return []
    doc_ids = []
    for fp in txt_files:
        try:
            doc_ids.append(await ingest_file(str(fp), force=force))
        except Exception as e:
            print(f"  ❌ Lỗi {fp.name}: {e}")
    return doc_ids


if __name__ == "__main__":
    import sys
    force = '--force' in sys.argv
    args  = [a for a in sys.argv[1:] if a != '--force']
    if not args:
        print("Usage: python -m core.ingestor <file.txt> [--force]")
        sys.exit(1)
    asyncio.run(ingest_file(args[0], force=force))
