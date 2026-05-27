# RAG Demo - Kien truc hien tai (L1/L2 + Coref + Self-Query)

Tai lieu nay mo ta DUNG he thong hien tai trong code `rag_demo`.

## 1) Tong quan nhanh

He thong la RAG 2 cap chunk:
- L1 = chunk theo cau truc tai lieu (section-level)
- L2 = chunk theo ngu nghia ben trong moi L1 (paragraph-level)

Diem khac biet chinh:
- Co tien xu ly coreference truoc khi chia L2 de giam tu mo ho (`no`, `ho`, `dieu nay`, ...)
- Retrieval uu tien Self-Query (LLM -> JSON filter) + Qdrant hybrid + Neo4j + RRF
- LLM tra loi tu `raw_text` cua L1 (khong phai text cat nho L2)

## 2) Phuong phap chunk dang dung (thu tu thuc thi)

### Buoc 1 - Lam sach text
- Dung `clean_text()` trong `core/chunker.py`
- Chuan hoa newline/space, giu noi dung goc

### Buoc 2 - Chunk L1 (Hierarchical Split)
- Dung `hierarchical_split()`
- Tach theo heading cau truc tai lieu (ALL-CAPS, Roman/CHUONG/PHAN/MUC, metric-heading...)
- Bo page-header lap lai
- Merge L1 qua nho (<80 token, tru heading manh)
- Output: L1 co `raw_text`, `clean_text`, `chunk_id`, `prev_id/next_id`

### Buoc 3 - Coref tren moi L1
- Dung `resolve_coref()` trong `core/coref_resolver.py`
- Mac dinh mode rule-based (`COREF_MODE=rule`), co ho tro neural
- Muc tieu: thay cum mo ho bang antecedent ro hon
- Luu y quan trong:
  - `raw_text` L1 giu nguyen
  - Ban coref dung cho chia L2 + embedding/index

### Buoc 4 - Chunk L2 (Semantic Split)
- Dung `semantic_split()`
- Tach cau, embed theo sentence, tim diem cat bang cosine distance + adaptive threshold
- Cat theo sentence-boundary, khong overlap
- Output L2:
  - `raw_text` = van goc
  - `clean_text` = da coref
  - `parent_id` tro ve L1

Ket luan: he thong dang la **hybrid chunking** = Hierarchical (L1) + Semantic (L2), co chen **Coref pre-processing** truoc semantic split.

## 3) Ingestion pipeline hien tai

Code chinh: `core/ingestor.py`

1. Parse file -> text (`core/file_parser.py`)
2. `clean_text`
3. Tao document record trong PostgreSQL
4. Tao L1 (`hierarchical_split`) va luu PostgreSQL + node Neo4j
5. Tao L2 (`semantic_split`, co coref)
6. LLM enrichment cho L2 (`title`, `summary`, `keywords`, `entities`, `relations`, `hypothetical_questions`)
7. Embed L2 (`core/embedder.py`)
8. Upsert Qdrant:
   - Dense vector
   - Sparse BM25 vector (fastembed)
   - Payload filter fields (`keywords`, `entities`, `entity_types`, `source_file`, ...)
9. Ghi graph vao Neo4j
10. Finalize document

## 4) Retrieval pipeline hien tai

Code chinh: `core/retriever.py` + `core/self_query.py`

1. Query rewrite (LLM) -> `original/technical/keywords`
2. Self-query (LLM) -> JSON:
   - `search_query`
   - `filter_mode` (`strict|relaxed|none`)
   - `qdrant_filter` (build tu JSON)
3. Parallel search:
   - Qdrant hybrid (dense + sparse BM25, co filter neu co)
   - Neo4j entity search
4. Hop nhat bang RRF
5. Map L2 -> L1 bang `parent_id`, dedupe parent
6. Rank lai cac L1 theo query, lay top `FINAL_TOP_K`
7. Gui full `raw_text` cua L1 vao LLM de tao cau tra loi
8. Ghi `search_logs`

## 5) So do luong chay

### 5.1 Ingest

```text
File -> Parse -> Clean
     -> L1 Hierarchical Split
     -> Coref (per L1)
     -> L2 Semantic Split
     -> LLM Enrichment (L2)
     -> Embedding (L2)
     -> Qdrant (dense+sparse+payload)
     -> Neo4j (entities/relations)
     -> PostgreSQL (documents/chunks/logical links)
```

### 5.2 Query

```text
User Query
  -> Query Rewrite (LLM)
  -> Self-Query JSON (LLM -> Qdrant filter)
  -> Search song song:
       Qdrant hybrid + Neo4j
  -> RRF merge
  -> L2 -> parent_id -> L1 dedupe
  -> Top L1 context
  -> LLM Answer (doc full context theo L1 raw_text)
```

## 6) Database vai tro

- PostgreSQL: document/chunk metadata, parent-child, enrichment, search logs
- Qdrant: retrieval chinh (dense + sparse BM25 + payload filters)
- Neo4j: entity graph retrieval bo tro
- Elasticsearch/Kibana: co trong compose de debug/thu nghiem, khong nam trong retrieval flow chinh hien tai

## 7) Lenh chay nhanh

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
docker compose up -d
python test_connections.py
python main.py ingest data\20260429_VNM_Ban_tin_NDT_Q1_2026.pdf
python main.py query "Cong ty nao se to chuc cuoc hop?"
```

## 8) Ghi chu ve bai toan ban neu

Neu cau tra loi dang ra "Cong ty se to chuc cuoc hop" ma khong ra ten cong ty, nguyen nhan thuong la:
- Coref chua resolve dung chu the trong L1 do
- Hoac L2 tim duoc nhung parent L1 top cuoi cung chua chua cau co ten cong ty

Diem can kiem tra nhanh:
- `COREF_ENABLED`, `COREF_MODE` trong `config.py`
- Payload `entities/keywords` cua L2 trong Qdrant
- Top L2 + top L1 in log verbose cua `core/retriever.py`
