# LLM_RAG - System Init Snapshot

Tai lieu nay mo ta trang thai THUC TE cua he thong tai thoi diem hien tai, dung nhu file init de setup/chay nhanh.

## 1) Tong quan

Workspace goc:
- `E:\AI_agent\LLM_RAG`

Thanh phan chinh:
- `server.py`: MCP server tool wrapper (run command, read/write/list file, tao/xoa/move path trong `ALLOWED_DIRECTORY`)
- `rag_demo/`: pipeline RAG chinh (ingest + retrieve + API + scripts van hanh)
- `venv/`: virtual environment

RAG stack (4 lop luu tru):
- PostgreSQL: metadata documents/chunks + search logs
- Qdrant: vector search
- Elasticsearch: keyword/BM25 search
- Neo4j: graph entity/relation search

## 2) Cau truc thu muc hien tai

```text
LLM_RAG/
  server.py
  LLM_RAG.md
  rag_demo/
    api/main.py
    core/
    db/
    llm/
    migrations/
      001_create_documents.sql
      002_create_chunks.sql
      003_create_indexes.sql
    data/
    docker-compose.yml
    requirements.txt
    config.py
    main.py
    test_connections.py
    inspect_db.py
    reset_all_db.py
```

## 3) Yeu cau moi truong

- Windows + PowerShell (theo workspace hien tai)
- Python 3.11+ (khuyen nghi 3.12)
- Docker Desktop
- Tao va kich hoat `venv`

Cai dependency:
```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
pip install -r requirements.txt
```

## 4) Cau hinh he thong (nguon su that: `rag_demo/config.py`)

LLM (OpenAI-compatible endpoint local/remote):
- `LLM_API_KEY` (default: `dummy`)
- `LLM_BASE_URL` (default: `http://192.168.1.36:8881/v1`)
- `LLM_MODEL` (default: `gemma-3-12b-it-Q6_K.gguf`)

Chunking / NLP:
- `CHUNKING_MODE` (default: `llm`)
- `CHUNK_SIZE_PARAGRAPH = 512`
- `COREF_ENABLED` (default true)
- `COREF_MODE` (default: `rule`)
- `NUM_HYPO_QUESTIONS = 5`

Embedding:
- `EMBED_MODEL = intfloat/multilingual-e5-large`
- `VECTOR_DIM = 1024`

Databases:
- `DATABASE_URL` -> `PG_DSN` (default trong code: `postgresql://rag_user:rag_password@localhost:5432/rag_db`)
- `QDRANT_URL` (default: `http://localhost:6333`)
- `ES_URL` (default: `http://localhost:9200`)
- `NEO4J_URL` (default: `bolt://localhost:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `password`)

Search params:
- `SEARCH_TOP_K = 3`
- `FINAL_TOP_K = 6`
- `RRF_K = 60`
- `SOURCE_MIN_RRF` (default: `0.025`)

Luu y quan trong:
- Docker map PostgreSQL ra host `5433:5432`.
- Neu dung docker-compose mac dinh, can set `DATABASE_URL` theo port host 5433 khi chay script tu host:
  - `postgresql://rag_user:rag_password@localhost:5433/rag_db`

## 5) Khoi dong ha tang DB

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
docker compose down
docker compose up -d
```

Service va ports:
- PostgreSQL: `localhost:5433`
- pgAdmin: `http://localhost:5050`
- Qdrant: `http://localhost:6333` (dashboard: `/dashboard`)
- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`
- Neo4j Browser: `http://localhost:7474` (bolt `7687`)

## 6) Smoke test ket noi

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python test_connections.py
```

Script nay check:
- ket noi 4 DB
- bang PostgreSQL: `documents`, `chunks`, `search_logs`

## 7) Chay ingest/query

CLI:
```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python main.py ingest data\<ten_file>
python main.py query "RAG la gi?"
```

API:
```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
uvicorn api.main:app --reload
```

Endpoints:
- `GET /health`
- `POST /ingest` body: `{ "file_path": "..." }`
- `POST /query` body: `{ "query": "...", "verbose": false }`

## 8) Scripts van hanh

Kiem tra du lieu trong 4 DB:
```powershell
python inspect_db.py
python inspect_db.py --pg
python inspect_db.py --doc <doc_id>
```

Reset data:
```powershell
python reset_all_db.py
python reset_all_db.py --yes
python reset_all_db.py --pg-only
python reset_all_db.py --qdrant-only
python reset_all_db.py --es-only
python reset_all_db.py --neo4j-only
```

## 9) MCP server o root (`server.py`)

Chay MCP server:
```powershell
cd E:\AI_agent\LLM_RAG
python server.py
```

Tool exposed:
- `run_terminal_command`
- `list_files`
- `read_file`
- `write_file`
- `make_directory`
- `delete_path`
- `move_path`

Phan vung an toan file-system dua tren env:
- `ALLOWED_DIRECTORY` (neu khong set thi mac dinh `os.getcwd()`)

## 10) Trinh tu init de chay end-to-end

1. Kich hoat venv + `pip install -r requirements.txt`
2. `docker compose up -d`
3. Dam bao `DATABASE_URL` dung port host (`5433`) neu chay tu host
4. `python test_connections.py`
5. `python main.py ingest <file>`
6. `python main.py query "..."`
7. (tu chon) `uvicorn api.main:app --reload` de test API

## 11) Ghi chu cap nhat

- File nay da duoc dong bo theo code va cau hinh hien co trong repo.
- Neu doi model/DB endpoint/env, cap nhat `rag_demo/config.py` (hoac bien moi truong) va cap nhat lai file nay.
