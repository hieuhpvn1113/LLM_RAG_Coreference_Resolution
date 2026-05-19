# RAG Demo (Vector + Keyword + Graph)

Demo he thong RAG truy xuat du lieu noi bo voi 4 lop luu tru:
- PostgreSQL: metadata tai lieu/chunk + search logs
- Qdrant: vector embedding search
- Elasticsearch: BM25 keyword search
- Neo4j: entity + relation graph search

## 1) Cau truc du an

```text
rag_demo/
  api/                 # FastAPI endpoints: /health, /ingest, /query
  core/                # chunker, enricher, embedder, ingestor, retriever
  db/                  # clients cho postgres/qdrant/es/neo4j
  llm/                 # LLM client + prompts
  data/                # du lieu mau
  migrations/          # SQL schema va index
  main.py              # CLI ingest/query
  test_connections.py  # smoke test ket noi 4 DB
  inspect_db.py        # script kiem tra nhanh data da ingest
  docker-compose.yml
  .env.example
```

## 2) Yeu cau

- Python 3.11+ (khuyen nghi 3.12)
- Docker Desktop
- API key cho LLM (Groq theo cau hinh trong `.env`)

## 3) Cai dat nhanh

```bash
cd rag_demo
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tao file env:

```bash
copy .env.example .env
```

Mo `.env` va dien cac gia tri can thiet (DB, LLM key, model, ...).

## 4) Khoi dong he thong

```bash
docker compose up -d
```

Kiem tra ket noi:

```bash
python test_connections.py
```

Neu tat ca OK, ban co the ingest va query.

## 5) Nap du lieu (ingest)

### Cach 1: CLI

```bash
python main.py ingest data/input.txt
```

### Cach 2: API

Chay server:

```bash
uvicorn api.main:app --reload
```

Goi ingest:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"file_path":"data/input.txt"}'
```

## 6) Hoi dap (query)

### CLI

```bash
python main.py query "RAG la gi?"
```

### API

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"RAG la gi?", "verbose": true}'
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

## 7) Kiem tra da nap du lieu chua

- Chay script:

```bash
python inspect_db.py
```

- Hoac kiem tra nhanh:
  - Neo4j Browser: http://localhost:7474
  - Elasticsearch: http://localhost:9200
  - Qdrant: http://localhost:6333/dashboard
  - PostgreSQL: `psql -h localhost -p 5433 -U rag_user -d rag_db`

## 8) Luong test toi thieu de xac nhan hoat dong

1. `docker compose up -d`
2. `python test_connections.py`
3. `python main.py ingest data/input.txt`
4. `python main.py query "RAG la gi?"`
5. `python inspect_db.py`

Neu 5 buoc tren pass, he thong da chay end-to-end.

## 9) Luu y quan trong

- Khong chay `python api/main.py` truc tiep. Dung:
  - `python -m api.main` hoac
  - `uvicorn api.main:app --reload`
- Port PostgreSQL map ra may host la `5433` (khong phai 5432).
- Lan ingest dau voi tai lieu lon co the cham do embedding + LLM enrichment.

## 10) Lenh huu ich

```bash
# Xem logs cac service
docker compose logs -f

# Restart 1 service
docker compose restart elasticsearch

# Dung he thong
docker compose down
```
