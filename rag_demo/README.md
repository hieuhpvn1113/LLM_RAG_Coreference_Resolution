# RAG Demo (Vector + Keyword + Graph)

Tai lieu nay mo ta quy trinh khoi dong, test, va kiem tra du lieu theo trang thai code hien tai.

## 1) Tong quan he thong

RAG su dung 4 tang luu tru:
- PostgreSQL: metadata document/chunk + search logs
- Qdrant: vector search
- Elasticsearch: keyword/BM25 search
- Neo4j: entity/relation graph

## 2) Yeu cau

- Python 3.11+ (khuyen nghi 3.12)
- Docker Desktop
- Da cai dependency Python tu `requirements.txt`

## 3) Cai dat nhanh

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4) Cau hinh can luu y

Nguon su that cau hinh: `config.py`

Luu y quan trong ve PostgreSQL:
- Trong Docker Compose: PostgreSQL map `5433:5432`
- Neu chay script tu host, nen set `DATABASE_URL` dung port host `5433`, vi du:

```powershell
$env:DATABASE_URL="postgresql://rag_user:rag_password@localhost:5433/rag_db"
```

## 5) Quy trinh khoi dong he thong

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
docker compose down
docker compose up -d
```

Cac endpoint/GUI de kiem tra service:
- PostgreSQL: `localhost:5433`
- pgAdmin: `http://localhost:5050`
- Qdrant: `http://localhost:6333/dashboard`
- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`
- Neo4j Browser: `http://localhost:7474`

## 6) Chay test he thong (smoke test)

Sau khi services len, chay:

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python test_connections.py
```

Script se test:
- Ket noi PostgreSQL, Qdrant, Elasticsearch, Neo4j
- Kiem tra bang PostgreSQL bat buoc: `documents`, `chunks`, `search_logs`

Neu pass 4/4 service thi co the ingest/query.

## 7) Chay ingest va query

CLI ingest:

```powershell
python main.py ingest data\20260429_VNM_Ban_tin_NDT_Q1_2026.pdf
```

CLI query:

```powershell
python main.py query "Noi dung chinh cua tai lieu la gi?"
```

API server:

```powershell
uvicorn api.main:app --reload
```

API endpoints:
- `GET /health`
- `POST /ingest` body: `{ "file_path": "..." }`
- `POST /query` body: `{ "query": "...", "verbose": false }`

## 8) Kiem tra du lieu nam o dau

### A. Kiem tra bang script tong hop

```powershell
python inspect_db.py
```

Loc theo tung DB:

```powershell
python inspect_db.py --pg
python inspect_db.py --qdrant
python inspect_db.py --es
python inspect_db.py --neo4j
```

Loc theo document:

```powershell
python inspect_db.py --doc <doc_id>
```

### B. Kiem tra tren GUI/endpoint

- PostgreSQL:
  - pgAdmin: `http://localhost:5050`
- Qdrant:
  - Dashboard: `http://localhost:6333/dashboard`
- Elasticsearch:
  - API: `http://localhost:9200`
  - Kibana: `http://localhost:5601`
- Neo4j:
  - Browser: `http://localhost:7474`

## 9) Reset du lieu (neu can)

```powershell
python reset_all_db.py
python reset_all_db.py --yes
python reset_all_db.py --pg-only
python reset_all_db.py --qdrant-only
python reset_all_db.py --es-only
python reset_all_db.py --neo4j-only
```

## 10) Luong E2E khuyen nghi

1. `docker compose up -d`
2. `python test_connections.py`
3. `python main.py ingest <file>`
4. `python main.py query "..."`
5. `python inspect_db.py`
