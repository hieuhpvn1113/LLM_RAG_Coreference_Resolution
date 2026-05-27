# config.py — Cấu hình toàn bộ hệ thống RAG
import os
from dotenv import load_dotenv

load_dotenv()

# ── Local LLM (OpenAI-compatible) ───────────────────────────
LLM_API_KEY   = os.getenv("LLM_API_KEY", "dummy")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "http://192.168.1.36:8881/v1")
LLM_MODEL     = os.getenv("LLM_MODEL", "gemma-3-12b-it-Q6_K.gguf")

# ── Chunking mode ───────────────────────────────────────────
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "llm")

# ── Embedding ───────────────────────────────────────────────
EMBED_MODEL = "intfloat/multilingual-e5-large"   # 1024 dims

# ── Chunking ────────────────────────────────────────────────
CHUNK_SIZE_PARAGRAPH = 512

# ── Coreference Resolution ──────────────────────────────────
COREF_ENABLED = os.getenv("COREF_ENABLED", "true").lower() == "true"
COREF_MODE    = os.getenv("COREF_MODE", "rule")

# ── Hypothetical Questions ──────────────────────────────────
NUM_HYPO_QUESTIONS = 5

# ── PostgreSQL ──────────────────────────────────────────────
PG_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5432/rag_db"
)

# ── Qdrant (Vector DB + BM25 Sparse) ────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "rag_chunks"
VECTOR_DIM        = 1024   # dense dim (multilingual-e5-large)

# ── Neo4j (Graph DB) ────────────────────────────────────────
NEO4J_URL      = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Search ──────────────────────────────────────────────────
SEARCH_TOP_K  = 6     # số kết quả mỗi DB (Qdrant/Neo4j)
RRF_TOP_K     = 5     # số ứng viên giữ lại sau merge
FINAL_TOP_K   = 5     # số chunk/parent đưa vào LLM context
RRF_K         = 60    # hằng số RRF

SOURCE_MIN_RRF = float(os.getenv("SOURCE_MIN_RRF", "0.025"))
