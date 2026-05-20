# config.py — Cấu hình toàn bộ hệ thống RAG
import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq API ────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL  = "https://api.groq.com/openai/v1"
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Chunking mode ───────────────────────────────────────────
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "llm")

# ── Embedding ───────────────────────────────────────────────
EMBED_MODEL = "intfloat/multilingual-e5-base"   # 768 dims

# ── Chunking ────────────────────────────────────────────────
# CHUNK_SIZE_PARAGRAPH: soft limit — chunk sẽ flush khi token ≥ giá trị này
# VÀ vừa kết thúc câu. Chunk luôn kết thúc bằng câu hoàn chỉnh.
CHUNK_SIZE_PARAGRAPH = 512    # token — Level 2 (soft limit, sentence-boundary)
SEMANTIC_THRESHOLD   = 0.24   # ngưỡng cosine distance để cắt semantic unit

# ── Hypothetical Questions ──────────────────────────────────
NUM_HYPO_QUESTIONS = 5

# ── PostgreSQL ──────────────────────────────────────────────
PG_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5432/rag_db"
)

# ── Qdrant ──────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "rag_chunks"
VECTOR_DIM        = 768

# ── Elasticsearch ───────────────────────────────────────────
ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = "rag_chunks"

# ── Neo4j ───────────────────────────────────────────────────
NEO4J_URL      = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Search ──────────────────────────────────────────────────
SEARCH_TOP_K = 3
FINAL_TOP_K  = 6
RRF_K        = 60

# ── Hybrid Search weights (Qdrant) ──────────────────────────
DENSE_WEIGHT  = 0.7
SPARSE_WEIGHT = 0.3
