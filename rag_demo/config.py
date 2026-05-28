# config.py â€” Cáº¥u hÃ¬nh toÃ n bá»™ há»‡ thá»‘ng RAG
import os
from dotenv import load_dotenv

load_dotenv()

# â”€â”€ Local LLM (OpenAI-compatible) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLM_API_KEY   = os.getenv("LLM_API_KEY", "dummy")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "http://192.168.1.36:8881/v1")
LLM_MODEL     = os.getenv("LLM_MODEL", "gemma-3-12b-it-Q6_K.gguf")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# â”€â”€ Chunking mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "llm")

# â”€â”€ Embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMBED_MODEL = "intfloat/multilingual-e5-large"   # 1024 dims

# â”€â”€ Chunking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHUNK_SIZE_PARAGRAPH = 512

# â”€â”€ Coreference Resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COREF_ENABLED = os.getenv("COREF_ENABLED", "true").lower() == "true"
COREF_MODE    = os.getenv("COREF_MODE", "llm")

# â”€â”€ Hypothetical Questions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NUM_HYPO_QUESTIONS = 5

# â”€â”€ PostgreSQL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PG_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5432/rag_db"
)

# â”€â”€ Qdrant (Vector DB + BM25 Sparse) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "rag_chunks"
VECTOR_DIM        = 1024   # dense dim (multilingual-e5-large)

# â”€â”€ Neo4j (Graph DB) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NEO4J_URL      = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# â”€â”€ Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
USE_NEO4J     = os.getenv("USE_NEO4J", "true").lower() == "true"
SEARCH_TOP_K  = 6     # sá»‘ káº¿t quáº£ má»—i DB (Qdrant/Neo4j)
RRF_TOP_K     = 5     # sá»‘ á»©ng viÃªn giá»¯ láº¡i sau merge
FINAL_TOP_K   = 5     # sá»‘ chunk/parent Ä‘Æ°a vÃ o LLM context
RRF_K         = 60    # háº±ng sá»‘ RRF

SOURCE_MIN_RRF = float(os.getenv("SOURCE_MIN_RRF", "0.025"))

