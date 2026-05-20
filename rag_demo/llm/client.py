# llm/client.py — Groq API client (OpenAI-compatible)
"""
Tất cả LLM call đều đi qua Groq API.
Endpoint: https://api.groq.com/openai/v1
Model mặc định: llama-3.3-70b-versatile (free tier, rất nhanh)

Alias LLMClient / AsyncLLMClient giữ nguyên để không phá code cũ.
"""
import httpx
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


def _build_headers() -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY chưa được set!\n"
            "→ Lấy key miễn phí tại https://console.groq.com\n"
            "→ Thêm vào file .env:  GROQ_API_KEY=gsk_xxxx"
        )
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }


def _build_payload(system: str, user: str, max_tokens: int) -> dict:
    return {
        "model":       GROQ_MODEL,
        "max_tokens":  max_tokens,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------
class GroqClient:
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = httpx.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=_build_headers(),
            json=_build_payload(system, user, max_tokens),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------
class AsyncGroqClient:
    async def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=_build_headers(),
                json=_build_payload(system, user, max_tokens),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Alias — giữ nguyên tên cũ để không cần sửa import ở enricher / retriever
# ---------------------------------------------------------------------------
LLMClient      = GroqClient
AsyncLLMClient = AsyncGroqClient
