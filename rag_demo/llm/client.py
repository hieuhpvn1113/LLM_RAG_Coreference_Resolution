# llm/client.py — Local LLM client (OpenAI-compatible)
"""
Tất cả LLM call đều đi qua local LLM server (OpenAI-compatible API).
Model: Gemma 3 12B Q6  (gemma-3-12b-it-Q6_K.gguf)
Endpoint: http://192.168.1.36:8881/v1

Alias LLMClient / AsyncLLMClient giữ nguyên để không phá code cũ.
"""
import httpx
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


def _build_headers() -> dict:
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }


def _build_payload(system: str, user: str, max_tokens: int) -> dict:
    return {
        "model":       LLM_MODEL,
        "max_tokens":  max_tokens,
        "temperature": LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------
class LocalLLMClient:
    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=_build_headers(),
            json=_build_payload(system, user, max_tokens),
            timeout=120,  # local model có thể chậm hơn, tăng timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------
class AsyncLocalLLMClient:
    async def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers=_build_headers(),
                json=_build_payload(system, user, max_tokens),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Alias — giữ nguyên tên cũ để không cần sửa import ở enricher / retriever
# ---------------------------------------------------------------------------
LLMClient      = LocalLLMClient
AsyncLLMClient = AsyncLocalLLMClient
