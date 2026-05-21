# core/enricher.py — LLM Enrichment cho mỗi Level 2 chunk
"""
Gọi Local LLM để sinh metadata phong phú cho mỗi chunk:
  - title, summary, keywords, entities, relations, hypothetical_questions

Không có rate limit với local LLM — không cần delay giữa các request.
Chỉ retry khi gặp lỗi thật sự (network, timeout, JSON lỗi).
"""

import json
import re
import asyncio

import httpx

from llm.client import AsyncLLMClient
from llm.prompts import ENRICHMENT_SYSTEM, ENRICHMENT_USER


def _parse_llm_json(raw: str) -> dict:
    if not raw:
        return {}
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw.strip())
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _default_enrichment() -> dict:
    return {
        "title": "",
        "summary": "",
        "keywords": [],
        "entities": [],
        "relations": [],
        "hypothetical_questions": [],
    }


def _validate_enrichment(data: dict) -> dict:
    result = _default_enrichment()
    if isinstance(data.get("title"), str):
        result["title"] = data["title"][:200]
    if isinstance(data.get("summary"), str):
        result["summary"] = data["summary"][:1000]
    if isinstance(data.get("keywords"), list):
        result["keywords"] = [str(k) for k in data["keywords"][:10]]
    if isinstance(data.get("entities"), list):
        entities = []
        for e in data["entities"][:20]:
            if isinstance(e, dict) and "name" in e:
                entities.append({
                    "name": str(e["name"])[:100],
                    "type": str(e.get("type", "CONCEPT"))[:20],
                })
        result["entities"] = entities
    if isinstance(data.get("relations"), list):
        relations = []
        for r in data["relations"][:20]:
            if isinstance(r, dict) and "from" in r and "to" in r:
                relations.append({
                    "from":     str(r["from"])[:100],
                    "relation": str(r.get("relation", "RELATES_TO"))[:50],
                    "to":       str(r["to"])[:100],
                })
        result["relations"] = relations
    if isinstance(data.get("hypothetical_questions"), list):
        result["hypothetical_questions"] = [
            str(q)[:300] for q in data["hypothetical_questions"][:5]
        ]
    return result


def _retry_after(exc: Exception) -> float | None:
    """
    Đọc header Retry-After từ HTTPStatusError 429.
    Trả về số giây cần đợi, hoặc None nếu không phải lỗi 429.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        header = exc.response.headers.get("retry-after") or \
                 exc.response.headers.get("x-ratelimit-reset-requests")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        return 30.0   # fallback nếu server không trả header
    return None


async def enrich_chunk(chunk_text: str, llm: AsyncLLMClient,
                       max_retries: int = 5) -> dict:
    """
    Enrich 1 chunk text qua Local LLM.
    Retry khi gặp lỗi, không có delay cố định giữa các request thành công.
    """
    MAX_CHARS = 3000
    if len(chunk_text) > MAX_CHARS:
        chunk_text = chunk_text[:MAX_CHARS] + '...'

    user_prompt = ENRICHMENT_USER.format(chunk_text=chunk_text)

    for attempt in range(1, max_retries + 1):
        try:
            raw = await llm.complete(
                system=ENRICHMENT_SYSTEM,
                user=user_prompt,
                max_tokens=1200,
            )
            data = _parse_llm_json(raw)
            if data:
                return _validate_enrichment(data)

            print(f"\n    ⚠️  JSON rỗng attempt {attempt}/{max_retries}, raw: {raw[:80]!r}")
            if attempt < max_retries:
                await asyncio.sleep(1.0)

        except Exception as e:
            wait = _retry_after(e)
            if wait is not None:
                print(f"\n    ⏳ Rate limit (429) — đợi {wait:.0f}s rồi retry [{attempt}/{max_retries}]...")
                await asyncio.sleep(wait + 1)
            else:
                print(f"\n    ⚠️  LLM error attempt {attempt}/{max_retries}: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)

    print("    ⚠️  Enrichment thất bại sau tất cả retries — dùng fallback defaults")
    return _default_enrichment()
