# core/self_query.py — Self-Querying: LLM dịch câu hỏi → Qdrant Filter
"""
Luồng:
  1. LLM đọc câu hỏi tự nhiên
  2. Xuất JSON chứa filter conditions + search query
  3. Build Qdrant Filter object để lọc cứng trước khi search vector/BM25

Filter fields có trong Qdrant payload:
  - source_file   : str
  - keywords      : list[str]
  - entities      : list[str]
  - entity_types  : list[str]  (PERSON, ORG, CONCEPT, LOCATION)
  - level         : int
  - token_count   : int
"""

import json
import re

from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Range,
)

from llm.client import AsyncLLMClient


# ── Prompt ────────────────────────────────────────────────────────────────────

SELF_QUERY_SYSTEM = """
Bạn là bộ dịch câu hỏi sang filter JSON cho hệ thống RAG.
Phân tích câu hỏi và trả về JSON với cấu trúc sau — CHỈ JSON thuần túy, không markdown.

{
  "search_query": "câu hỏi rút gọn để tìm kiếm vector/BM25",
  "filters": {
    "source_file":   "tên file cụ thể nếu hỏi về file nào đó, null nếu không rõ",
    "keywords_any":  ["từ khóa 1", "từ khóa 2"],
    "entities_any":  ["tên thực thể 1", "tên thực thể 2"],
    "entity_types_any": ["PERSON", "ORG", "CONCEPT", "LOCATION"],
    "min_token_count": null,
    "max_token_count": null
  },
  "filter_mode": "strict | relaxed | none"
}

Quy tắc:
- filter_mode = "strict"  : dùng khi câu hỏi có thực thể/tên/địa điểm rõ ràng
- filter_mode = "relaxed" : dùng khi câu hỏi mang tính khái niệm, có một vài từ khóa
- filter_mode = "none"    : câu hỏi quá tổng quát, không có điều kiện lọc cụ thể
- keywords_any: chỉ lấy danh từ chính, tối đa 5 từ
- entities_any: tên người, tổ chức, địa điểm, sản phẩm được nhắc đến
- Nếu không có điều kiện cho trường nào → đặt null hoặc []
""".strip()

SELF_QUERY_USER = "Câu hỏi: {query}"


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw.strip())
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ── Build Qdrant Filter ───────────────────────────────────────────────────────

def build_qdrant_filter(parsed: dict) -> Filter | None:
    """
    Chuyển dict filters → Qdrant Filter object.
    Trả về None nếu không có điều kiện nào.
    """
    filters_raw = parsed.get("filters", {})
    mode        = parsed.get("filter_mode", "none")

    if mode == "none" or not filters_raw:
        return None

    must_conditions = []

    # source_file — exact match
    sf = filters_raw.get("source_file")
    if sf and isinstance(sf, str):
        must_conditions.append(
            FieldCondition(key="source_file", match=MatchValue(value=sf))
        )

    # keywords_any — chunk phải chứa ÍT NHẤT 1 trong các keyword
    kw = filters_raw.get("keywords_any")
    if kw and isinstance(kw, list) and len(kw) > 0:
        clean_kw = [str(k).strip().lower() for k in kw if k]
        if clean_kw:
            if mode == "strict":
                must_conditions.append(
                    FieldCondition(key="keywords", match=MatchAny(any=clean_kw))
                )
            else:
                # relaxed: thêm vào should thay vì must (không bắt buộc)
                pass  # để search_query xử lý qua BM25

    # entities_any — chunk phải chứa ÍT NHẤT 1 entity
    ents = filters_raw.get("entities_any")
    if ents and isinstance(ents, list) and len(ents) > 0:
        clean_ents = [str(e).strip() for e in ents if e]
        if clean_ents and mode == "strict":
            must_conditions.append(
                FieldCondition(key="entities", match=MatchAny(any=clean_ents))
            )

    # entity_types_any
    etypes = filters_raw.get("entity_types_any")
    if etypes and isinstance(etypes, list) and len(etypes) > 0:
        valid_types = [t for t in etypes if t in ("PERSON", "ORG", "CONCEPT", "LOCATION")]
        if valid_types and mode == "strict":
            must_conditions.append(
                FieldCondition(key="entity_types", match=MatchAny(any=valid_types))
            )

    # token_count range
    min_tok = filters_raw.get("min_token_count")
    max_tok = filters_raw.get("max_token_count")
    if min_tok is not None or max_tok is not None:
        must_conditions.append(
            FieldCondition(
                key="token_count",
                range=Range(
                    gte=int(min_tok) if min_tok is not None else None,
                    lte=int(max_tok) if max_tok is not None else None,
                )
            )
        )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


# ── Main entry ────────────────────────────────────────────────────────────────

async def self_query(query: str, llm: AsyncLLMClient) -> dict:
    """
    Gọi LLM để phân tích câu hỏi → trả về:
    {
        "search_query" : str,          # query rút gọn để vector/BM25
        "qdrant_filter": Filter | None, # filter object cho Qdrant
        "filter_mode"  : str,          # strict / relaxed / none
        "raw_parsed"   : dict,         # JSON gốc LLM trả về (debug)
    }
    """
    try:
        raw = await llm.complete(
            system=SELF_QUERY_SYSTEM,
            user=SELF_QUERY_USER.format(query=query),
            max_tokens=400,
        )
        parsed = _parse_json(raw)
    except Exception as e:
        print(f"  ⚠️  Self-query LLM error: {e}")
        parsed = {}

    if not parsed:
        return {
            "search_query":  query,
            "qdrant_filter": None,
            "filter_mode":   "none",
            "raw_parsed":    {},
        }

    search_query   = parsed.get("search_query") or query
    filter_mode    = parsed.get("filter_mode", "none")
    qdrant_filter  = build_qdrant_filter(parsed)

    return {
        "search_query":  search_query,
        "qdrant_filter": qdrant_filter,
        "filter_mode":   filter_mode,
        "raw_parsed":    parsed,
    }
