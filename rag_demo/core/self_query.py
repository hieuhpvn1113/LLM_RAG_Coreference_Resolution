# core/self_query.py â€” Self-Querying: LLM dá»‹ch cÃ¢u há»i â†’ Qdrant Filter
"""
Luá»“ng:
  1. LLM Ä‘á»c cÃ¢u há»i tá»± nhiÃªn
  2. Xuáº¥t JSON chá»©a filter conditions + search query
  3. Build Qdrant Filter object Ä‘á»ƒ lá»c cá»©ng trÆ°á»›c khi search vector/BM25

Filter fields cÃ³ trong Qdrant payload:
  - source_file   : str
  - keywords      : list[str]
  - entities      : list[str]
  - entity_types  : list[str]  (PERSON, ORG, CONCEPT, LOCATION)
  - scopes        : list[str]  (vd: domain_scope_2026)
  - level         : int
  - token_count   : int
"""

import json
import re
import unicodedata

from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Range,
)

from llm.client import AsyncLLMClient


# â”€â”€ Prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SELF_QUERY_SYSTEM = """
Báº¡n lÃ  bá»™ dá»‹ch cÃ¢u há»i sang filter JSON cho há»‡ thá»‘ng RAG.
PhÃ¢n tÃ­ch cÃ¢u há»i vÃ  tráº£ vá» JSON vá»›i cáº¥u trÃºc sau â€” CHá»ˆ JSON thuáº§n tÃºy, khÃ´ng markdown.

{
  "search_query": "cÃ¢u há»i rÃºt gá»n Ä‘á»ƒ tÃ¬m kiáº¿m vector/BM25",
  "filters": {
    "source_file":   "tÃªn file cá»¥ thá»ƒ náº¿u há»i vá» file nÃ o Ä‘Ã³, null náº¿u khÃ´ng rÃµ",
    "keywords_any":  ["tá»« khÃ³a 1", "tá»« khÃ³a 2"],
    "entities_any":  ["tÃªn thá»±c thá»ƒ 1", "tÃªn thá»±c thá»ƒ 2"],
    "entity_types_any": ["PERSON", "ORG", "CONCEPT", "LOCATION"],
    "scopes_any": ["domain_scope_2026"],
    "min_token_count": null,
    "max_token_count": null
  },
  "filter_mode": "strict | relaxed | none"
}

Quy táº¯c:
- filter_mode = "strict"  : dÃ¹ng khi cÃ¢u há»i cÃ³ thá»±c thá»ƒ/tÃªn/Ä‘á»‹a Ä‘iá»ƒm rÃµ rÃ ng
- filter_mode = "relaxed" : dÃ¹ng khi cÃ¢u há»i mang tÃ­nh khÃ¡i niá»‡m, cÃ³ má»™t vÃ i tá»« khÃ³a
- filter_mode = "none"    : cÃ¢u há»i quÃ¡ tá»•ng quÃ¡t, khÃ´ng cÃ³ Ä‘iá»u kiá»‡n lá»c cá»¥ thá»ƒ
- keywords_any: chá»‰ láº¥y danh tá»« chÃ­nh, tá»‘i Ä‘a 5 tá»«
- entities_any: tÃªn ngÆ°á»i, tá»• chá»©c, Ä‘á»‹a Ä‘iá»ƒm, sáº£n pháº©m Ä‘Æ°á»£c nháº¯c Ä‘áº¿n
- scopes_any: dÃ¹ng khi cÃ¢u há»i Ã¡m chá»‰ má»™t ngá»¯ cáº£nh cá»¥ thá»ƒ (vd ÄHÄCÄ mot doanh nghiep cu the)
- Náº¿u khÃ´ng cÃ³ Ä‘iá»u kiá»‡n cho trÆ°á»ng nÃ o â†’ Ä‘áº·t null hoáº·c []
""".strip()

SELF_QUERY_USER = "CÃ¢u há»i: {query}"


# â”€â”€ Parser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Build Qdrant Filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_qdrant_filter(parsed: dict) -> Filter | None:
    """
    Chuyá»ƒn dict filters â†’ Qdrant Filter object.
    Tráº£ vá» None náº¿u khÃ´ng cÃ³ Ä‘iá»u kiá»‡n nÃ o.
    """
    filters_raw = parsed.get("filters", {})
    mode        = parsed.get("filter_mode", "none")

    if mode == "none" or not filters_raw:
        return None

    must_conditions = []

    # source_file â€” exact match
    sf = filters_raw.get("source_file")
    if sf and isinstance(sf, str):
        must_conditions.append(
            FieldCondition(key="source_file", match=MatchValue(value=sf))
        )

    # keywords_any â€” xu ly mem qua search_query (dense/sparse), khong filter cung
    kw = filters_raw.get("keywords_any")
    if kw and isinstance(kw, list) and len(kw) > 0:
        clean_kw = [str(k).strip().lower() for k in kw if k]
        if clean_kw:
            pass

    # entities_any â€” chunk pháº£i chá»©a ÃT NHáº¤T 1 entity
    ents = filters_raw.get("entities_any")
    if ents and isinstance(ents, list) and len(ents) > 0:
        clean_ents = []
        for e in ents:
            if not e:
                continue
            token = str(e).strip()
            # Bo qua token dang ngay/so de tranh over-filter (vd 31/03/2025).
            if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", token):
                continue
            if re.fullmatch(r"[\d\.,%]+", token):
                continue
            clean_ents.append(token)
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

    # scopes_any
    # Khong filter cung theo scope vi de gay mat recall khi metadata scope khong dong nhat.
    scopes = filters_raw.get("scopes_any")
    if scopes and isinstance(scopes, list) and len(scopes) > 0:
        clean_scopes = [str(s).strip().lower() for s in scopes if s]
        if clean_scopes:
            pass

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# â”€â”€ Main entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def self_query(query: str, llm: AsyncLLMClient) -> dict:
    """
    Gá»i LLM Ä‘á»ƒ phÃ¢n tÃ­ch cÃ¢u há»i â†’ tráº£ vá»:
    {
        "search_query" : str,          # query rÃºt gá»n Ä‘á»ƒ vector/BM25
        "qdrant_filter": Filter | None, # filter object cho Qdrant
        "filter_mode"  : str,          # strict / relaxed / none
        "raw_parsed"   : dict,         # JSON gá»‘c LLM tráº£ vá» (debug)
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

