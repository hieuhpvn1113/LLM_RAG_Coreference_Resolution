import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.meta_db import MetaDB
from llm.client import AsyncLLMClient

GEN_SYSTEM = """You generate high-quality RAG evaluation samples.
Return ONLY valid JSON array. No markdown, no explanation.
Each item format:
{
  \"question\": \"...\",
  \"ground_truth\": \"...\"
}
Rules:
- Questions must be answerable from the provided context only.
- Ground truth must be concise, factual, and directly supported by context.
- Use Vietnamese.
- Avoid duplicate questions.
"""

GEN_USER_TMPL = """Context:\n{context}\n\nCreate exactly {n} question-ground_truth pairs as required."""


def _extract_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output does not contain JSON array")
    return json.loads(text[start:end + 1])


async def _load_contexts(limit: int):
    db = MetaDB()
    await db.connect()
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_id::text, title, raw_text, source_file
                FROM chunks
                WHERE level = 1
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def main_async(args):
    contexts = await _load_contexts(args.context_limit)
    if not contexts:
        raise ValueError("No level-1 chunks found. Please ingest documents first.")

    llm = AsyncLLMClient()
    all_samples = []

    for c in contexts:
        title = (c.get("title") or "").strip()
        raw_text = (c.get("raw_text") or "").strip()
        if not raw_text:
            continue

        truncated = raw_text[: args.max_context_chars]
        header = f"Title: {title}\nSource: {c.get('source_file','')}\n"
        prompt = GEN_USER_TMPL.format(context=header + truncated, n=args.questions_per_context)

        out = await llm.complete(system=GEN_SYSTEM, user=prompt, max_tokens=1400)
        pairs = _extract_json_array(out)

        for p in pairs:
            q = str(p.get("question", "")).strip()
            gt = str(p.get("ground_truth", "")).strip()
            if q and gt:
                all_samples.append({"question": q, "ground_truth": gt})

    # de-dup by question
    seen = set()
    dedup = []
    for item in all_samples:
        key = item["question"].lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in dedup:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Auto Testset Generated ===")
    print(f"Contexts used: {len(contexts)}")
    print(f"Samples written: {len(dedup)}")
    print(f"Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-generate RAG eval dataset from ingested sources")
    parser.add_argument("--output", default="evaluation/dataset.auto.jsonl")
    parser.add_argument("--context-limit", type=int, default=8)
    parser.add_argument("--questions-per-context", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=3500)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
