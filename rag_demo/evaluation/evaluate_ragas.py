import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.retriever import search


def _load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 1) Try JSON array first.
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list of objects.")
        return data

    # 2) Try strict JSONL (one JSON object per line).
    rows = []
    ok_jsonl = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            ok_jsonl = False
            break
    if ok_jsonl and rows:
        return rows

    # 3) Fallback: multiple pretty JSON objects separated by blank lines.
    decoder = json.JSONDecoder()
    rows = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError("Each dataset item must be a JSON object.")
        rows.append(obj)
        idx = next_idx
    return rows


async def _run_rag(dataset_rows: list[dict], verbose: bool) -> list[dict]:
    records = []
    for idx, row in enumerate(dataset_rows, start=1):
        question = row["question"]
        result = await search(question, verbose=verbose)
        contexts = [c.get("raw_text", "") for c in result.get("parent_sources", [])]
        records.append(
            {
                "id": idx,
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": row.get("ground_truth", ""),
            }
        )
    return records


def _strip_citation_tags(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = _strip_citation_tags(text)
    text = re.sub(r"[^\w\s%/\.:-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _is_factoid_question(question: str) -> bool:
    q = (question or "").lower()
    keys = ["bao nhiêu", "khi nào", "ngày", "%", "tỷ lệ", "mấy", "là gì", "bao lâu", "thời gian"]
    return any(k in q for k in keys) or bool(re.search(r"\d", q))


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_text(pred).split()
    g_toks = _normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    hit = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            hit += 1
            common[t] -= 1
    if hit == 0:
        return 0.0
    precision = hit / len(p_toks)
    recall = hit / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _build_judge_models() -> tuple[ChatOpenAI, HuggingFaceEmbeddings]:
    base_url = LLM_BASE_URL.rstrip("/")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=base_url,
        temperature=0,
    )
    # Use local embeddings to avoid dependency on /embeddings endpoint
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    return llm, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality with Ragas")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output-dir", default="evaluation/reports", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose RAG pipeline logs")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError("Dataset is empty")

    rag_records = asyncio.run(_run_rag(rows, verbose=args.verbose))

    eval_rows = []
    for row in rag_records:
        clean_answer = _strip_citation_tags(row["answer"])
        eval_rows.append(
            {
                "question": row["question"],
                "answer": clean_answer,
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
        )

    hf_dataset = Dataset.from_list(eval_rows)
    llm, embeddings = _build_judge_models()

    result = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    score_df = result.to_pandas()
    em_list = []
    f1_list = []
    for r in eval_rows:
        if _is_factoid_question(r["question"]):
            em = 1.0 if _normalize_text(r["answer"]) == _normalize_text(r["ground_truth"]) else 0.0
            f1 = _token_f1(r["answer"], r["ground_truth"])
        else:
            em = None
            f1 = None
        em_list.append(em)
        f1_list.append(f1)
    score_df["em_factoid"] = em_list
    score_df["f1_factoid"] = f1_list
    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    overall = {}
    for col in metric_cols:
        if col in score_df.columns:
            overall[col] = float(score_df[col].dropna().mean())
    overall["em_factoid"] = float(score_df["em_factoid"].dropna().mean()) if score_df["em_factoid"].notna().any() else None
    overall["f1_factoid"] = float(score_df["f1_factoid"].dropna().mean()) if score_df["f1_factoid"].notna().any() else None

    score_path = output_dir / "per_sample_scores.csv"
    overall_path = output_dir / "overall_metrics.json"
    details_path = output_dir / "run_details.json"

    score_df.to_csv(score_path, index=False)
    overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    details_path.write_text(
        json.dumps({"records": rag_records, "overall": overall}, indent=2),
        encoding="utf-8",
    )

    print("=== Ragas Evaluation Completed ===")
    print(f"Samples: {len(rag_records)}")
    print(f"Overall metrics: {overall}")
    print(f"Per-sample scores: {score_path}")
    print(f"Overall metrics file: {overall_path}")
    print(f"Run details: {details_path}")


if __name__ == "__main__":
    main()
