import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EMBED_MODEL, FINAL_TOP_K, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.retriever import search


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        if "question" not in data:
            raise ValueError(f"Missing 'question' in {path} at line {line_no}.")
        records.append(data)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def normalize_reference(record: dict[str, Any]) -> str | None:
    for key in ("reference", "ground_truth", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_reference_contexts(record: dict[str, Any]) -> list[str] | None:
    value = record.get("reference_contexts")
    if not isinstance(value, list):
        return None
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or None


def format_context(chunk: dict[str, Any]) -> str:
    title = (chunk.get("title") or "").strip()
    source_file = (chunk.get("source_file") or "").strip()
    body = (chunk.get("raw_text") or "").strip()

    lines: list[str] = []
    if title:
        lines.append(f"Title: {title}")
    if source_file:
        lines.append(f"Source: {source_file}")
    if body:
        lines.append(body)
    return "\n".join(lines).strip()


async def collect_rag_samples(
    rows: list[dict[str, Any]],
    max_contexts: int,
    verbose_search: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        question = str(row["question"]).strip()
        result = await search(question, verbose=verbose_search)
        contexts: list[str] = []
        for chunk in result.get("parent_sources", [])[:max_contexts]:
            rendered = format_context(chunk)
            if rendered:
                contexts.append(rendered)

        eval_row: dict[str, Any] = {
            "user_input": question,
            "response": result.get("answer", ""),
            "retrieved_contexts": contexts,
        }

        reference = normalize_reference(row)
        if reference:
            eval_row["reference"] = reference

        reference_contexts = normalize_reference_contexts(row)
        if reference_contexts:
            eval_row["reference_contexts"] = reference_contexts

        evaluation_rows.append(eval_row)
        debug_rows.append(
            {
                "id": row.get("id", idx),
                "question": question,
                "reference": reference,
                "reference_contexts": reference_contexts,
                "answer": result.get("answer", ""),
                "latency_ms": result.get("latency_ms"),
                "contexts": contexts,
                "top_chunks": result.get("top_chunks", []),
                "parent_sources": result.get("parent_sources", []),
            }
        )
        print(f"[collect {idx}/{len(rows)}] {question}")

    return evaluation_rows, debug_rows


def _metric_alias_map() -> dict[str, str]:
    return {
        "faithfulness": "faithfulness",
        "response_relevancy": "answer_relevancy",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
        "factual_correctness": "factual_correctness",
    }


def _import_metrics() -> dict[str, Any]:
    try:
        from ragas.metrics import (
            Faithfulness,
            FactualCorrectness,
            LLMContextPrecisionWithReference,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
        from ragas.metrics.collections import FactualCorrectness

    return {
        "Faithfulness": Faithfulness,
        "FactualCorrectness": FactualCorrectness,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextPrecisionWithoutReference": LLMContextPrecisionWithoutReference,
        "LLMContextRecall": LLMContextRecall,
        "ResponseRelevancy": ResponseRelevancy,
    }


def build_metrics(
    metric_names: list[str],
    has_reference: bool,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> list[Any]:
    classes = _import_metrics()
    metrics: list[Any] = []

    for metric_name in metric_names:
        if metric_name == "faithfulness":
            metrics.append(classes["Faithfulness"](llm=evaluator_llm))
        elif metric_name == "answer_relevancy":
            metrics.append(
                classes["ResponseRelevancy"](
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )
            )
        elif metric_name == "context_precision":
            klass = (
                classes["LLMContextPrecisionWithReference"]
                if has_reference
                else classes["LLMContextPrecisionWithoutReference"]
            )
            metrics.append(klass(llm=evaluator_llm))
        elif metric_name == "context_recall":
            if has_reference:
                metrics.append(classes["LLMContextRecall"](llm=evaluator_llm))
        elif metric_name == "factual_correctness":
            if has_reference:
                metrics.append(classes["FactualCorrectness"](llm=evaluator_llm))
    return metrics


def parse_metric_names(metric_arg: str) -> list[str]:
    aliases = _metric_alias_map()
    metric_names: list[str] = []
    for item in metric_arg.split(","):
        raw = item.strip().lower()
        if not raw:
            continue
        if raw not in aliases:
            raise ValueError(
                f"Unsupported metric '{raw}'. Supported: {', '.join(sorted(aliases))}"
            )
        canonical = aliases[raw]
        if canonical not in metric_names:
            metric_names.append(canonical)
    if not metric_names:
        raise ValueError("At least one metric is required.")
    return metric_names


def summarize_dataframe(df: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_samples": int(len(df)),
    }
    numeric_columns = []
    for column in df.columns:
        if column in {"user_input", "response", "reference", "retrieved_contexts", "reference_contexts"}:
            continue
        series = df[column]
        if getattr(series, "dtype", None) is None:
            continue
        if str(series.dtype) not in {"float64", "float32", "int64", "int32"}:
            continue
        numeric_columns.append(column)

    metric_averages: dict[str, float] = {}
    for column in numeric_columns:
        values = [float(v) for v in df[column].tolist() if v is not None and not math.isnan(float(v))]
        if values:
            metric_averages[column] = round(sum(values) / len(values), 4)
    summary["metric_averages"] = metric_averages
    return summary


def tag_weak_cases(df: Any) -> Any:
    thresholds = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.7,
        "context_recall": 0.7,
        "factual_correctness": 0.7,
    }

    weak_flags: list[bool] = []
    weak_reasons: list[str] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        for metric_name, threshold in thresholds.items():
            if metric_name not in row:
                continue
            value = row[metric_name]
            if value is None:
                continue
            try:
                if float(value) < threshold:
                    reasons.append(f"{metric_name}<{threshold}")
            except Exception:
                continue
        weak_flags.append(bool(reasons))
        weak_reasons.append(", ".join(reasons))

    df["is_weak_case"] = weak_flags
    df["weak_reasons"] = weak_reasons
    return df


def ensure_runtime_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("ragas", "pandas", "langchain_openai", "langchain_community"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        raise ModuleNotFoundError(
            "Missing dependencies for RAG evaluation: "
            + ", ".join(missing)
            + ". Install from requirements.txt before running this script."
        )


def build_evaluator_clients(judge_model: str, judge_base_url: str, judge_api_key: str) -> tuple[Any, Any]:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    lc_llm = ChatOpenAI(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        temperature=0.0,
    )
    lc_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainLLMWrapper(lc_llm), LangchainEmbeddingsWrapper(lc_embeddings)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current RAG pipeline with ragas.")
    parser.add_argument("--dataset", required=True, help="Path to dataset .jsonl")
    parser.add_argument("--output-dir", default="evaluation/reports", help="Output directory")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall,factual_correctness",
        help="Comma-separated metric list",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=FINAL_TOP_K,
        help="How many top L1 contexts from search() are passed into evaluation",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit dataset size for quick test")
    parser.add_argument("--judge-model", default=LLM_MODEL, help="Judge model for ragas")
    parser.add_argument("--judge-base-url", default=LLM_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--judge-api-key", default=LLM_API_KEY, help="Judge API key")
    parser.add_argument(
        "--verbose-search",
        action="store_true",
        help="Print the full retrieval pipeline while collecting answers",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(dataset_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if args.max_contexts < 1:
        raise ValueError("--max-contexts must be >= 1")

    metric_names = parse_metric_names(args.metrics)
    ref_flags = [bool(normalize_reference(row)) for row in rows]
    has_reference = all(ref_flags)
    has_partial_reference = any(ref_flags) and not has_reference
    if has_partial_reference:
        raise ValueError(
            "Dataset contains a mix of samples with and without reference answers. "
            "Either add `reference` for all rows or run only no-reference metrics."
        )

    print(f"Loaded {len(rows)} samples from {dataset_path}")
    print(f"Metrics: {', '.join(metric_names)}")
    print(f"Reference answers present: {has_reference}")

    evaluation_rows, debug_rows = asyncio.run(
        collect_rag_samples(
            rows=rows,
            max_contexts=args.max_contexts,
            verbose_search=args.verbose_search,
        )
    )

    collected_path = output_dir / "collected_dataset.json"
    run_details_path = output_dir / "run_details.json"
    save_json(collected_path, evaluation_rows)
    save_json(run_details_path, debug_rows)

    ensure_runtime_dependencies()

    from ragas import EvaluationDataset, evaluate

    evaluator_llm, evaluator_embeddings = build_evaluator_clients(
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
    )
    metrics = build_metrics(
        metric_names=metric_names,
        has_reference=has_reference,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
    )
    if not metrics:
        raise ValueError("No metrics could be built from the current dataset and --metrics selection.")

    evaluation_dataset = EvaluationDataset.from_list(evaluation_rows)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    df = tag_weak_cases(df)

    csv_path = output_dir / "per_sample_scores.csv"
    summary_path = output_dir / "overall_metrics.json"
    weak_cases_path = output_dir / "weak_cases.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    summary = summarize_dataframe(df)
    save_json(summary_path, summary)
    save_json(
        weak_cases_path,
        df[df["is_weak_case"]].to_dict(orient="records"),
    )

    print("\n=== RAGAS Evaluation Completed ===")
    print(f"Collected dataset : {collected_path}")
    print(f"Per-sample scores : {csv_path}")
    print(f"Summary           : {summary_path}")
    print(f"Weak cases        : {weak_cases_path}")
    print(f"Run details       : {run_details_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
