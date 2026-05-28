import argparse
import csv
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fold_ascii(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return normalize_text(text)


def token_f1(pred: str, ref: str) -> float:
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()
    if not p or not r:
        return 0.0

    p_count = {}
    r_count = {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in r:
        r_count[t] = r_count.get(t, 0) + 1

    overlap = 0
    for t, c in p_count.items():
        overlap += min(c, r_count.get(t, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_answer(raw_output: str) -> str:
    lines = raw_output.splitlines()
    for i, line in enumerate(lines):
        marker = fold_ascii(line)
        if "cau tra loi" in marker or ("tra loi" in marker and "ms" in marker):
            for follow in lines[i + 1 :]:
                s = follow.strip()
                if not s or set(s) <= {"="}:
                    continue
                return s

    for line in reversed(lines):
        s = line.strip()
        if s and set(s) != {"="}:
            return s
    return ""


def run_manual_query(question: str, timeout_sec: int) -> dict:
    cmd = [sys.executable, "main.py", "query", question]
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        env=env,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    answer = extract_answer(stdout)

    latency_ms = None
    m = re.search(r"\((\d+)ms\)", stdout)
    if m:
        latency_ms = int(m.group(1))

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "answer": answer,
        "latency_ms": latency_ms,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Evaluate RAG by running the same manual CLI path as python main.py query"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset .jsonl")
    parser.add_argument("--output-dir", default="evaluation/reports_v2", help="Output directory")
    parser.add_argument("--f1-threshold", type=float, default=0.65, help="Threshold for pass/fail")
    parser.add_argument("--timeout-sec", type=int, default=180, help="Timeout per query")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(dataset_path)
    if not rows:
        raise ValueError("Dataset is empty.")

    results = []
    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        ground_truth = row.get("ground_truth", "")

        run = run_manual_query(question, timeout_sec=args.timeout_sec)
        pred = run["answer"]

        pred_n = normalize_text(pred)
        gt_n = normalize_text(ground_truth)
        exact = pred_n == gt_n and gt_n != ""
        contains = (gt_n and gt_n in pred_n) or (pred_n and pred_n in gt_n)
        f1 = token_f1(pred, ground_truth)
        passed = (exact or contains) and f1 >= args.f1_threshold and run["returncode"] == 0

        results.append(
            {
                "id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": pred,
                "exact_match": exact,
                "contains_match": contains,
                "token_f1": round(f1, 4),
                "passed": passed,
                "latency_ms": run["latency_ms"],
                "returncode": run["returncode"],
                "stderr": run["stderr"].strip(),
            }
        )
        print(f"[{idx}/{len(rows)}] pass={passed} f1={f1:.3f} | {question}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed = [r for r in results if not r["passed"]]

    avg_f1 = sum(r["token_f1"] for r in results) / total if total else 0.0
    avg_latency = sum((r["latency_ms"] or 0) for r in results) / total if total else 0.0
    summary = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(passed_count / total, 4) if total else 0.0,
        "avg_token_f1": round(avg_f1, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "f1_threshold": args.f1_threshold,
    }

    json_path = output_dir / "run_details.json"
    csv_path = output_dir / "per_sample_scores.csv"
    summary_path = output_dir / "overall_metrics.json"
    fail_path = output_dir / "failed_cases.json"

    json_path.write_text(
        json.dumps({"summary": summary, "records": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fail_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(results[0].keys()) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Evaluation Completed (Manual-Path Based) ===")
    print(f"Summary: {summary}")
    print(f"Per-sample: {csv_path}")
    print(f"Overall: {summary_path}")
    print(f"Failed cases: {fail_path}")
    print(f"Details: {json_path}")


if __name__ == "__main__":
    main()
