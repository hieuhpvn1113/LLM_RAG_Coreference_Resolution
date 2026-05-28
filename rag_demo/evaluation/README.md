# Evaluation (Manual-Path Based)

Muc tieu: kiem tra RAG theo dung luong ban test tay, tuc la moi cau hoi deu chay:

`python main.py query "<question>"`

## Chay danh gia

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.auto.jsonl
```

## Tuy chon

- `--output-dir evaluation\reports_v2` : thu muc output
- `--f1-threshold 0.65` : nguong pass/fail
- `--timeout-sec 180` : timeout moi query

## File ket qua

- `overall_metrics.json`: tong quan pass rate, avg F1, avg latency
- `per_sample_scores.csv`: ket qua tung cau hoi
- `failed_cases.json`: cac cau fail de debug
- `run_details.json`: chi tiet day du cua run

