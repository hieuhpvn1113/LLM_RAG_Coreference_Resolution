# Ragas Evaluation

## 1) Cai dat

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
pip install -r requirements.txt
```

## 2) Chay danh gia

```powershell
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.jsonl
```

## 2.1) Tu dong tao bo cau hoi + dap an (AI)

```powershell
python evaluation\generate_testset.py --output evaluation\dataset.auto.jsonl
```

Sau do danh gia bo vua tao:

```powershell
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.auto.jsonl
```

## 3) Tuy chon

- `--output-dir evaluation\reports` : thu muc output
- `--verbose` : in log retrieval/generation

## 4) Ket qua

Script se tao:
- `overall_metrics.json`: diem tong the
- `per_sample_scores.csv`: diem tung cau hoi
- `run_details.json`: du lieu chi tiet run
