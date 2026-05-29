# RAG Evaluation With Ragas

Thu muc nay dung de test va cham diem he thong RAG hien tai bang `ragas`.

Script `evaluate_ragas.py` se:
- Goi truc tiep `core.retriever.search()`
- Lay `answer` va danh sach `retrieved_contexts` ma pipeline tra ve
- Chay `ragas` de tinh diem cho tung cau hoi
- Xuat bao cao tong hop va bao cao chi tiet

## 1. Cai dependency

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python -m pip install -r requirements.txt
```

Neu ban dung virtualenv:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Chuan bi dataset

Mac dinh script doc file `.jsonl`, moi dong la 1 mau test.

Toi thieu:

```json
{"question": "Cong ty nao se to chuc cuoc hop?", "reference": "Cong ty Co phan Sua Viet Nam (Vinamilk) se to chuc cuoc hop."}
```

Field ho tro:
- `question`: cau hoi bat buoc
- `reference`: dap an chuan de cham `context_recall` va `factual_correctness`
- `ground_truth`: alias cua `reference`
- `answer`: cung duoc dung lam alias cua `reference`
- `reference_contexts`: danh sach context chuan neu ban muon luu de debug retrieval

Ban co the copy tu file [dataset.sample.jsonl](E:\AI_agent\LLM_RAG\rag_demo\evaluation\dataset.sample.jsonl).

## 3. Chay danh gia

```powershell
cd E:\AI_agent\LLM_RAG\rag_demo
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.sample.jsonl
```

Chay nhanh voi it sample:

```powershell
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.sample.jsonl --max-samples 2
```

Bat log retrieval day du:

```powershell
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.sample.jsonl --verbose-search
```

## 4. Metrics dang ho tro

Mac dinh script bat:
- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`
- `factual_correctness`

Neu dataset khong co `reference`, script van co the cham:
- `faithfulness`
- `answer_relevancy`
- `context_precision`

Chon metrics thu cong:

```powershell
python evaluation\evaluate_ragas.py --dataset evaluation\dataset.sample.jsonl --metrics faithfulness,answer_relevancy,context_precision
```

## 5. File output

Mac dinh output nam trong `evaluation/reports/`:
- `collected_dataset.json`: du lieu da thu tu pipeline RAG truoc khi cham
- `per_sample_scores.csv`: diem tung sample
- `overall_metrics.json`: trung binh tung metric
- `weak_cases.json`: cac sample co metric thap hon nguong canh bao
- `run_details.json`: chi tiet answer, contexts, top chunks, parent sources

## 6. Luu y thuc te

- Judge LLM mac dinh dung cung server OpenAI-compatible trong `config.py`
- Embedding judge dung model local `intfloat/multilingual-e5-large`
- `context_recall` va `factual_correctness` chi co y nghia khi dataset co dap an chuan tot
- Neu judge model yeu, diem `ragas` se dao dong. Nen giu 1 model judge co dinh de so sanh cac lan chay

## 7. Tham khao

Script nay lam theo workflow chinh thuc cua Ragas: thu dataset gom `user_input`, `response`, `retrieved_contexts`, `reference`, sau do goi `evaluate(...)` voi metric classes nhu `Faithfulness`, `LLMContextRecall`, `FactualCorrectness`.

Nguon:
- Ragas guide: https://docs.ragas.io/en/v0.2.11/getstarted/rag_eval/
- Ragas metric customization: https://docs.ragas.io/en/v0.2.9/howtos/customizations/customize_models/
- Ragas context precision: https://docs.ragas.io/en/v0.2.11/concepts/metrics/available_metrics/context_precision/
- Ragas context recall: https://docs.ragas.io/en/v0.2.3/concepts/metrics/available_metrics/context_recall/
