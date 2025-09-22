# Baseline RAG Evaluation — HSBC Q1

This folder contains the **locked baseline test** for RAGAS evaluation on the HSBC Q1 2025 transcript. It serves as the reference point for all future retrieval changes (e.g. Bank filtering, document expansion, prompt tuning, etc).

All inputs, outputs, code, and visualisations are preserved exactly as used on **15 Sep 2025**, prior to any code changes.

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `HSBC_Q1-12_questions.xlsx` | Raw input questions used in the test (no bank filtering) |
| `HSBC_Q1_2025_transcript.pdf` | Original source document used for retrieval |
| `hsbc_q1_ragas_rows.jsonl` | Output of `ragas_builder.py` — structured context/fact rows per question |
| `hsbc_q1_ragas_results_20250915_131518.csv` | Output of `ragas_eval.py` — per-question precision and recall |
| `hsbc_q1_ragas_results_20250915_131518.json` | Redundant early export (not used) — kept for completeness |
| `context_precision_recall_chart.png` | Visual output from `ragas_report.py`, showing precision/recall per question |
| `baseline_metrics.txt` | Manually captured summary of average precision and recall |
| `ragas_builder.py` | Version of the builder script used to generate rows |
| `ragas_eval.py` | Version of the evaluator script used for scoring |
| `ragas_report.py` | Script used to produce the visual bar chart |

---

## 📌 Notes

- This baseline does **not** include any filtering by bank.
- It uses only a single document: the HSBC Q1 2025 transcript.
- No post-processing, prompt augmentation, or LLM logic changes were applied.
- All changes going forward should be compared against these outputs.

---

## 🧪 Metrics

Average context scores from this run:

```
context_precision:  0.535
context_recall:     0.792
```

(See `hsbc_q1_ragas_results_20250915_131518.csv` and `context_precision_recall_chart.png` for full breakdown.)