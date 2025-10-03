# RAGAS Evaluation: Barclays 20-Question Run (Use Case τ = 0.6)

This folder contains the full set of inputs, scripts, and outputs for the Barclays 20-question evaluation run using RAGAS with a threshold of **τ = 0.6**.

## 🔁 Commands Run

```bash
python ragas_builder.py
```
This built the input file and saved:
```
Saved 20 rows to barclays_20_input.jsonl
```

```bash
python ragas_eval.py
```
This ran the evaluation using GPT and printed:
```
Evaluating: 100%|██████████████████████████████████████████████████████████████████████| 40/40 [01:27<00:00,  2.18s/it]
=== RAGAS Metrics ===
{'context_precision': 0.4567, 'context_recall': 0.4750}
Results saved to barclays_20_ragas_results_20250930_101753.csv and barclays_20_ragas_results_20250930_101753.json
```

## 📄 Output Location

All paths are relative to the project root (sensitive directories masked).  
This run's outputs are stored in:

```
.../employer_project/results_david/barclays_20q/
├── input/
│   └── barclays_20_input.jsonl
├── output_tau_0.6/
│   ├── barclays_20_ragas_results_20250930_101753.csv
│   └── barclays_20_ragas_results_20250930_101753.json
├── scripts_tau_0.6/
│   ├── ragas_builder.py
│   ├── ragas_eval.py
│   └── ragas_report.py
```

## 📊 Notes on RAGAS Metrics

The values printed:
```python
{'context_precision': 0.4567, 'context_recall': 0.4750}
```
are the **simple arithmetic means** across all 20 questions.

You can verify this by averaging the values in the corresponding columns in the CSV file:

| context_precision | context_recall |
|-------------------|----------------|
| 0.598135198       | 0              |
| 0.483446712       | 0              |
| 0.573726274       | 1              |
| 0.757575758       | 1              |
| 0                 | 0              |
| 1                 | 1              |
| 0.15              | 0              |
| 0.333333333       | 1              |
| 0                 | 0              |
| 0.112121212       | 0.5            |
| 1                 | 1              |
| 0.275793651       | 0              |
| 0                 | 1              |
| 1                 | 0              |
| 0                 | 0              |
| 1                 | 1              |
| 0.551587302       | 0.5            |
| 0.71494709        | 0.5            |
| 0                 | 0              |
| 0.583333333       | 1              |

These average to:
- **context_precision ≈ 0.4567**
- **context_recall ≈ 0.4750**

---
