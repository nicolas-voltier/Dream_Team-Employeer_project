# Bank Filter Verification

This folder contains a minimal test to verify the BANK_FILTER logic added to `ragas_builder.py`.

- `test_bank_filter_questions.xlsx` contains 3 questions: 2 tagged HSBC, 1 tagged Barclays.
- `ragas_builder.py` was run with `BANK_FILTER = "HSBC"`.

✅ Output file (`hsbc_q1_ragas_rows_test.jsonl`) correctly includes only the 2 HSBC rows.  
❌ Barclays question was excluded as expected.

This confirms that per-bank filtering works before running the full baseline.
