import pandas as pd
import json
import asyncio
from process_graph import GraphProcessor

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = "data/Stress-Test_Questions.xlsx" # change as needed
OUTPUT_FILE = "stress_test_ragas_input.jsonl" # change as needed
THRESHOLD = 0.6
TOP_K = 5
DOC_LIMIT = 3
BANK_FILTER = "Barclays"  # Set to None to disable filtering

# -----------------------------
# Processor
# -----------------------------
gp = GraphProcessor()

def parse_query_output(raw_output: str):
    """
    Parse the string output from query_graph into structured dicts.
    Expected lines:
      - Fact: ...
      - From page: ...
      - Similarity: ...
    """
    results = []
    if not raw_output:
        return results

    current = {}
    for line in raw_output.splitlines():
        line = line.strip()
        if line.startswith("- Fact:"):
            if current:
                results.append(current)
                current = {}
            current["fact"] = line.replace("- Fact:", "").strip()
        elif line.startswith("- From page:"):
            current["page"] = line.replace("- From page:", "").strip()
        elif line.startswith("- Similarity:"):
            sim = line.replace("- Similarity:", "").strip()
            try:
                current["similarity"] = float(sim)
            except ValueError:
                current["similarity"] = None

    if current:
        results.append(current)
    return results

async def build_rows():
    try:
        df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    except ImportError:
        raise ImportError("Missing dependency: openpyxl. Install with `pip install openpyxl`.")

    rows = []
    for _, row in df.iterrows():
        bank = row.get("Bank", None)
        if BANK_FILTER and bank and bank != BANK_FILTER:
            continue  # Skip rows not matching the filter

        q = str(row["Question"]).strip()
        gt = str(row["Expected Answer"]).strip()

        raw_output = await gp.query_graph(
            q,
            threshold=THRESHOLD,
            limit=TOP_K,
            doc_limit=DOC_LIMIT,
            print_out=False
        )
        results = parse_query_output(raw_output)

        row_dict = {
            "question": q,
            "ground_truth": gt,
            "contexts": [r.get("fact") for r in results if "fact" in r],
            "contexts_meta": [
                {"page": r.get("page"), "similarity": r.get("similarity")}
                for r in results
            ],
            "retrieval_threshold": THRESHOLD,
            "top_k": TOP_K,
        }
        rows.append(row_dict)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(build_rows())