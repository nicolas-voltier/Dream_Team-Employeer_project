import json
import os
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

INPUT_FILE = "barclays_20_input_tau_0.7.jsonl"  # change as needed

# Generate timestamped output filenames
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"barclays_20_tau_07_ragas_results_{STAMP}.csv" # change as needed
OUTPUT_JSON = f"barclays_20_tau_07_ragas_results_{STAMP}.json" # change as needed

# Load environment variables from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found. Please set it in your .env file.")

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    rows = load_rows(INPUT_FILE)

    ds = Dataset.from_list([
        {
            "question": r["question"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    llm = ChatOpenAI()  # ← default: gpt-3.5-turbo, temp=1
    result = evaluate(ds, metrics=[context_precision, context_recall], llm=llm)

    print("=== RAGAS Metrics ===")
    print(result)

    try:
        df = result.to_pandas()
        df.to_csv(OUTPUT_CSV, index=False)
        df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)
        print(f"Results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")
    except Exception as e:
        print(f"Could not save results: {e}")

if __name__ == "__main__":
    main()
