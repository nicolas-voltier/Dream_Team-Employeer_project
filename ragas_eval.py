import json
import os
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from dotenv import load_dotenv

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

INPUT_FILE = "hsbc_q1_ragas_rows.jsonl"

# Generate timestamped output filenames
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"hsbc_q1_ragas_results_{STAMP}.csv"
OUTPUT_JSON = f"hsbc_q1_ragas_results_{STAMP}.json"

# Load environment variables from .env
load_dotenv()

# Ensure API key is visible
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

    # Build dataset for RAGAS
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    # Run evaluation (LLM-backed, requires OPENAI_API_KEY)
    result = evaluate(ds, metrics=[context_precision, context_recall])
    
    custom_llm = ChatOpenAI(model="gpt-5-mini") 
    ragas_llm = LangchainLLMWrapper(custom_llm)
    result = evaluate(ds, metrics=[context_precision, context_recall],llm=ragas_llm)

    print("=== RAGAS Metrics ===")
    print(result)

    # --- Save results (timestamped to avoid conflicts) ---
    try:
        df = result.to_pandas()
        df.to_csv(OUTPUT_CSV, index=False)
        df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)
        print(f"Results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")
    except Exception as e:
        print(f"Could not save results: {e}")
    # -------------------------------------------------------------

if __name__ == "__main__":
    main()
