import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Directory to look for result files
RESULTS_DIR = "."
DEFAULT_PATTERN = "barclays_20_tau_09_ragas_results_*.csv"

def get_latest_csv():
    files = glob.glob(os.path.join(RESULTS_DIR, DEFAULT_PATTERN))
    if not files:
        raise FileNotFoundError("No results CSV found. Run ragas_eval.py first.")
    return max(files, key=os.path.getctime)

def main(file=None):
    # Pick latest CSV if not provided
    csv_file = file if file else get_latest_csv()
    print(f"Loading results from {csv_file}")

    df = pd.read_csv(csv_file)

    # Add numeric IDs since CSV has none
    df["question_id"] = range(1, len(df) + 1)

    print("\n=== Per-question metrics ===")
    print(df[["question_id", "context_precision", "context_recall"]])

    print("\n=== Averages ===")
    print(df[["context_precision", "context_recall"]].mean())

    # Compute average score for sorting
    df["avg_score"] = (df["context_precision"] + df["context_recall"]) / 2
    df_sorted = df.sort_values(by="avg_score", ascending=False)

    # Plot sorted by average score
    ax = df_sorted.plot(
        x="question_id",
        y=["context_precision", "context_recall"],
        kind="bar",
        figsize=(14, 6),
        rot=0,
        width=0.7
    )
    plt.title("Context Precision and Recall per Question (sorted by avg score)")
    plt.xlabel("Question ID")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()