import argparse
import os

import pandas as pd


DEFAULT_INPUT = os.path.join("data", "combined", "matched_pairs.jsonl")
DEFAULT_OUTPUT = os.path.join(
    "data", "combined", "match_review_sample.csv"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a small random sample of matched pairs for "
            "fast manual quality review."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to matched_pairs.jsonl",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save the review CSV",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows to sample for manual review",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Matched pairs not found at {args.input}")

    df = pd.read_json(args.input, lines=True)
    sample_size = min(args.sample_size, len(df))
    sample = df.sample(n=sample_size, random_state=args.seed).copy()

    sample["manual_match_label"] = ""
    sample["manual_notes"] = ""

    review_columns = [
        "matched_patient_uid",
        "patient_context",
        "question",
        "ground_truth",
        "hallucinated_ans",
        "difficulty",
        "match_type",
        "match_confidence",
        "cosine_similarity",
        "bridge_cosine",
        "manual_match_label",
        "manual_notes",
    ]
    review_columns = [
        column for column in review_columns if column in sample.columns
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sample[review_columns].to_csv(args.output, index=False)

    print(f"Saved review sample: {args.output}")
    print(f"Rows: {sample_size}")
    if "match_confidence" in sample.columns:
        print("Confidence mix:")
        print(sample["match_confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()
