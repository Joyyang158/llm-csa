"""Measure the average tokenised length of generations in a CSV column."""

import argparse

import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Compute the average tokenised length of a CSV column."
    )
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--column", required=True,
                        help="Column whose values should be tokenised.")
    parser.add_argument("--tokenizer", required=True,
                        help="HuggingFace tokenizer id (e.g. Qwen/Qwen3-0.6B).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.column not in df.columns:
        raise ValueError(f"Column not found: {args.column}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    total = 0
    count = 0
    for value in df[args.column]:
        if pd.isna(value):
            continue
        ids = tokenizer.encode(str(value), add_special_tokens=False)
        total += len(ids)
        count += 1

    avg = total / count if count > 0 else 0.0
    print(f"Tokenizer        : {args.tokenizer}")
    print(f"Column           : {args.column}")
    print(f"Rows tokenised   : {count}")
    print(f"Average length   : {avg:.2f}")


if __name__ == "__main__":
    main()
