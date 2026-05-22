"""Grade model answers on a benchmark CSV.

Replaces the previous trio of standalone evaluation scripts. All grading
logic lives in :mod:`src.utils.grading`; this script is a thin CLI wrapper
that loads a CSV, runs grading, prints per-attempt / per-source breakdowns,
and optionally writes the dataframe back out with an ``is_correct`` column.

Two modes:

  * ``single`` — each row's eval column holds a single generation.
  * ``multi``  — each row's eval column holds a list of generations.
                 For ``math``, a row is correct under pass@1 (any attempt
                 correct). For ``science``, a row is correct under majority
                 vote across the shuffled attempts.
"""

import argparse
import os

import pandas as pd

from src.utils.grading import grade_dataframe


def main():
    parser = argparse.ArgumentParser(description="Grade benchmark answers.")
    parser.add_argument("--csv_path", required=True,
                        help="CSV with model generations to be graded.")
    parser.add_argument("--eval_col", required=True,
                        help="Column holding the generation(s).")
    parser.add_argument("--domain", choices=["math", "science"], required=True)
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--output_path", default=None,
                        help="Optional path to save the dataframe with the "
                             "added ``is_correct`` column.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(f"  File        : {args.csv_path}")

    df = grade_dataframe(
        df=df,
        generations_col=args.eval_col,
        domain=args.domain,
        mode=args.mode,
    )

    if args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output_path, index=False)
        print(f"\nSaved to: {args.output_path}")


if __name__ == "__main__":
    main()
