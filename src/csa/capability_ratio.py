"""Capability Ratio (CR) — paper Eq. 8.

CR measures whether CSA training preserves the model's underlying problem-
solving ability. It is the percentage of the base model's solve accuracy
retained after CSA training:

                Acc(post-training)
        CR = -------------------- × 100%
                Acc(pre-training)

Both accuracies are measured on the same evaluation set, and (per Appendix
D.5) each is averaged over multiple independent decoding runs to reduce
variance from stochastic sampling.

Inputs to this script are two CSVs of *raw* generations from the same set of
queries — one from the base model, one from the post-training model. Each
CSV's generations column should be a list of N independent samples per row
(produced by ``src/data/generate_answers.py`` with ``num_generations=5`` for
math or ``num_shuffles=5`` for science).

The script grades both CSVs with the same domain-specific aggregation rule
(any-correct for math, majority-vote for science — see ``src/utils/grading``),
prints both accuracies, and reports their ratio.
"""

import argparse
import logging

import pandas as pd

from src.utils.grading import compute_is_correct


def solve_accuracy(csv_path: str, generations_col: str, domain: str) -> tuple:
    df = pd.read_csv(csv_path)
    is_correct = compute_is_correct(df, generations_col, domain)
    n = len(is_correct)
    n_correct = int(is_correct.sum())
    acc = float(is_correct.mean()) if n > 0 else 0.0
    return acc, n_correct, n


def main():
    parser = argparse.ArgumentParser(description="Compute Capability Ratio (CR).")
    parser.add_argument("--pre_csv", required=True,
                        help="Generations CSV from the base (pre-training) model.")
    parser.add_argument("--post_csv", required=True,
                        help="Generations CSV from the post-training model.")
    parser.add_argument("--generations_col", default="generation",
                        help="Column holding the list of generations.")
    parser.add_argument("--domain", choices=["math", "science"], required=True,
                        help="Domain — sets the aggregation rule.")
    args = parser.parse_args()

    pre_acc, pre_correct, pre_n = solve_accuracy(
        args.pre_csv, args.generations_col, args.domain,
    )
    post_acc, post_correct, post_n = solve_accuracy(
        args.post_csv, args.generations_col, args.domain,
    )

    if pre_n != post_n:
        logging.warning(
            "Row counts differ: pre=%d, post=%d. CR comparisons assume the "
            "two CSVs cover the same evaluation queries.",
            pre_n, post_n,
        )

    cr = (post_acc / pre_acc * 100.0) if pre_acc > 0 else float("nan")

    print("\n=========== Capability Ratio (CR) Report ===========")
    print(f"Domain                : {args.domain}")
    print(f"Pre-training CSV      : {args.pre_csv}")
    print(f"  Acc(pre)            : {pre_acc:.4f}  ({pre_correct}/{pre_n})")
    print(f"Post-training CSV     : {args.post_csv}")
    print(f"  Acc(post)           : {post_acc:.4f}  ({post_correct}/{post_n})")
    print("----------------------------------------------------")
    print(f"CR                    : {cr:.1f}%")
    print("====================================================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
