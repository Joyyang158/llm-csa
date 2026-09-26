"""Capability Ratio: post/pre mean per-attempt solve accuracy, times 100.

Both CSVs must cover the same queries in the same order, with the same
positive number of independent attempts per query (five in the paper).
For science, each shuffled attempt is graded against its remapped answer.
Any-correct / majority-correct aggregation is used for CSA labels only.
"""

import argparse
import logging

import pandas as pd

from src.utils.data import validate_query_alignment
from src.utils.grading import compute_attempt_correctness


def accuracy_from_frame(df, generations_col, domain):
    scores = compute_attempt_correctness(df, generations_col, domain)
    counts = {len(row) for row in scores}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("CR requires the same positive number of attempts for every query.")
    k = counts.pop()
    n_attempts = len(scores) * k
    n_correct = sum(map(sum, scores))
    return n_correct / n_attempts, n_correct, n_attempts, k


def solve_accuracy(csv_path: str, generations_col: str, domain: str) -> tuple:
    """Return mean accuracy, correct attempts, and total attempts."""
    return accuracy_from_frame(pd.read_csv(csv_path), generations_col, domain)[:3]


def main():
    parser = argparse.ArgumentParser(description="Compute Capability Ratio (CR).")
    parser.add_argument("--pre_csv", required=True)
    parser.add_argument("--post_csv", required=True)
    parser.add_argument("--generations_col", default="generation")
    parser.add_argument("--domain", choices=["math", "science"], required=True)
    args = parser.parse_args()

    pre = pd.read_csv(args.pre_csv)
    post = pd.read_csv(args.post_csv)
    validate_query_alignment(pre, post)
    pre_acc, pre_correct, pre_n, pre_k = accuracy_from_frame(pre, args.generations_col, args.domain)
    post_acc, post_correct, post_n, post_k = accuracy_from_frame(post, args.generations_col, args.domain)
    if pre_k != post_k:
        raise ValueError("Pre/post CSVs must have the same number of attempts per query.")
    cr = f"{post_acc / pre_acc * 100.0:.1f}%" if pre_acc > 0 else "NA (zero pre-training accuracy)"

    print("\n=========== Capability Ratio (CR) Report ===========")
    print(f"Domain                : {args.domain}")
    print(f"Queries / attempts    : {len(pre)} / {pre_k} per query")
    print(f"Pre-training CSV      : {args.pre_csv}")
    print(f"  Acc(pre)            : {pre_acc:.4f}  ({pre_correct}/{pre_n} attempts)")
    print(f"Post-training CSV     : {args.post_csv}")
    print(f"  Acc(post)           : {post_acc:.4f}  ({post_correct}/{post_n} attempts)")
    print(f"CR                    : {cr}")
    print("====================================================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
