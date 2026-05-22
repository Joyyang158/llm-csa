"""Evaluate CSA (Capability Self-Assessment) predictions against ground-truth
correctness labels.

Two grading modes:

  * ``column``      — read the per-row ``is_correct`` label directly from a
                      column in ``--csv_path``. Use this when correctness has
                      already been computed upstream.
  * ``generations`` — read raw model generations from ``--gt_csv_path`` and
                      grade them on the fly. The domain (math vs science)
                      determines the aggregation rule (any-correct vs
                      majority-vote).

Reported metrics (paper §4.1, Appendix D.5):

  1. **CDS** — Capability Discrimination Score. The unpooled two-proportion
     z-statistic between solve accuracy on rows the model labels SELF_SOLVE
     vs. DELEGATE. A larger CDS indicates a sharper, more statistically
     reliable separation between what the model can and cannot solve.
  2. **Accuracy** — fraction of CSA decisions that match ``is_correct``.
  3. **Self-Solve Rate (SSR)** — fraction of queries the model chooses to
     self-solve. Compared against the ground-truth SSR, this is the
     calibration signal in Figure 4 (left).
  4. **M-F1** — macro F1 averaged over SELF_SOLVE and DELEGATE classes.

Capability Ratio (CR), the second axis of evaluation in the paper, is
computed by a separate script: see ``src/csa/capability_ratio.py``.
"""

import argparse
import logging
import math

import pandas as pd

from src.utils.grading import compute_is_correct


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def f1_for_label(y_true: pd.Series, y_pred: pd.Series, pos_label: int) -> float:
    tp = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    fp = ((y_true != pos_label) & (y_pred == pos_label)).sum()
    fn = ((y_true == pos_label) & (y_pred != pos_label)).sum()
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """M-F1: unweighted mean of per-class F1 (paper Eq. 6)."""
    return (
        f1_for_label(y_true, y_pred, 0) + f1_for_label(y_true, y_pred, 1)
    ) / 2


def cds(p_s: float, n_s: int, p_d: float, n_d: int) -> float:
    """Capability Discrimination Score (paper Eq. 5).

    Unpooled two-proportion z-statistic comparing solve accuracy among rows
    the model predicts SELF_SOLVE (``p_s``, ``n_s``) vs. those it predicts
    DELEGATE (``p_d``, ``n_d``).
    """
    if n_s <= 0 or n_d <= 0:
        return 0.0
    se = math.sqrt(p_s * (1 - p_s) / n_s + p_d * (1 - p_d) / n_d)
    return (p_s - p_d) / se if se > 0 else 0.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_metrics(
    df: pd.DataFrame,
    prediction_col: str,
    total_rows: int,
    n_invalid: int,
    base_accuracy: float,
    extra_lines: list,
) -> None:
    y_pred = df[prediction_col]
    y_true = df["is_correct"]
    n_valid = len(df)
    invalid_ratio = n_invalid / total_rows if total_rows > 0 else 0.0

    n_pred_self = int((y_pred == 1).sum())
    n_pred_del = int((y_pred == 0).sum())

    p_s = (
        int(((y_pred == 1) & (y_true == 1)).sum()) / n_pred_self
        if n_pred_self > 0 else 0.0
    )
    p_d = (
        int(((y_pred == 0) & (y_true == 1)).sum()) / n_pred_del
        if n_pred_del > 0 else 0.0
    )
    cds_score = cds(p_s, n_pred_self, p_d, n_pred_del)
    accuracy = float((y_pred == y_true).mean())
    ssr_pred = float((y_pred == 1).mean())
    ssr_true = float((y_true == 1).mean())
    m_f1 = macro_f1(y_true, y_pred)

    print("\n============== CSA Evaluation Report ==============")
    for line in extra_lines:
        print(line)
    print(f"Base solve rate     : {base_accuracy:.3f}  (over ALL rows)")
    print(f"Total rows          : {total_rows}")
    print(f"  Invalid (pred=-1) : {n_invalid}  ({invalid_ratio * 100:.1f}%)")
    print(f"  Valid rows        : {n_valid}")
    print(f"  predict=1 count   : {n_pred_self}  (p_S={p_s:.3f})")
    print(f"  predict=0 count   : {n_pred_del}  (p_D={p_d:.3f})")
    print("---------------------------------------------------")
    print(f"1. CDS              : {cds_score:.3f}")
    print(f"2. Accuracy         : {accuracy:.3f}")
    print(f"3. Self-Solve Rate  : {ssr_pred:.3f}  (ground-truth SSR={ssr_true:.3f})")
    print(f"4. M-F1             : {m_f1:.3f}")
    print("===================================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CSA evaluation.")

    parser.add_argument("--csv_path", required=True,
                        help="CSV containing the CSA prediction column.")
    parser.add_argument("--prediction_col", required=True,
                        help="CSA prediction column (values in {-1, 0, 1}).")

    parser.add_argument("--grade_mode", choices=["column", "generations"],
                        required=True)

    # ``column`` mode
    parser.add_argument("--is_correct_col", default="is_correct",
                        help="Column with the ground-truth label (mode=column).")

    # ``generations`` mode
    parser.add_argument("--gt_csv_path", default=None,
                        help="CSV holding raw generations (mode=generations).")
    parser.add_argument("--generations_col", default=None,
                        help="Column with raw generations (mode=generations).")
    parser.add_argument("--domain", choices=["math", "science"], default=None,
                        help="Domain — sets the aggregation rule (mode=generations).")

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.prediction_col not in df.columns:
        raise ValueError(f"Missing prediction column: {args.prediction_col}")

    extra_lines = [
        f"Main CSV            : {args.csv_path}",
        f"Prediction column   : {args.prediction_col}",
    ]

    # ---- Compute is_correct ----
    if args.grade_mode == "column":
        if args.is_correct_col not in df.columns:
            raise ValueError(
                f"Missing is_correct column: {args.is_correct_col}. "
                "Switch to --grade_mode generations to grade on the fly."
            )
        df["is_correct"] = df[args.is_correct_col].astype(int)
        extra_lines.append(f"is_correct source   : column '{args.is_correct_col}'")
    else:
        if not (args.gt_csv_path and args.generations_col and args.domain):
            raise ValueError(
                "--grade_mode generations requires --gt_csv_path, "
                "--generations_col, and --domain."
            )
        gt_df = pd.read_csv(args.gt_csv_path)
        if len(gt_df) != len(df):
            raise ValueError(
                f"Row count mismatch: predictions={len(df)}, gt={len(gt_df)}. "
                "Alignment assumes identical row order."
            )
        is_correct = compute_is_correct(gt_df, args.generations_col, args.domain)
        df = df.reset_index(drop=True)
        is_correct = is_correct.reset_index(drop=True)
        df["is_correct"] = is_correct.astype(int)
        extra_lines += [
            f"is_correct source   : generations in {args.gt_csv_path}",
            f"Generations column  : {args.generations_col}",
            f"Domain (agg rule)   : {args.domain}",
        ]

    # ---- Filter invalid predictions ----
    total_rows = len(df)
    base_accuracy = float(df["is_correct"].mean()) if total_rows > 0 else 0.0
    invalid_mask = df[args.prediction_col] == -1
    n_invalid = int(invalid_mask.sum())
    df = df[~invalid_mask].reset_index(drop=True)

    report_metrics(df, args.prediction_col, total_rows, n_invalid, base_accuracy, extra_lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
