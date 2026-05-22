"""Correctness grading for math and science benchmark outputs.

Both grading flows take a list of model generations per row and reduce them to
a single binary ``is_correct`` label. The aggregation rule differs by domain:

  * math    — *any-correct* over N generations (pass@1-style).
  * science — *majority vote* over N attempts (mirrors the shuffled MCQ flow,
              where each attempt sees a different option ordering).
"""

import re
from typing import Optional

import pandas as pd

from src.utils.parsing import (
    extract_boxed_answer,
    extract_letter_answer,
    parse_list_like,
    safe_literal_eval,
)


# ---------------------------------------------------------------------------
# Math: gold-answer normalisation
# ---------------------------------------------------------------------------

def _normalise_gsm8k_answer(text) -> str:
    """GSM8K gold answers are formatted as ``... #### <answer>``; extract the tail."""
    s = "" if pd.isna(text) else str(text)
    match = re.search(r"####\s*(.+)", s)
    return match.group(1).strip().replace(",", "") if match else s


def _normalise_math500_answer(expr) -> str:
    """Strip ``\\left( ... \\right)`` and whitespace from MATH500 gold expressions."""
    s = "" if pd.isna(expr) else str(expr)
    s = re.sub(r"\\left\s*\(", "(", s)
    s = re.sub(r"\\right\s*\)", ")", s)
    s = re.sub(r"\\left\s*\[", "[", s)
    s = re.sub(r"\\right\s*\]", "]", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s


def normalise_math_gold_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``answer`` column based on the per-row ``source`` value."""
    if "source" not in df.columns:
        return df

    df = df.copy()
    df["answer"] = df["answer"].astype(str)
    src_lower = df["source"].astype(str).str.lower()

    mask_math = src_lower.str.contains("math500", na=False)
    if mask_math.any():
        df.loc[mask_math, "answer"] = df.loc[mask_math, "answer"].apply(
            _normalise_math500_answer
        )

    mask_gsm = src_lower.str.contains("gsm8k", na=False)
    if mask_gsm.any():
        df.loc[mask_gsm, "answer"] = df.loc[mask_gsm, "answer"].apply(
            _normalise_gsm8k_answer
        )

    return df


# ---------------------------------------------------------------------------
# Per-attempt correctness checks
# ---------------------------------------------------------------------------

def is_math_correct(prediction: str, gold: str) -> int:
    """1 if the prediction's last ``\\boxed{}`` matches the gold string."""
    pred = extract_boxed_answer(prediction)
    if pred is None:
        return 0
    return int(pred.strip() == gold.strip())


def is_science_correct(prediction: str, gold_letter: str) -> int:
    """1 if the prediction's extracted letter answer matches the gold letter."""
    pred = extract_letter_answer(prediction)
    if pred is None:
        return 0
    return int(pred == gold_letter.strip())


# ---------------------------------------------------------------------------
# Single-mode grading helpers (one generation per row)
# ---------------------------------------------------------------------------

def grade_math_single(prediction, gold: str) -> int:
    """Grade a single math generation (not a list)."""
    return is_math_correct(str(prediction), gold)


def grade_science_single(attempt_value, row_idx: Optional[int] = None) -> int:
    """Grade a single science attempt (a dict, or a list with one dict)."""
    attempt = safe_literal_eval(attempt_value, row_idx=row_idx)
    if isinstance(attempt, list):
        attempt = attempt[0] if attempt else None
    if not isinstance(attempt, dict):
        return 0
    gold = str(attempt.get("correct_label", "")).strip()
    gen = attempt.get("generation", "")
    if isinstance(gen, list):
        gen = gen[0] if gen else ""
    return is_science_correct(str(gen), gold)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _per_source_table(df: pd.DataFrame, value_col: str, label: str) -> None:
    """Print a per-``source`` accuracy breakdown."""
    if "source" not in df.columns:
        return
    grouped = df.groupby("source")[value_col].agg(["sum", "count", "mean"]).sort_index()
    src_width = max(len("source"), grouped.index.astype(str).str.len().max())
    print(f"\n--- Per-source {label} ---")
    print(f"  {'source'.ljust(src_width)}  {'correct':>10}  {'total':>6}  {'acc':>6}")
    print(f"  {'-' * src_width}  {'-' * 10}  {'-' * 6}  {'-' * 6}")
    for src, row in grouped.iterrows():
        print(
            f"  {str(src).ljust(src_width)}  {int(row['sum']):>10}  "
            f"{int(row['count']):>6}  {row['mean']:>6.3f}"
        )


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def compute_is_correct(
    df: pd.DataFrame,
    generations_col: str,
    domain: str,
) -> pd.Series:
    """Compute a binary ``is_correct`` series for a dataframe of generations.

    Used by the CSA evaluation script to grade raw generations on the
    fly when no precomputed ``is_correct`` column is available.
    """
    if generations_col not in df.columns:
        raise ValueError(f"Missing column: {generations_col}")

    if domain == "math":
        df = normalise_math_gold_answers(df)
        gold = df["answer"].astype(str).str.strip()
        return pd.Series(
            [
                int(any(is_math_correct(str(p), g) for p in parse_list_like(v)))
                for v, g in zip(df[generations_col], gold)
            ],
            index=df.index,
            name="is_correct",
        )

    if domain == "science":
        results = []
        for i, value in enumerate(df[generations_col]):
            attempts = safe_literal_eval(value, row_idx=i)
            if attempts is None:
                results.append(0)
                continue
            if not isinstance(attempts, list):
                attempts = [attempts]
            row_scores = []
            for att in attempts:
                if not isinstance(att, dict):
                    row_scores.append(0)
                    continue
                gold_letter = str(att.get("correct_label", "")).strip()
                gen = att.get("generation", "")
                if isinstance(gen, list):
                    gen = gen[0] if gen else ""
                row_scores.append(is_science_correct(str(gen), gold_letter))
            threshold = len(row_scores) // 2 + 1
            results.append(int(sum(row_scores) >= threshold))
        return pd.Series(results, index=df.index, name="is_correct")

    raise ValueError(f"Unknown domain: {domain!r}. Expected 'math' or 'science'.")


def grade_dataframe(
    df: pd.DataFrame,
    generations_col: str,
    domain: str,
    mode: str = "multi",
) -> pd.DataFrame:
    """Grade a dataframe and print per-attempt / per-source statistics.

    Modes:
      * ``single`` — one generation per row.
      * ``multi``  — a list of generations (math) or a list of dict attempts
                     (science) per row. Reports per-attempt accuracy and an
                     aggregate row label (any-correct for math, majority-vote
                     for science).

    Returns the dataframe with an added ``is_correct`` column.
    """
    df = df.copy()
    if generations_col not in df.columns:
        raise ValueError(f"Missing column: {generations_col}")

    print("=" * 50)
    print(f"  Eval column : {generations_col}")
    print(f"  Domain      : {domain}")
    print(f"  Mode        : {mode}")
    print(f"  Num rows    : {len(df)}")
    print("=" * 50)

    if domain == "math":
        df = normalise_math_gold_answers(df)
        gold = df["answer"].astype(str).str.strip()
    else:
        gold = None  # science gold is embedded in each attempt dict

    # ---- Single mode ----
    if mode == "single":
        if domain == "math":
            is_correct = [grade_math_single(p, g) for p, g in zip(df[generations_col], gold)]
        else:
            is_correct = [
                grade_science_single(v, i) for i, v in enumerate(df[generations_col])
            ]
        df["is_correct"] = pd.Series(is_correct, index=df.index).astype(int)
        print(f"Accuracy: {df['is_correct'].mean():.4f}")
        _per_source_table(df, "is_correct", "accuracy")
        return df

    # ---- Multi mode ----
    if mode != "multi":
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'single' or 'multi'.")

    if domain == "math":
        per_attempt = []
        for preds_value, g in zip(df[generations_col], gold):
            preds = parse_list_like(preds_value)
            per_attempt.append([is_math_correct(str(p), g) for p in preds])
    else:
        per_attempt = []
        for i, value in enumerate(df[generations_col]):
            attempts = safe_literal_eval(value, row_idx=i)
            if attempts is None:
                per_attempt.append([])
                continue
            if not isinstance(attempts, list):
                attempts = [attempts]
            row_scores = []
            for att in attempts:
                if not isinstance(att, dict):
                    row_scores.append(0)
                    continue
                gold_letter = str(att.get("correct_label", "")).strip()
                gen = att.get("generation", "")
                if isinstance(gen, list):
                    gen = gen[0] if gen else ""
                row_scores.append(is_science_correct(str(gen), gold_letter))
            per_attempt.append(row_scores)

    # Per-attempt accuracy
    n_attempts = [len(row) for row in per_attempt]
    if n_attempts:
        max_n = max(n_attempts)
        print(f"  Attempts per row: min={min(n_attempts)}, max={max_n}")
        print("\n--- Per-attempt accuracy ---")
        for i in range(max_n):
            attempt_correct = [row[i] for row in per_attempt if i < len(row)]
            count = len(attempt_correct)
            acc_i = sum(attempt_correct) / count if count > 0 else 0.0
            print(f"  Attempt {i + 1}: {acc_i:.3f} ({sum(attempt_correct)}/{count})")
        all_scores = [s for row in per_attempt for s in row]
        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"  Average    : {avg:.3f}")

    # Aggregation rule depends on domain
    if domain == "math":
        is_correct = [int(any(row)) for row in per_attempt]
        df["is_correct"] = pd.Series(is_correct, index=df.index).astype(int)
        print(f"\n--- Pass@1 (any correct) ---")
        print(f"  Pass@1: {df['is_correct'].mean():.3f}")
        _per_source_table(df, "is_correct", "pass@1")
    else:
        is_correct = [
            int(sum(row) >= (len(row) // 2 + 1)) for row in per_attempt
        ]
        df["is_correct"] = pd.Series(is_correct, index=df.index).astype(int)
        print(f"\n--- Majority vote ---")
        print(f"  Accuracy: {df['is_correct'].mean():.3f}")
        _per_source_table(df, "is_correct", "majority-vote accuracy")

    return df
