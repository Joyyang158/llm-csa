"""Shared dataset validation and science question formatting."""

import pandas as pd

from src.utils.parsing import format_options


def science_choices(example) -> str:
    """Accept raw benchmark options or an already formatted choices column."""
    options = example.get("options")
    if isinstance(options, (list, tuple)) or (
        isinstance(options, str) and options.strip()
    ):
        return format_options(options)
    choices = example.get("choices")
    if isinstance(choices, str) and choices.strip():
        return choices
    raise ValueError("Science data requires nonempty 'options' or 'choices'.")


def validate_binary_labels(values, name="is_correct") -> None:
    if not pd.Series(values).isin([0, 1]).all():
        raise ValueError(f"{name} must contain only 0 or 1, without missing values.")


def validate_query_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Fail rather than silently pair predictions with different queries."""
    if left.empty or right.empty:
        raise ValueError("Evaluation data must not be empty.")
    if len(left) != len(right):
        raise ValueError(f"Row count mismatch: {len(left)} vs {len(right)}.")
    if "question" not in left or "question" not in right:
        raise ValueError("Both CSVs need a 'question' column to verify query alignment.")
    for col in ("question", "question_id", "source", "category", "options",
                "choices", "answer", "answer_index"):
        if col not in left or col not in right:
            continue
        a = left[col].reset_index(drop=True)
        b = right[col].reset_index(drop=True)
        same = a.eq(b) | (a.isna() & b.isna())
        if not same.all():
            raise ValueError(
                f"Query alignment mismatch in '{col}'; use the same queries "
                "in the same order in both CSVs."
            )
