"""IO helpers shared across data-generation and inference scripts.

Centralises checkpoint/resume logic that was previously duplicated in
multiple files.
"""

import logging
import os
from typing import Tuple

import pandas as pd


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not yet exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_with_resume(
    input_csv: str,
    output_csv: str,
    sentinel_col: str,
) -> Tuple[pd.DataFrame, int]:
    """Return ``(df, start_idx)`` for resuming a long-running job.

    If ``output_csv`` exists and contains ``sentinel_col``, resume from the
    first row whose sentinel cell is NaN. Otherwise, load ``input_csv`` fresh
    and start from row 0.
    """
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        if sentinel_col in df.columns:
            mask = df[sentinel_col].isna()
            start_idx = int(mask.idxmax()) if mask.any() else len(df)
            logging.info(
                "[Resume] Found checkpoint at %s, continuing from row %d / %d",
                output_csv, start_idx, len(df),
            )
            return df, start_idx

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    return pd.read_csv(input_csv), 0


def save_checkpoint(
    df: pd.DataFrame,
    output_csv: str,
    current_row: int,
    save_every: int,
) -> None:
    """Save ``df`` to ``output_csv`` every ``save_every`` rows."""
    if save_every > 0 and (current_row + 1) % save_every == 0:
        df.to_csv(output_csv, index=False)
        logging.info("[Checkpoint] saved row %d to %s", current_row + 1, output_csv)


def ensure_object_column(df: pd.DataFrame, col: str) -> None:
    """Ensure a column exists with ``object`` dtype (to hold list/dict cells)."""
    if col not in df.columns:
        df[col] = pd.NA
    df[col] = df[col].astype(object)
