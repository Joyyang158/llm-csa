"""Parsing utilities for model outputs.

Consolidates logic that was previously duplicated across CSA-related and
evaluation scripts (decision parsing, boxed-answer extraction, option
formatting, robust literal evaluation of CSV-stored Python/JSON values).
"""

import ast
import json
import re
from typing import Any, List, Optional, Sequence, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Routing decision parsing
# ---------------------------------------------------------------------------

DECISION_MAP = {"SELF_SOLVE": 1, "DELEGATE": 0}
INVALID_DECISION = -1


def parse_decision(text: str) -> int:
    """Extract a routing decision from a model output.

    Returns 1 for SELF_SOLVE, 0 for DELEGATE, and -1 if no valid decision is
    found. Prefers tagged decisions (``<decision>SELF_SOLVE</decision>``) over
    bare keywords, and uses the *last* match in either case so trailing
    decisions override earlier reasoning.
    """
    tagged = re.findall(r"<decision>\s*(SELF_SOLVE|DELEGATE)", text, re.IGNORECASE)
    if tagged:
        return DECISION_MAP.get(tagged[-1].upper(), INVALID_DECISION)

    fallback = re.findall(r"\b(SELF_SOLVE|DELEGATE)\b", text, re.IGNORECASE)
    if fallback:
        return DECISION_MAP.get(fallback[-1].upper(), INVALID_DECISION)

    return INVALID_DECISION


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: Union[str, float, None]) -> Optional[str]:
    """Extract the last ``\\boxed{...}`` content from a model output."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).split("</think>")[-1]
    matches = re.findall(r"\\boxed\{(.*?)}", s)
    return matches[-1] if matches else None


def extract_letter_answer(text: Union[str, float, None]) -> Optional[str]:
    """Extract a single uppercase letter answer (A, B, C, ...) from output.

    Looks inside the last ``\\boxed{...}`` first; if no boxed content is found,
    falls back to the first standalone uppercase letter in the trailing
    (post-``</think>``) portion of the text.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{(.*?)}", s)
    target = boxed[-1] if boxed else s
    match = re.search(r"\b([A-Z])\b", target)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Multiple-choice option formatting
# ---------------------------------------------------------------------------

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWX"  # supports up to 24 options


def format_options(options: Union[str, Sequence[str]]) -> str:
    """Format a list of option strings as labelled multiple-choice text."""
    if isinstance(options, str):
        options = ast.literal_eval(options)
    return "\n".join(
        f"({OPTION_LABELS[i]}) {opt}" for i, opt in enumerate(options)
    )


def parse_options(raw: str) -> List[str]:
    """Parse a stringified Python list of option strings."""
    return ast.literal_eval(raw)


# ---------------------------------------------------------------------------
# Robust literal evaluation for CSV-stored values
# ---------------------------------------------------------------------------

_BACKSLASH_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _escape_lone_backslashes(text: str) -> str:
    r"""Double any backslash that isn't already part of a valid JSON escape.

    JSON only recognises ``\" \\ \/ \b \f \n \r \t \uXXXX``. When a CSV cell
    contains LaTeX-style content like ``\boxed{...}`` or ``\frac{a}{b}``, naive
    ``json.loads`` will choke on the lone backslashes. This rescues such
    strings by escaping them.
    """
    return _BACKSLASH_ESCAPE_RE.sub(r"\\\\", text)


def safe_literal_eval(value: Any, row_idx: Optional[int] = None) -> Any:
    """Robustly parse a CSV cell that should hold a Python/JSON literal.

    Tries, in order:
      1. ``json.loads`` on the raw string
      2. ``json.loads`` after escaping lone backslashes
      3. ``ast.literal_eval`` (handles single-quoted dicts, ``True``/``None``)
      4. ``json.loads`` after Python-repr to JSON normalisation

    Returns ``None`` for empty / NaN cells, and emits a warning (without
    raising) when nothing parses.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None

    last_err = None

    try:
        return json.loads(s)
    except Exception as e:
        last_err = ("json", e)

    try:
        return json.loads(_escape_lone_backslashes(s))
    except Exception as e:
        last_err = ("json-escaped", e)

    try:
        return ast.literal_eval(s)
    except Exception as e:
        last_err = ("ast", e)

    try:
        normalised = (
            s.replace("None", "null")
             .replace("True", "true")
             .replace("False", "false")
             .replace("'", '"')
        )
        return json.loads(_escape_lone_backslashes(normalised))
    except Exception as e:
        last_err = ("json-after-normalise", e)

    where = f" (row {row_idx})" if row_idx is not None else ""
    kind, err = last_err
    print(
        f"[warn] Could not parse cell{where} (last error from {kind}: "
        f"{type(err).__name__}: {err}). Cell len={len(s)}, "
        f"first 300 chars: {s[:300]!r}"
    )
    return None


def parse_list_like(value: Any, row_idx: Optional[int] = None) -> list:
    """Parse a value that should be a list; wrap scalars in a single-element list."""
    parsed = safe_literal_eval(value, row_idx=row_idx)
    if parsed is None:
        return []
    return parsed if isinstance(parsed, (list, tuple)) else [parsed]
