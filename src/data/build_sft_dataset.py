"""Assemble label-conditioned self or teacher analyses for SFT."""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.data import validate_binary_labels


def build_sft_frame(model, analyses, mode="teacher", source_col="routing_analysis",
                    output_col="SFT_analysis"):
    if mode not in {"self", "teacher"}:
        raise ValueError("mode must be self or teacher")
    for frame in (model, analyses):
        if "question" not in frame or frame["question"].isna().any():
            raise ValueError("Both inputs need nonempty question values.")
    if "is_correct" not in model:
        raise ValueError("model_csv must contain is_correct labels.")
    validate_binary_labels(model["is_correct"])
    needed = [source_col, "is_correct"] if mode == "self" else ["label_0_analysis", "label_1_analysis"]
    missing = set(needed) - set(analyses.columns)
    if missing:
        raise ValueError(f"analysis_csv missing columns: {sorted(missing)}")
    # Include shared query identifiers/options to avoid joining distinct MCQs.
    keys = [c for c in ("question", "question_id", "source", "category", "options", "choices")
            if c in model and c in analyses]
    if analyses.duplicated(keys).any():
        raise ValueError("Analysis queries are not unique; cannot select an unambiguous match.")
    lookup = analyses.set_index(keys)
    query_index = pd.MultiIndex.from_frame(model[keys]) if len(keys) > 1 else pd.Index(model[keys[0]])
    if not query_index.isin(lookup.index).all():
        raise ValueError("Some model queries have no matching analysis.")
    matched = lookup.reindex(query_index).reset_index(drop=True)
    labels = model["is_correct"].reset_index(drop=True)
    if mode == "self":
        validate_binary_labels(matched["is_correct"])
        if not matched["is_correct"].eq(labels).all():
            raise ValueError("Self-analysis labels differ from the target model labels.")
        selected = matched[source_col]
    else:
        selected = matched["label_1_analysis"].where(labels.eq(1), matched["label_0_analysis"])
    if not selected.map(lambda x: isinstance(x, str) and bool(x.strip())
                        and not x.lstrip().startswith("Error")).all():
        raise ValueError("Selected analyses contain missing, empty, or failed generations.")
    result = model.copy()
    result[output_col] = selected.to_numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_csv", required=True)
    parser.add_argument("--analysis_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--mode", choices=["self", "teacher"], default="teacher")
    parser.add_argument("--source_analysis_col", default="routing_analysis")
    parser.add_argument("--analysis_col", default="SFT_analysis")
    args = parser.parse_args()
    result = build_sft_frame(pd.read_csv(args.model_csv), pd.read_csv(args.analysis_csv),
                             args.mode, args.source_analysis_col, args.analysis_col)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Saved {len(result)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
