"""Build the SFT training CSV by selecting the correct analysis per row.

The teacher-binary analysis stage (see :mod:`src.data.generate_analysis` with
``--mode teacher_binary``) produces, for every question, two analyses — one
written as if the target model is correct (label=1) and one as if it is not
(label=0).

This script joins those two analyses against a per-row ``is_correct`` label
and emits a single ``analysis`` column matching the row's actual label. The
resulting CSV is the SFT training data for the CSA model.
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build SFT training data from binary analyses.")
    parser.add_argument("--model_csv", required=True,
                        help="CSV with per-row ``is_correct`` labels.")
    parser.add_argument("--analysis_csv", required=True,
                        help="CSV with per-question ``label_0_analysis`` and "
                             "``label_1_analysis`` columns.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--analysis_col", default="analysis",
                        help="Column name for the selected analysis.")
    args = parser.parse_args()

    df_model = pd.read_csv(args.model_csv)
    df_analysis = pd.read_csv(args.analysis_csv)

    required = {"label_0_analysis", "label_1_analysis", "question"}
    missing = required - set(df_analysis.columns)
    if missing:
        raise ValueError(f"analysis_csv missing columns: {sorted(missing)}")

    lookup = {}
    for _, row in df_analysis.iterrows():
        q = row["question"]
        if q not in lookup:
            lookup[q] = (row["label_0_analysis"], row["label_1_analysis"])

    selected = []
    missing_count = 0
    for _, row in df_model.iterrows():
        q = row["question"]
        if q not in lookup:
            selected.append(None)
            missing_count += 1
            continue
        label_0, label_1 = lookup[q]
        selected.append(label_1 if int(row["is_correct"]) == 1 else label_0)

    df_model[args.analysis_col] = selected
    if missing_count:
        print(f"[warn] {missing_count} questions had no analysis match.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_model.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df_model)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
