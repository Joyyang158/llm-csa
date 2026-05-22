"""Build the science dataset by filtering MMLU-Pro to science categories.

Source: ``TIGER-Lab/MMLU-Pro`` (test split). Filtered to four categories
(biology, chemistry, health, physics) and stratified-split into a train and
a test set.

Output schema (columns):
    question      — the problem statement
    options       — JSON-encoded list of option strings
    answer_index  — int index of the correct option in ``options``
    category      — one of {"biology", "chemistry", "health", "physics"}

Note:
    The exact rows produced by this script may differ from the dataset used
    in our paper, because (1) the upstream HuggingFace dataset is updated
    from time to time, and (2) the stratified split can shift slightly
    across library versions even with a fixed random seed.

    To reproduce our results exactly, please use the prepared dataset we
    release alongside this repo — it follows the same category filter and
    split ratios as documented above.

    If you only need a dataset with the same composition (not the exact
    same rows), running this script as-is is fine.
"""

import argparse
import json
import os

from datasets import load_dataset
from sklearn.model_selection import train_test_split


TARGET_CATEGORIES = ["biology", "chemistry", "health", "physics"]
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser(
        description="Build the science dataset from MMLU-Pro."
    )
    parser.add_argument("--output_dir", default="data/science",
                        help="Directory to write train.csv and test.csv into.")
    parser.add_argument("--test_size", type=int, default=700,
                        help="Number of rows to put in the test split.")
    args = parser.parse_args()

    print("Loading MMLU-Pro ...")
    df = load_dataset("TIGER-Lab/MMLU-Pro", split="test").to_pandas()
    df = df[df["category"].isin(TARGET_CATEGORIES)].reset_index(drop=True)

    df = df.drop(columns=["cot_content"], errors="ignore")

    print(f"Filtered to {len(df)} science rows. Category distribution:")
    print(df["category"].value_counts())

    df["options"] = df["options"].apply(lambda x: json.dumps(list(x)))

    test_frac = args.test_size / len(df)
    train_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    print("\nTrain category proportions:")
    print(train_df["category"].value_counts(normalize=True).round(4))
    print("\nTest category proportions:")
    print(test_df["category"].value_counts(normalize=True).round(4))

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved to {train_path} and {test_path}")


if __name__ == "__main__":
    main()
