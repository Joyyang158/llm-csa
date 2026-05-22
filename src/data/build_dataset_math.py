"""Build the math dataset by merging GSM8K, MATH-500, and AIME.

Output schema (columns):
    question  — the problem statement
    answer    — the gold answer (raw form; final normalisation happens in
                grading.py at evaluation time)
    source    — one of {"GSM8K", "MATH500", "AIME"}

Splits:
    GSM8K    — 1000 train / 300 test (sampled from the official splits)
    MATH-500 — 400 train / 100 test (sampled from the test set)
    AIME     — train: di-zhang-fdu/AIME_1983_2024 with Year <= 2021
               test:  AI-MO/aimo-validation-aime

Note:
    The exact rows produced by this script may differ from the dataset used
    in our paper, because (1) the upstream HuggingFace datasets are updated
    from time to time, and (2) sampling order can shift slightly across
    library versions even with a fixed random seed.

    To reproduce our results exactly, please use the prepared dataset we
    release alongside this repo — it follows the same source mix and split
    ratios as documented above.

    If you only need a dataset with the same composition (not the exact
    same rows), running this script as-is is fine.
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


RANDOM_STATE = 42


def build_gsm8k(train_n: int, test_n: int) -> tuple:
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].to_pandas()[["question", "answer"]].dropna()
    test = ds["test"].to_pandas()[["question", "answer"]].dropna()
    train = train.sample(n=train_n, random_state=RANDOM_STATE).reset_index(drop=True)
    test = test.sample(n=test_n, random_state=RANDOM_STATE).reset_index(drop=True)
    train["source"] = "GSM8K"
    test["source"] = "GSM8K"
    return train, test


def build_math500(train_n: int, test_n: int) -> tuple:
    ds = load_dataset("HuggingFaceH4/MATH-500")
    df = ds["test"].to_pandas().rename(columns={"problem": "question"})
    df = df[["question", "answer"]]
    df["source"] = "MATH500"
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return (
        df.iloc[:train_n].reset_index(drop=True),
        df.iloc[train_n: train_n + test_n].reset_index(drop=True),
    )


def build_aime() -> tuple:
    train_ds = load_dataset("di-zhang-fdu/AIME_1983_2024")
    train = train_ds["train"].to_pandas().rename(
        columns={"Question": "question", "Answer": "answer"}
    )
    train = train[train["Year"] <= 2021][["question", "answer"]].dropna()
    train["source"] = "AIME"

    test_ds = load_dataset("AI-MO/aimo-validation-aime")
    test = test_ds["train"].to_pandas()[["problem", "solution"]].dropna().rename(
        columns={"problem": "question", "solution": "answer"}
    )
    test["source"] = "AIME"

    return train.reset_index(drop=True), test.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build the math dataset.")
    parser.add_argument("--output_dir", default="data/math",
                        help="Directory to write train.csv and test.csv into.")
    parser.add_argument("--gsm8k_train_n", type=int, default=1000)
    parser.add_argument("--gsm8k_test_n", type=int, default=300)
    parser.add_argument("--math500_train_n", type=int, default=400)
    parser.add_argument("--math500_test_n", type=int, default=100)
    args = parser.parse_args()

    print("Loading GSM8K ...")
    gsm_train, gsm_test = build_gsm8k(args.gsm8k_train_n, args.gsm8k_test_n)

    print("Loading MATH-500 ...")
    math_train, math_test = build_math500(args.math500_train_n, args.math500_test_n)

    print("Loading AIME ...")
    aime_train, aime_test = build_aime()

    df_train = pd.concat([gsm_train, math_train, aime_train], ignore_index=True)
    df_test = pd.concat([gsm_test, math_test, aime_test], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("\nTrain counts by source:")
    print(df_train["source"].value_counts().to_dict())
    print("\nTest counts by source:")
    print(df_test["source"].value_counts().to_dict())
    print(f"\nSaved to {train_path} and {test_path}")


if __name__ == "__main__":
    main()
