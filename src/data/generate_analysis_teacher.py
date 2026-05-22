"""Generate routing analysis via a teacher model on the Together API.

For each row, writes *two* analyses — one for label=0 (DELEGATE) and one
for label=1 (SELF_SOLVE) — into the columns ``label_0_analysis`` and
``label_1_analysis``. The row's own ``is_correct`` value is not used here;
both labels are explained, and the SFT data builder later picks the matching
analysis for each row.
"""

import argparse
import logging
import os

import pandas as pd
from func_timeout import FunctionTimedOut, func_set_timeout
from tqdm import tqdm

from src.utils.io import ensure_parent_dir, save_checkpoint
from src.utils.parsing import format_options
from src.utils.prompts import ANALYSIS_TEACHER, label_to_decision


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_query(row: pd.Series) -> str:
    """Format the row's question (with options for MCQ) into a query string."""
    question = row["question"]
    if "options" in row and pd.notna(row["options"]):
        choices = format_options(row["options"])
        return f"Question: {question}\nChoices:\n{choices}"
    return f"Question: {question}"


def build_prompt(row: pd.Series, label: int) -> str:
    decision = label_to_decision(label)
    return ANALYSIS_TEACHER.format(query=build_query(row), decision=decision)


# ---------------------------------------------------------------------------
# Teacher API call
# ---------------------------------------------------------------------------

@func_set_timeout(1200)
def call_teacher(client, model_name: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def safe_call_teacher(client, model_name: str, prompt: str, max_tokens: int) -> str:
    try:
        return call_teacher(client, model_name, prompt, max_tokens)
    except FunctionTimedOut:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    from together import Together

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    client = Together(api_key=api_key)

    ensure_parent_dir(args.output_csv)
    df = pd.read_csv(args.input_csv)

    label_cols = {0: "label_0_analysis", 1: "label_1_analysis"}
    for col in label_cols.values():
        if col not in df.columns:
            df[col] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="teacher"):
        for label, col in label_cols.items():
            existing = row[col]
            if pd.notna(existing) and not str(existing).startswith("Error"):
                continue

            prompt = build_prompt(row, label)
            if idx == 0 and label == 0:
                logging.info("Sample prompt:\n%s", prompt)

            df.at[idx, col] = safe_call_teacher(
                client, args.teacher_model, prompt, args.max_tokens,
            )

        save_checkpoint(df, args.output_csv, idx, args.save_every)

    df.to_csv(args.output_csv, index=False)
    logging.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate teacher-written routing analysis (Together API)."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--teacher_model", required=True,
                        help="Together-hosted teacher model id.")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(parse_args())