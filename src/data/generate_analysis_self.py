"""Generate routing analysis via a local model with vLLM.

For each row, the target model produces a single first-person rationale
conditioned on the row's existing ``is_correct`` label. The output is
written to ``--output_col`` (default ``routing_analysis``).
"""

import argparse
import logging

import pandas as pd
from tqdm import tqdm

from src.utils.inference import generate_vllm, load_vllm
from src.utils.io import ensure_parent_dir, save_checkpoint
from src.utils.parsing import format_options
from src.utils.prompts import ANALYSIS_SELF, label_to_decision


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


def build_prompt(row: pd.Series) -> str:
    decision = label_to_decision(row["is_correct"])
    return ANALYSIS_SELF.format(query=build_query(row), decision=decision)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    ensure_parent_dir(args.output_csv)
    df = pd.read_csv(args.input_csv)
    if args.output_col not in df.columns:
        df[args.output_col] = None

    tokenizer, model = load_vllm(args.model_name)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="self"):
        existing = row[args.output_col]
        if pd.notna(existing) and not str(existing).startswith("Error"):
            continue

        prompt = build_prompt(row)
        if idx == 0:
            logging.info("Sample prompt:\n%s", prompt)

        outputs = generate_vllm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_type=args.model_type,
            max_tokens=args.max_tokens,
            num_generations=1,
        )
        df.at[idx, args.output_col] = outputs[0]
        save_checkpoint(df, args.output_csv, idx, args.save_every)

    df.to_csv(args.output_csv, index=False)
    logging.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate self-written routing analysis (local vLLM)."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", required=True,
                        help="HF id or local path of the self-analysis model.")
    parser.add_argument("--model_type", default="qwen")
    parser.add_argument("--output_col", default="routing_analysis")
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(parse_args())