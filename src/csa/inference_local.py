"""Run a CSA model over a CSV of queries and record its decisions.

Two modes selected by ``--num_generations``:

  * 1   — single-shot: record one ``decision`` and one ``analysis`` per row.
  * N>1 — rollout: sample N times per row and record the per-class counts
          plus one example output per class. Used for collecting GRPO-style
          training rollouts.
"""

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.inference import generate_vllm, load_vllm
from src.utils.io import ensure_parent_dir, save_checkpoint
from src.utils.parsing import format_options, parse_decision
from src.utils.prompts import (
    CSA_DECISION_BINARY,
    CSA_DECISION_WITH_ANALYSIS,
)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def build_query(row: pd.Series, domain: str) -> str:
    """Format a row's question (and choices, for science) into a query string."""
    question = row["question"]
    if domain == "science":
        if "options" in row and pd.notna(row["options"]):
            choices = format_options(row["options"])
        else:
            choices = row["choices"]
        return f"Question: {question}\nChoices:\n{choices}"
    return f"Question: {question}"


def get_prompt_template(binary_only: bool) -> str:
    return CSA_DECISION_BINARY if binary_only else CSA_DECISION_WITH_ANALYSIS


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def detect_resume_index(
    save_path: str,
    sentinel_col: str,
) -> Tuple[pd.DataFrame, int]:
    """Return ``(df, start_idx)`` for resuming from an existing checkpoint."""
    if not os.path.exists(save_path):
        return None, 0
    df_ckpt = pd.read_csv(save_path)
    if sentinel_col not in df_ckpt.columns:
        return None, 0
    mask = df_ckpt[sentinel_col].isna()
    start_idx = int(mask.idxmax()) if mask.any() else len(df_ckpt)
    return df_ckpt, start_idx


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_single(df: pd.DataFrame, start_idx: int, model, tokenizer, args):
    """Single-shot mode: one decision and one analysis per row."""
    template = get_prompt_template(binary_only=args.binary_only)
    latencies = []

    for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df)):
        row = df.iloc[idx]
        prompt = template.format(query=build_query(row, args.domain))

        if idx == start_idx:
            logging.info("Example prompt:\n%s", prompt)

        outputs, dt = generate_vllm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_type=args.model_type,
            max_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            num_generations=1,
            return_time=True,
        )
        text = outputs[0]
        if idx >= args.warmup:
            latencies.append(dt)

        df.at[idx, args.decision_col] = parse_decision(text)
        df.at[idx, args.analysis_col] = text

        save_checkpoint(df, args.output_csv, idx, args.save_every)

    df.to_csv(args.output_csv, index=False)
    if latencies:
        logging.info("Avg latency: %.3fs", float(np.mean(latencies)))


def run_rollout(df: pd.DataFrame, start_idx: int, model, tokenizer, args):
    """Rollout mode: sample N times, record class counts and one example per class."""
    template = get_prompt_template(binary_only=args.binary_only)

    for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df)):
        row = df.iloc[idx]
        prompt = template.format(query=build_query(row, args.domain))

        if idx == start_idx:
            logging.info("Example prompt:\n%s", prompt)

        generations = generate_vllm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_type=args.model_type,
            max_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        c0 = c1 = cinv = 0
        ex0 = ex1 = None
        for text in generations:
            d = parse_decision(text)
            if d == 1:
                c1 += 1
                ex1 = ex1 or text
            elif d == 0:
                c0 += 1
                ex0 = ex0 or text
            else:
                cinv += 1

        df.at[idx, args.count0_col] = c0
        df.at[idx, args.count1_col] = c1
        df.at[idx, args.invalid_col] = cinv
        df.at[idx, args.example0_col] = ex0 or ""
        df.at[idx, args.example1_col] = ex1 or ""

        save_checkpoint(df, args.output_csv, idx, args.save_every)

    df.to_csv(args.output_csv, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CSA inference with vLLM.")

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--model_name", required=True,
                        help="HF id or local path of the CSA model model.")
    parser.add_argument("--model_type", default="qwen")
    parser.add_argument("--domain", choices=["math", "science"], required=True)
    parser.add_argument("--binary_only", action="store_true",
                        help="Use the decision-only prompt (no analysis block).")

    parser.add_argument("--num_generations", type=int, default=1,
                        help="1 for single-shot mode; >1 for rollout mode.")
    parser.add_argument("--max_new_tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--decision_col", default="decision")
    parser.add_argument("--analysis_col", default="analysis")

    parser.add_argument("--count0_col", default="count_0")
    parser.add_argument("--count1_col", default="count_1")
    parser.add_argument("--invalid_col", default="count_invalid")
    parser.add_argument("--example0_col", default="example_0")
    parser.add_argument("--example1_col", default="example_1")

    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    is_rollout = args.num_generations > 1
    sentinel_col = args.count0_col if is_rollout else args.decision_col

    df_ckpt, start_idx = detect_resume_index(args.output_csv, sentinel_col)
    if df_ckpt is not None and start_idx > 0:
        df = df_ckpt
        logging.info(
            "[Resume] Found checkpoint, continuing from row %d / %d",
            start_idx, len(df),
        )
    else:
        df = pd.read_csv(args.input_csv)
        start_idx = 0

    if start_idx >= len(df):
        logging.info("[Resume] All rows already processed.")
        return

    ensure_parent_dir(args.output_csv)
    tokenizer, model = load_vllm(args.model_name)

    if is_rollout:
        run_rollout(df, start_idx, model, tokenizer, args)
    else:
        run_single(df, start_idx, model, tokenizer, args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
