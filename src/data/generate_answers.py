"""Generate model answers on a benchmark CSV (vLLM only).

Two domains are supported:

  * ``math``    — open-ended; produces a list of ``num_generations`` raw
                  outputs per row.
  * ``science`` — multiple-choice; for each row, the option list is shuffled
                  ``num_shuffles`` times and the model is asked once per
                  shuffle. The shuffled order and re-mapped correct letter
                  are recorded alongside each generation.
"""

import argparse
import json
import logging
import random
from typing import List

import pandas as pd
from tqdm import tqdm

from src.utils.inference import generate_vllm, load_vllm
from src.utils.io import ensure_object_column, ensure_parent_dir, save_checkpoint
from src.utils.parsing import OPTION_LABELS, parse_options
from src.utils.prompts import get_benchmark_prompt


# ---------------------------------------------------------------------------
# Per-row generation
# ---------------------------------------------------------------------------

def _format_choices(options: List[str]) -> str:
    return "\n".join(f"({OPTION_LABELS[i]}) {opt}" for i, opt in enumerate(options))


def _run(prompt: str, model, tokenizer, args, num_generations: int) -> List[str]:
    return generate_vllm(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        model_type=args.model_type,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        num_generations=num_generations,
    )


def generate_math_row(row: pd.Series, model, tokenizer, args) -> List[str]:
    """Math: one prompt, ``num_generations`` raw outputs."""
    prompt = get_benchmark_prompt("math").format(question=row["question"])
    return _run(prompt, model, tokenizer, args, num_generations=args.num_generations)


def generate_science_row(row: pd.Series, model, tokenizer, args) -> List[dict]:
    """Science: shuffle options ``num_shuffles`` times, sample once per shuffle.

    Each returned dict has:
      * ``shuffled_order``: list[int] mapping new positions to original indices
      * ``correct_label``:  the letter the gold answer landed on
      * ``generation``:     the model's raw output for this shuffle
    """
    options = parse_options(str(row["options"]))
    answer_idx = int(row["answer_index"])

    results = []
    for _ in range(args.num_shuffles):
        order = list(range(len(options)))
        random.shuffle(order)
        shuffled = [options[j] for j in order]
        new_answer_letter = OPTION_LABELS[order.index(answer_idx)]

        prompt = get_benchmark_prompt("science").format(
            question=row["question"],
            choices=_format_choices(shuffled),
        )
        outputs = _run(prompt, model, tokenizer, args, num_generations=1)

        results.append({
            "shuffled_order": order,
            "correct_label": new_answer_letter,
            "generation": outputs[0],
        })
    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    logging.info("Domain        : %s", args.domain)
    logging.info("Model         : %s", args.model_name)
    logging.info("Thinking mode : %s", args.enable_thinking)
    logging.info("Output column : %s", args.output_col)

    ensure_parent_dir(args.output_csv)
    df = pd.read_csv(args.input_csv)
    ensure_object_column(df, args.output_col)

    tokenizer, model = load_vllm(args.model_name)

    for i in tqdm(range(len(df)), desc="Generating"):
        if pd.notna(df.at[i, args.output_col]):
            continue

        if args.domain == "math":
            value = generate_math_row(df.iloc[i], model, tokenizer, args)
            df.at[i, args.output_col] = value
        else:  # science
            value = generate_science_row(df.iloc[i], model, tokenizer, args)
            df.at[i, args.output_col] = json.dumps(value, ensure_ascii=False)

        if i == 0:
            logging.info("=== Sample output ===\n%s", value)

        save_checkpoint(df, args.output_csv, i, args.save_every)

    df.to_csv(args.output_csv, index=False)
    logging.info("Done: processed %d rows.", len(df))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model answers on a benchmark CSV (vLLM)."
    )

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_col", required=True,
                        help="Column to store generations.")

    parser.add_argument("--domain", choices=["math", "science"], required=True)

    parser.add_argument("--model_name", required=True,
                        help="HuggingFace ID or local path of the model.")
    parser.add_argument("--model_type", default="qwen",
                        help="Model family for chat template selection.")
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Generations per row (math domain).")
    parser.add_argument("--num_shuffles", type=int, default=5,
                        help="Shuffled attempts per row (science domain).")

    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(parse_args())
