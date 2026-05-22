"""Diversity-Filtered Warm-up (DFW) — paper §3.2(d), Algorithm 1, Stage 1.

Constructs the diversified subset D_div ⊆ D used for the warm-up phase of
two-stage GRPO training. For each training query, this script samples K
rollouts from the initial policy π_θ_0 and retains only those queries whose
K rollouts include *both* SELF_SOLVE and DELEGATE decisions. The resulting
subset is exactly the data on which the GRPO advantage is informative on the
very first update — base models predict SELF_SOLVE on nearly every query
(paper §2.2), so without this filter within-group reward variance vanishes
and policy gradient carries no learning signal.

Outputs a CSV with the same columns as the input training set, plus three
diagnostic columns:

  * ``dfw_n_self``    — number of SELF_SOLVE rollouts (out of K)
  * ``dfw_n_delegate`` — number of DELEGATE rollouts (out of K)
  * ``dfw_n_invalid`` — number of unparseable rollouts (out of K)

Two findings from the paper (Appendix E.1) about DFW subset shape:
  1. Stronger models yield smaller D_div (more internally consistent).
  2. The retained queries skew toward DELEGATE (57–70% in the paper),
     concentrating training signal on capability-boundary cases.

Both are reported automatically when this script finishes.
"""

import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils.inference import generate_vllm, load_vllm
from src.utils.io import ensure_parent_dir
from src.utils.parsing import format_options, parse_decision
from src.utils.prompts import CSA_DECISION_WITH_ANALYSIS


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def build_query(row: pd.Series, domain: str) -> str:
    question = row["question"]
    if domain == "science":
        if "options" in row and pd.notna(row["options"]):
            choices = format_options(row["options"])
        else:
            choices = row["choices"]
        return f"Question: {question}\nChoices:\n{choices}"
    return f"Question: {question}"


# ---------------------------------------------------------------------------
# Main DFW loop
# ---------------------------------------------------------------------------

def run(args):
    ensure_parent_dir(args.output_csv)
    df = pd.read_csv(args.input_csv)
    logging.info("Loaded %d training rows from %s", len(df), args.input_csv)
    logging.info("Sampling K=%d rollouts per query from %s",
                 args.num_rollouts, args.model_name)

    tokenizer, model = load_vllm(args.model_name)

    n_self_list = []
    n_del_list = []
    n_inv_list = []

    for idx in tqdm(range(len(df)), desc="DFW rollouts"):
        row = df.iloc[idx]
        prompt = CSA_DECISION_WITH_ANALYSIS.format(
            query=build_query(row, args.domain)
        )
        if idx == 0:
            logging.info("Example prompt:\n%s", prompt)

        generations = generate_vllm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_type=args.model_type,
            max_tokens=args.max_new_tokens,
            num_generations=args.num_rollouts,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        n_self = n_del = n_inv = 0
        for text in generations:
            d = parse_decision(text)
            if d == 1:
                n_self += 1
            elif d == 0:
                n_del += 1
            else:
                n_inv += 1
        n_self_list.append(n_self)
        n_del_list.append(n_del)
        n_inv_list.append(n_inv)

    df["dfw_n_self"] = n_self_list
    df["dfw_n_delegate"] = n_del_list
    df["dfw_n_invalid"] = n_inv_list

    # Diversification filter: keep rows where BOTH labels appeared.
    keep_mask = (df["dfw_n_self"] > 0) & (df["dfw_n_delegate"] > 0)
    diversified = df[keep_mask].reset_index(drop=True)

    diversified.to_csv(args.output_csv, index=False)

    # Reporting: paper §3.2(d) and Appendix E.1 numbers.
    n_total = len(df)
    n_div = len(diversified)
    print("\n================ DFW Subset Report ================")
    print(f"Model               : {args.model_name}")
    print(f"Domain              : {args.domain}")
    print(f"Rollouts per query  : {args.num_rollouts}")
    print(f"Total queries       : {n_total}")
    print(f"Diversified subset  : {n_div}  ({n_div / n_total * 100:.1f}%)")
    if n_div > 0:
        # Aggregate label counts WITHIN the diversified subset (Finding 2).
        sum_self = int(diversified["dfw_n_self"].sum())
        sum_del = int(diversified["dfw_n_delegate"].sum())
        denom = sum_self + sum_del
        if denom > 0:
            pct_del = sum_del / denom * 100.0
            print(f"Subset DELEGATE rate: {pct_del:.1f}%  (rollout-level)")
            print("  (paper finding 2: typically 57–70%, gap widens with scale)")
    print(f"Saved subset to     : {args.output_csv}")
    print("===================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct a diversity-filtered subset for GRPO warm-up."
    )

    parser.add_argument("--input_csv", required=True,
                        help="Training CSV with question/options/answer columns.")
    parser.add_argument("--output_csv", required=True,
                        help="Path to save the diversified subset CSV.")

    parser.add_argument("--model_name", required=True,
                        help="Initial policy π_θ_0 (HF id or local path).")
    parser.add_argument("--model_type", default="qwen")
    parser.add_argument("--domain", choices=["math", "science"], required=True)

    parser.add_argument("--num_rollouts", type=int, default=16,
                        help="K — number of rollouts per query (paper uses 16).")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(parse_args())
