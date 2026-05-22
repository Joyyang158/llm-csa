"""Generate model answers on a benchmark CSV via a hosted LLM API.

Supports Together, OpenAI (GPT), Anthropic (Claude), and Google (Gemini).
Reads the API key from a provider-specific environment variable:

    together   -> TOGETHER_API_KEY
    openai     -> OPENAI_API_KEY
    anthropic  -> ANTHROPIC_API_KEY
    google     -> GOOGLE_API_KEY
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from func_timeout import FunctionTimedOut, func_set_timeout
from tqdm import tqdm

from src.utils.io import ensure_object_column, ensure_parent_dir, save_checkpoint
from src.utils.parsing import format_options
from src.utils.prompts import get_benchmark_prompt


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

PROVIDER_ENV = {
    "together":  "TOGETHER_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


def make_client(provider: str):
    env_var = PROVIDER_ENV[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} environment variable is not set.")

    if provider == "together":
        from together import Together
        return Together(api_key=api_key)
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    if provider == "google":
        from google import genai
        return genai.Client(api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# API call (dispatch by provider)
# ---------------------------------------------------------------------------

@func_set_timeout(3600)
def call_api(client, provider: str, model_name: str, prompt: str, max_tokens: int) -> str:
    if provider in ("together", "openai"):
        # Together and OpenAI share the same chat.completions schema
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    if provider == "anthropic":
        resp = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks; concatenate text blocks
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

    if provider == "google":
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"max_output_tokens": max_tokens},
        )
        return resp.text

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Per-row prompt building
# ---------------------------------------------------------------------------

def build_prompt(row: pd.Series, domain: str) -> str:
    if domain == "math":
        return get_benchmark_prompt("math").format(question=row["question"])
    if "options" in row and pd.notna(row["options"]):
        choices = format_options(row["options"])
    else:
        choices = row["choices"]
    return get_benchmark_prompt("science").format(
        question=row["question"], choices=choices
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args):
    client = make_client(args.provider)

    logging.info("Provider      : %s", args.provider)
    logging.info("Domain        : %s", args.domain)
    logging.info("Model         : %s", args.model_name)
    logging.info("Output column : %s", args.output_col)

    ensure_parent_dir(args.output_csv)
    df = pd.read_csv(args.input_csv)
    ensure_object_column(df, args.output_col)

    for i in tqdm(range(len(df)), desc="Generating"):
        if pd.notna(df.at[i, args.output_col]):
            continue

        prompt = build_prompt(df.iloc[i], args.domain)
        generations = []
        gen_times = []
        for _ in range(args.num_generations):
            start = time.time()
            try:
                gen = call_api(client, args.provider, args.model_name,
                               prompt, args.max_tokens)
            except FunctionTimedOut:
                logging.warning("Row %d timed out.", i)
                gen = "Error: Timeout"
            except Exception as e:
                logging.error("Row %d: %s", i, e)
                gen = f"Error: {e}"
            gen_times.append(time.time() - start)
            generations.append(gen)

        if i == 0:
            logging.info("=== Sample prompt ===\n%s", prompt)
            logging.info("=== Sample output ===\n%s", generations)

        df.at[i, args.output_col] = generations
        df.at[i, "avg_generation_time"] = float(np.mean(gen_times))
        save_checkpoint(df, args.output_csv, i, args.save_every)

    df.to_csv(args.output_csv, index=False)
    logging.info("Done: processed %d rows.", len(df))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model answers via a hosted LLM API."
    )
    parser.add_argument("--provider",
                        choices=["together", "openai", "anthropic", "google"],
                        required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_col", required=True)
    parser.add_argument("--domain", choices=["math", "science"], required=True)
    parser.add_argument("--model_name", required=True,
                        help="Provider-specific model identifier.")
    parser.add_argument("--max_tokens", type=int, default=100000)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(parse_args())