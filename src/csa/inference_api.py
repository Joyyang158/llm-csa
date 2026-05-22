"""Run a closed-source model as the CSA model via its API.

Supports OpenAI, Google Gemini, Anthropic Claude, and Together AI. The model
family is selected via ``--model_family``; credentials are read from the
corresponding environment variable (``OPENAI_API_KEY`` / ``GOOGLE_API_KEY`` /
``ANTHROPIC_API_KEY`` / ``TOGETHER_API_KEY``).
"""

import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils.io import save_checkpoint
from src.utils.parsing import format_options, parse_decision
from src.utils.prompts import CSA_DECISION_WITH_ANALYSIS

logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def call_openai(model_name: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_gemini(model_name: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


def call_claude(model_name: str, prompt: str) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def call_together(model_name: str, prompt: str) -> str:
    from together import Together
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content


PROVIDER_MAP = {
    "openai": call_openai,
    "gpt": call_openai,
    "google": call_gemini,
    "gemini": call_gemini,
    "anthropic": call_claude,
    "claude": call_claude,
    "together": call_together,
    "togetherai": call_together,
}


def call_llm(model_name: str, model_family: str, prompt: str) -> str:
    family = model_family.lower()
    if family not in PROVIDER_MAP:
        raise ValueError(
            f"Unsupported model_family: {model_family!r}. "
            f"Options: {sorted(set(PROVIDER_MAP))}"
        )
    return PROVIDER_MAP[family](model_name, prompt)


# ---------------------------------------------------------------------------
# Prompt building
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CSA inference via closed-source APIs.")

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_family", required=True,
                        help="Provider family: openai / gemini / claude / together.")
    parser.add_argument("--domain", choices=["math", "science"], required=True)

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--decision_col", default="decision")
    parser.add_argument("--analysis_col", default="analysis")
    parser.add_argument("--save_every", type=int, default=50)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    start_idx = 0
    if args.decision_col in df.columns:
        already_done = df[args.decision_col].notna()
        if already_done.any():
            start_idx = int(already_done.sum())
            logging.info(
                "[Resume] %d rows already completed; resuming from row %d.",
                start_idx, start_idx,
            )

    for idx in tqdm(range(len(df)), initial=start_idx, total=len(df)):
        if idx < start_idx:
            continue

        row = df.iloc[idx]
        prompt = CSA_DECISION_WITH_ANALYSIS.format(
            query=build_query(row, args.domain)
        )

        if idx == start_idx:
            logging.info("Example prompt:\n%s", prompt)

        text = call_llm(args.model_name, args.model_family, prompt)

        df.at[idx, args.decision_col] = parse_decision(text)
        df.at[idx, args.analysis_col] = text
        save_checkpoint(df, args.output_csv, idx, args.save_every)

    df.to_csv(args.output_csv, index=False)
    logging.info("Saved %d rows to %s", len(df), args.output_csv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()