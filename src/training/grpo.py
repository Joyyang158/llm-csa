"""GRPO trainer for CSA — the RLVR stage of the paper.

Implements the inner ``TRAIN-GRPO`` procedure from Algorithm 1. The reward
is binary (paper §3.2(d)):

  * +1.0 if the model's parsed decision matches the ground-truth
    ``is_correct`` label (1 -> SELF_SOLVE, 0 -> DELEGATE).
  * -1.0 otherwise (including unparseable outputs).

This script is the *single-stage* trainer. The full RLVR pipeline of
Algorithm 1 runs it twice:

  Stage 1 — Diversity-Filtered Warm-up (DFW). Train on the diversified
            subset D_div produced by ``src/training/dfw.py``. Breaks the
            initial SELF_SOLVE prior and yields θ_warm.
  Stage 2 — Full GRPO. Resume from θ_warm and train on the full dataset D.

The two-stage launcher (``scripts/training/run_rlvr.sh``) chains those
calls. For OLMo-2 models, paper Appendix D.6 notes that DFW is unnecessary
because rollouts are already diverse — in that case run only Stage 2 with
``scripts/training/run_grpo.sh``.

Full-parameter fine-tuning only (no LoRA).
"""

import argparse
import os
from datetime import datetime
from typing import List

import torch
import wandb
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.training.chat_template import render_user_prompt
from src.training.prompts import GRPO_PROMPT_TEMPLATE
from src.utils.parsing import parse_decision


# ---------------------------------------------------------------------------
# Config + bookkeeping
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def wandb_login() -> None:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)


def load_base_model_and_tokenizer(model_name: str):
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # DeepSpeed handles device placement.
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_reward_fn(print_every: int = 10):
    """Build a GRPO reward function with periodic debug printing.

    The closure keeps a step counter so we can sample one (prompt, completion,
    label) triple every ``print_every`` steps without flooding the logs.
    """
    state = {"step": 0}

    def reward_fn(prompts: List[str], completions: List[str],
                  is_correct: List[int], **kwargs) -> List[float]:
        state["step"] += 1
        rewards = []
        for completion, gt in zip(completions, is_correct):
            prediction = parse_decision(completion)
            rewards.append(1.0 if prediction == int(gt) else -1.0)

        # Always log the reward distribution; KL collapses when this is
        # constant, so it's worth surfacing on every step.
        print(f">>> [Step {state['step']}] Rewards: {rewards}")

        if state["step"] % print_every == 0:
            print(f"\n{'=' * 20} [Step {state['step']}] Sample {'=' * 20}")
            print(f"Prompt:\n{prompts[0]}")
            print(f"Generated:\n{completions[0]}")
            print(f"Ground truth: {is_correct[0]}")
            print(f"{'=' * 60}\n")

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_query(example: dict, domain: str) -> str:
    """Format a single dataset row into a query string."""
    question = example["question"]
    if domain == "science":
        return f"Question: {question}\nChoices:\n{example['choices']}"
    return f"Question: {question}"


def prepare_dataset(dataset_name: str, tokenizer, model_type: str, domain: str):
    """Load training data either from the HF Hub or a local CSV.

    A path ending in ``.csv`` is loaded with the ``csv`` builder so that the
    DFW-produced subset (a local file) and an HF-hosted full dataset can be
    used through the same code path.
    """
    if dataset_name.endswith(".csv"):
        ds = load_dataset("csv", data_files={"train": dataset_name})
    else:
        ds = load_dataset(dataset_name, trust_remote_code=False)

    def format_row(example):
        query = build_query(example, domain)
        user_content = GRPO_PROMPT_TEMPLATE.format(query=query)
        prompt = render_user_prompt(tokenizer, user_content, model_type)
        return {"prompt": prompt, "is_correct": example["is_correct"]}

    ds = ds.map(format_row)
    print("=" * 20 + " Sample prompt " + "=" * 20)
    print(ds["train"][0]["prompt"])
    print("=" * 60)
    return ds["train"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_args(config: dict) -> GRPOConfig:
    return GRPOConfig(
        output_dir=os.path.join(config["base_output_dir"], "GRPO_checkpoints"),
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        max_prompt_length=config["max_prompt_length"],
        beta=config.get("beta", 0.04),
        num_train_epochs=config["epochs"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        save_only_model=config["save_only_model"],
        bf16=True,
        report_to=config["report_to"],
        use_vllm=config["use_vllm"],
        vllm_mode="colocate",
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        eval_accumulation_steps=config.get("eval_accumulation_steps", 1),
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        vllm_gpu_memory_utilization=config["vllm_gpu_memory_utilization"],
    )


def build_run_name(config: dict) -> str:
    model_short = config["model_name"].split("/")[-1]
    dataset_short = config["dataset_name"].split("/")[-1]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"GRPO_{timestamp}_{model_short}_{dataset_short}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> None:
    if config.get("domain") not in {"math", "science"}:
        raise ValueError(
            f"domain must be 'math' or 'science', got {config.get('domain')!r}"
        )
    if "model_type" not in config:
        raise ValueError("config must contain 'model_type' (e.g. 'qwen', 'llama').")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for the CSA model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    validate_config(config)
    print(
        f"[info] domain={config['domain']}, model_type={config['model_type']}"
    )

    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb_login()
        wandb.init(
            project=config["wandb_project"],
            name=build_run_name(config),
            config=config,
        )

    model, tokenizer = load_base_model_and_tokenizer(config["model_name"])
    train_dataset = prepare_dataset(
        dataset_name=config["dataset_name"],
        tokenizer=tokenizer,
        model_type=config["model_type"],
        domain=config["domain"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(print_every=config.get("debug_print_every", 10))],
        args=build_training_args(config),
        train_dataset=train_dataset,
    )

    print("Starting GRPO training ...")
    trainer.train()

    final_path = os.path.join(config["base_output_dir"], "final_model")
    trainer.save_model(final_path)
    print(f"Model saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
