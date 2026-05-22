"""Supervised fine-tuning (SFT) for the CSA model.

A single script for all three SFT variants used in the paper, selected by
the ``sft_mode`` field in the YAML config:

  * ``label``   — Train on (query, ground-truth decision). Produces a model
                  that outputs only the ``<decision>`` tag.
  * ``teacher`` — Train on (query, teacher-written analysis + decision).
                  Requires a pre-computed ``SFT_analysis`` column in the
                  dataset.
  * ``self``    — Train on (query, self-written analysis + decision).
                  Requires a pre-computed ``SFT_analysis`` column produced
                  by the target model itself.

Launch with ``accelerate launch`` and a DeepSpeed config — see
``scripts/training/run_sft.sh`` for an example.
"""

import argparse
import os
from datetime import datetime

import torch
import wandb
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.training.chat_template import render_user_prompt
from src.training.prompts import get_prompt_template, get_response_template


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
    else:
        print("[warn] WANDB_API_KEY not set; W&B logging may be disabled.")


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _user_content(prompt_template: str, example: dict, domain: str) -> str:
    if domain == "math":
        return prompt_template.format(query=example["question"])
    return prompt_template.format(
        question=example["question"], choices=example["choices"],
    )


def _assistant_content(response_template: str, example: dict, sft_mode: str) -> str:
    decision = "SELF_SOLVE" if int(example["is_correct"]) == 1 else "DELEGATE"
    if sft_mode == "label":
        return response_template.format(decision=decision)
    return response_template.format(
        decision=decision, analysis=example["SFT_analysis"],
    )


def build_example(
    tokenizer,
    example: dict,
    prompt_template: str,
    response_template: str,
    domain: str,
    sft_mode: str,
    model_type: str,
):
    """Render one dataset row into a (prompt, completion) pair."""
    user = _user_content(prompt_template, example, domain)
    assistant = _assistant_content(response_template, example, sft_mode)

    prompt = render_user_prompt(tokenizer, user, model_type)
    completion = assistant + tokenizer.eos_token
    return prompt, completion


def required_columns(domain: str, sft_mode: str) -> set:
    cols = {"question", "is_correct"}
    if domain == "science":
        cols.add("choices")
    if sft_mode in {"teacher", "self"}:
        cols.add("SFT_analysis")
    return cols


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int,
    domain: str,
    sft_mode: str,
    model_type: str,
):
    """Load the dataset, render prompts/completions, and filter overlong rows."""
    ds = load_dataset(dataset_name, trust_remote_code=False)
    prompt_template = get_prompt_template(domain, sft_mode)
    response_template = get_response_template(sft_mode)
    cols = required_columns(domain, sft_mode)

    def render_batch(batch):
        prompts, completions = [], []
        n = len(batch["question"])
        for i in range(n):
            example = {c: batch[c][i] for c in cols}
            p, c = build_example(
                tokenizer, example, prompt_template, response_template,
                domain, sft_mode, model_type,
            )
            prompts.append(p)
            completions.append(c)
        return {"prompt": prompts, "completion": completions}

    def fits_in_max_length(example) -> bool:
        ids = tokenizer(example["prompt"] + example["completion"])["input_ids"]
        return len(ids) <= max_length

    split = ds["train"]
    split = split.map(render_batch, batched=True, remove_columns=split.column_names)
    before = len(split)
    split = split.filter(fits_in_max_length)
    print(
        f"Filtered {before - len(split)} over-length samples "
        f"(>{max_length} tokens). Kept {len(split)}."
    )

    print("=== Sample rendered example ===")
    print("--- prompt ---")
    print(split[0]["prompt"])
    print("--- completion ---")
    print(split[0]["completion"])
    print("===============================")
    return split


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_run_name(config: dict) -> str:
    model_short = config["model_name"].split("/")[-1]
    dataset_short = config["dataset_name"].split("/")[-1]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"{timestamp}_{model_short}_{dataset_short}_{config['sft_mode']}"


def build_training_args(config: dict) -> SFTConfig:
    mp = config.get("mixed_precision")
    return SFTConfig(
        output_dir=os.path.join(config["base_output_dir"], "SFT_checkpoints"),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        fp16=(mp == "fp16"),
        bf16=(mp == "bf16"),
        logging_strategy=config["logging_strategy"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        save_only_model=config["save_only_model"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["seed"],
        report_to=config["report_to"],
        dataset_text_field=config["dataset_text_field"],
        dataset_num_proc=config["dataset_num_proc"],
        completion_only_loss=config["completion_only_loss"],
    )


def train(model, train_dataset, config: dict):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb_login()
        wandb.init(
            project=config["wandb_project"],
            name=build_run_name(config),
            config=config,
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=build_training_args(config),
    )
    stats = trainer.train()

    final_path = os.path.join(config["base_output_dir"], "final_model")
    trainer.save_model(final_path)
    print(f"Model saved to {final_path}")

    wandb.finish()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> None:
    domain = config.get("domain")
    sft_mode = config.get("sft_mode")
    if domain not in {"math", "science"}:
        raise ValueError(f"domain must be 'math' or 'science', got {domain!r}")
    if sft_mode not in {"label", "teacher", "self"}:
        raise ValueError(
            f"sft_mode must be 'label', 'teacher', or 'self', got {sft_mode!r}"
        )
    if "model_type" not in config:
        raise ValueError("config must contain 'model_type' (e.g. 'qwen', 'llama').")


def main():
    parser = argparse.ArgumentParser(description="SFT training for the CSA model.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    validate_config(config)
    print(
        f"[info] domain={config['domain']}, sft_mode={config['sft_mode']}, "
        f"model_type={config['model_type']}"
    )

    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    train_dataset = prepare_dataset(
        dataset_name=config["dataset_name"],
        tokenizer=tokenizer,
        max_length=config["max_seq_length"],
        domain=config["domain"],
        sft_mode=config["sft_mode"],
        model_type=config["model_type"],
    )
    train(model, train_dataset, config)


if __name__ == "__main__":
    main()
