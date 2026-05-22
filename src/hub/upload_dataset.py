"""Upload a local CSV file to the HuggingFace Hub as a dataset."""

import argparse
import logging
import os

from datasets import load_dataset
from huggingface_hub import login


def main():
    parser = argparse.ArgumentParser(description="Upload a CSV to the HF Hub.")
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file.")
    parser.add_argument("--repo_id", required=True,
                        help="Target HF repo, e.g. 'username/dataset-name'.")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a HuggingFace access token with write permissions."
        )

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    login(token=token)

    logging.info("Loading %s ...", args.csv_path)
    dataset = load_dataset("csv", data_files={"train": args.csv_path})
    logging.info("Dataset: %s", dataset)

    dataset.push_to_hub(args.repo_id, private=args.private, token=token)
    logging.info("Done: https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
