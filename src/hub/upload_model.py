"""Upload a local model checkpoint folder to the HuggingFace Hub.

Credentials are read from the ``HF_TOKEN`` environment variable; nothing is
hardcoded.
"""

import argparse
import logging
import os

from huggingface_hub import create_repo, login, upload_folder


def main():
    parser = argparse.ArgumentParser(description="Upload a model folder to the HF Hub.")
    parser.add_argument("--local_dir", required=True,
                        help="Local checkpoint directory to upload.")
    parser.add_argument("--repo_id", required=True,
                        help="Target HF repo, e.g. 'username/model-name'.")
    parser.add_argument("--repo_type", default="model",
                        choices=["model", "dataset", "space"])
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a HuggingFace access token with write permissions."
        )

    login(token=token)
    create_repo(
        args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    upload_folder(
        repo_id=args.repo_id,
        folder_path=args.local_dir,
        repo_type=args.repo_type,
    )
    logging.info("Uploaded %s -> %s", args.local_dir, args.repo_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
