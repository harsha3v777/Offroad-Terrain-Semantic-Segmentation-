"""
deploy_space.py
───────────────
Run this AFTER upload_model.py to push the Gradio app to HF Spaces.

Usage:
    python deploy_space.py --username your-hf-username --token hf_...
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SPACE_DIR  = "./hf_space"
SPACE_NAME = "offroad-segmentation"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",   required=True, help="Your HuggingFace username")
    parser.add_argument("--token",      required=True, help="Your HuggingFace WRITE token")
    parser.add_argument("--model-repo", default=None,   help="Model repo ID (default: username/offroad-mask2former)")
    args = parser.parse_args()

    model_repo = args.model_repo or f"{args.username}/offroad-mask2former"
    space_id   = f"{args.username}/{SPACE_NAME}"
    api        = HfApi(token=args.token)

    # Patch MODEL_REPO into app.py before uploading
    app_path   = Path(SPACE_DIR) / "app.py"
    app_source = app_path.read_text(encoding="utf-8")
    app_source = app_source.replace(
        'os.environ.get("MODEL_REPO", "your-username/offroad-mask2former")',
        f'os.environ.get("MODEL_REPO", "{model_repo}")',
    )
    app_path.write_text(app_source, encoding="utf-8")
    print(f"[✓] Patched MODEL_REPO → {model_repo}")

    print(f"[1/3] Creating Space: {space_id} ...")
    create_repo(
        repo_id=space_id,
        token=args.token,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )
    print("      ✅ Space created.")

    print(f"[2/3] Uploading app files from '{SPACE_DIR}' ...")
    api.upload_folder(
        folder_path=SPACE_DIR,
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy offroad segmentation Gradio app",
    )
    print("      ✅ Upload complete.")

    print(f"[3/3] 🎉 Your Space is live at:")
    print(f"      https://huggingface.co/spaces/{space_id}")
    print()
    print("      Note: It may take 2–5 minutes to build and install dependencies.")

if __name__ == "__main__":
    main()
