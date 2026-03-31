"""
upload_model.py
───────────────
Run this ONCE locally to push your trained Mask2Former weights to HF Hub.

Usage:
    python upload_model.py --username your-hf-username --token hf_...

After running successfully, update MODEL_REPO in hf_space/app.py to:
    "your-hf-username/offroad-mask2former"
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

MODEL_DIR  = "./runs_mask2former/mask2former_best"
REPO_NAME  = "offroad-mask2former"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True,  help="Your HuggingFace username")
    parser.add_argument("--token",    required=True,  help="Your HuggingFace WRITE token")
    args = parser.parse_args()

    repo_id = f"{args.username}/{REPO_NAME}"
    api     = HfApi(token=args.token)

    print(f"[1/3] Creating repository: {repo_id} ...")
    create_repo(
        repo_id=repo_id,
        token=args.token,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print("      ✅ Repository ready.")

    print(f"[2/3] Uploading model files from '{MODEL_DIR}' ...")
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory '{MODEL_DIR}' not found. "
            "Make sure you have trained the model first."
        )

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned Mask2Former offroad segmentation model",
    )
    print("      ✅ Upload complete.")

    print(f"[3/3] Done! Your model is live at:")
    print(f"      https://huggingface.co/{repo_id}")
    print()
    print("  Next: open hf_space/app.py and set:")
    print(f"      MODEL_REPO = \"{repo_id}\"")
    print("  Then run: python deploy_space.py --username <you> --token <token>")

if __name__ == "__main__":
    main()
