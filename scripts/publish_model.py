"""User-run upload of the already benchmarked and evaluated release."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, help="Your Hugging Face username/model-repository")
parser.add_argument("--folder", default="deploy/vercel-api/model")
args = parser.parse_args()
folder = Path(args.folder)
if not (folder / "manifest.json").exists():
    raise SystemExit("Run main.py prepare-vercel before uploading")
card = json.loads((folder / "model_card.json").read_text())
if not (card.get("deployment_benchmark") or {}).get("passed") or not card.get("test_metrics"):
    raise SystemExit("The release needs a successful benchmark and explicit test evaluation")
api = HfApi()
api.create_repo(args.repo, private=True, exist_ok=True)
if not api.model_info(args.repo).private:
    raise SystemExit("Use a private model repository; the existing repository is public")
commit = api.upload_folder(repo_id=args.repo, folder_path=str(folder), commit_message="Publish evaluated inference artifact")
print("MODEL_REPO_ID=" + args.repo)
print("MODEL_REVISION=" + commit.oid)
