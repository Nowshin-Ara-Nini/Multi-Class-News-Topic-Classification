"""Fetch a release from an immutable private Hub revision at build time."""
import hashlib
import json
import os
import re
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ["MODEL_REPO_ID"]
revision = os.environ["MODEL_REVISION"]
if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
    raise ValueError("MODEL_REVISION must be a full commit SHA, not a mutable branch")
root = Path("model").resolve()
root.mkdir(exist_ok=True)
manifest_file = hf_hub_download(repo, "manifest.json", revision=revision, token=os.environ["HF_TOKEN"])
manifest = json.loads(Path(manifest_file).read_text())
for name, expected in manifest.items():
    destination = (root / name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Invalid artifact path")
    downloaded = Path(hf_hub_download(repo, name, revision=revision, token=os.environ["HF_TOKEN"]))
    content = downloaded.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
pipeline = json.loads((root / "pipeline_config.json").read_text())
if pipeline["preprocessing_mode"] != "raw":
    import nltk
    for resource in ("wordnet", "stopwords", "punkt_tab", "omw-1.4"):
        if not nltk.download(resource, download_dir="nltk_data", quiet=True, raise_on_error=True):
            raise RuntimeError(f"Missing NLTK resource: {resource}")
print(f"Prepared {len(manifest)} verified model files")
