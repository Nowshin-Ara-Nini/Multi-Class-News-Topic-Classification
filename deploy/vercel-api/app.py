"""Vercel FastAPI entry point; model files are fetched during build only."""
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NLTK_DATA", str(Path(__file__).parent / "nltk_data"))
from api.main import app
