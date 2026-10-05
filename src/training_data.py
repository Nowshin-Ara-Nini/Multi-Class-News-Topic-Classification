"""Training-only cleaning and reproducible splits. Never opens a test file."""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path


def identity(text: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(str(text))).casefold()
    value = re.sub(r"<[^>]*>", " ", value)
    return " ".join(re.findall(r"\w+", value))


def training_split(config):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    source = config.paths.train_path.resolve()
    # Compare paths, not contents: training must work with the test CSV absent.
    if source == config.paths.test_path.resolve() or source.name.casefold() == "test_data.csv":
        raise ValueError("The test CSV cannot be used as training data")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source)
    text, label = config.data.text_column, config.data.label_column
    if not {text, label}.issubset(df.columns):
        raise ValueError(f"Training CSV needs columns {text!r} and {label!r}")
    original_count = len(df)
    df = df.dropna(subset=[text, label]).copy()
    df["_identity"] = df[text].map(identity)
    df = df[df["_identity"].ne("")]
    conflicts = df.groupby("_identity")[label].nunique()
    df = df[~df["_identity"].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates("_identity").sort_values("_identity").reset_index(drop=True)
    if set(df[label]) != set(config.data.class_names):
        raise ValueError("Training labels do not match configured classes")
    manifest_path = Path(config.data.split_manifest or Path(config.paths.results_dir) / "split_manifest.json")
    protocol = {"source_sha256": checksum, "normalization_version": 1,
                "seed": config.data.random_state, "val_split": config.data.val_split,
                "text_column": text, "label_column": label}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            raise ValueError("Split manifest differs from training data/config; choose a new run directory")
        train_ids, val_ids = manifest["train_indices"], manifest["validation_indices"]
        if set(train_ids) & set(val_ids) or sorted(train_ids + val_ids) != list(range(len(df))):
            raise ValueError("Invalid or overlapping split indices")
    else:
        train_ids, val_ids = train_test_split(list(range(len(df))), test_size=config.data.val_split,
                                             random_state=config.data.random_state, stratify=df[label])
        manifest = {"protocol": protocol, "original_rows": original_count, "clean_rows": len(df),
                    "conflicting_groups": int((conflicts > 1).sum()),
                    "class_counts": df[label].value_counts().to_dict(),
                    "train_indices": train_ids, "validation_indices": val_ids,
                    "test_data_accessed": False}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    split_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return df.iloc[train_ids].copy(), df.iloc[val_ids].copy(), split_hash
