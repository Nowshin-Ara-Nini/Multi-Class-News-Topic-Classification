"""Explicit final evaluation, portable export, and Vercel preparation commands."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact_digest(folder):
    folder = Path(folder)
    h = hashlib.sha256()
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.name not in {"model_card.json", "history.json", "comparison.json", "manifest.json"}:
            h.update(file.relative_to(folder).as_posix().encode())
            h.update(digest(file).encode())
    return h.hexdigest()


def attach_test_metrics(folder, csv, metrics):
    card_path = folder / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    card["test_metrics"] = dict(metrics, dataset_sha256=digest(csv), artifact_sha256=artifact_digest(folder),
                                protocol="Explicit test-only evaluation; no model selection")
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")


def evaluate_selected(args):
    import pandas as pd
    from main import _evaluate_checkpoint
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for family, record in selection["families"].items():
        folder = Path(record["metrics"]["model_dir"])
        metrics = _evaluate_checkpoint(folder, frame, output / family)
        attach_test_metrics(folder, Path(args.data), metrics)
        results[family] = metrics
    (output / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame.from_dict(results, orient="index").to_csv(output / "test_results.csv")
    print("Evaluated frozen per-family selections. The validation-selected winner was not changed.")


def comparison(selection_path):
    if not Path(selection_path).exists():
        return []
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    rows = []
    study_path = Path(selection_path).with_name("study.json")
    study = json.loads(study_path.read_text()) if study_path.exists() else {"trials": {}}
    for family, record in selection["families"].items():
        card_path = Path(record["metrics"]["model_dir"]) / "model_card.json"
        card = json.loads(card_path.read_text())
        baselines = [r["metrics"]["best_val_f1"] for r in study["trials"].values()
                     if r["status"] == "complete" and r["phase"] == "screen" and r["family"] == family]
        rows.append({"family": family, "baseline_val_f1": max(baselines) if baselines else None,
                     "val_f1": record["metrics"]["best_val_f1"],
                     "test_f1": (card.get("test_metrics") or {}).get("f1_macro"),
                     "training_seconds": record["metrics"].get("training_seconds"),
                     "selected": family == selection["winner"]})
    return sorted(rows, key=lambda row: row["val_f1"], reverse=True)


def export_model(args):
    import torch
    import joblib
    source = Path(args.model) if args.model else Path(json.loads(Path(args.selection).read_text())["model_dir"])
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("Export destination must be empty. Use a new --output directory to preserve previous releases.")
    target.mkdir(parents=True, exist_ok=True)
    pipeline = json.loads((source / "pipeline_config.json").read_text(encoding="utf-8"))
    if pipeline["feature_type"] == "word2vec":
        raise ValueError("Legacy Word2Vec needs the full external embedding model. Train/export a word2vec_ids checkpoint first.")
    for name in ("config.yaml", "label_encoder.joblib", "vectorizer.joblib", "model_card.json", "model.joblib"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    prep = joblib.load(source / "preprocessor.joblib")
    if hasattr(prep, "clear_cache"):
        prep.clear_cache()
    joblib.dump(prep, target / "preprocessor.joblib")
    if (source / "tokenizer").exists():
        shutil.copytree(source / "tokenizer", target / "tokenizer")
    if (source / "model.pt").exists():
        checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
        kwargs = dict(checkpoint.get("model_kwargs") or pipeline["model_kwargs"])
        model_name = checkpoint.get("model_name") or pipeline["model_name"]
        if model_name == "distilbert" and "base_config" not in kwargs:
            config_path = source / "base_model" / "config.json"
            if not config_path.exists():
                raise ValueError("Legacy DistilBERT needs its local base_model/config.json for offline export")
            kwargs["base_config"] = json.loads(config_path.read_text())
        weights = checkpoint["model_state_dict"]
        if args.quantize:
            from src.models import create_model
            model = create_model(model_name, **kwargs).eval()
            model.load_state_dict(weights)
            weights = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).state_dict()
        torch.save(dict(model_name=model_name, model_kwargs=kwargs, model_state_dict=weights,
                        quantized=args.quantize), target / "model.pt")
        pipeline.update(model_name=model_name, model_kwargs=kwargs, quantized=bool(args.quantize))
    elif args.quantize:
        raise ValueError("INT8 export applies only to PyTorch checkpoints")
    (target / "pipeline_config.json").write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    card_path = target / "model_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    card.update(test_metrics=None, deployment_benchmark=None, export_source=str(source.resolve()),
                quantized=bool(args.quantize), artifact_sha256=artifact_digest(target))
    weight = target / ("model.pt" if (target / "model.pt").exists() else "model.joblib")
    card["weights_sha256"] = digest(weight)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    (target / "comparison.json").write_text(json.dumps(comparison(args.selection), indent=2), encoding="utf-8")
    print(f"Exported {target}. Benchmark and evaluate this artifact explicitly before deployment.")


def benchmark(args):
    """Explicit user-run CPU profiling and full validation parity; never loads test data."""
    import gc
    import os
    import threading
    import time
    import psutil
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from configs.config import Config
    from src.training_data import training_split
    from src.inference import NewsClassifier

    torch.set_num_threads(1)
    config = Config.from_yaml(Path(args.reference) / "config.yaml")
    _, val, _ = training_split(config)
    texts = val[config.data.text_column].tolist()
    process = psutil.Process(os.getpid())
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    deployed = NewsClassifier.from_checkpoint(args.model, device="cpu")
    deployed.config.inference.batch_size = 10
    deployed.predict(texts[0])
    cold = time.perf_counter() - started
    times = []
    for text in texts[:20]:
        started = time.perf_counter()
        deployed.predict(text)
        times.append(time.perf_counter() - started)
    started = time.perf_counter()
    deployed.batch_predict(texts[:10])
    batch_seconds = time.perf_counter() - started
    predictions = deployed.batch_predict(texts)
    labels = deployed.label_encoder.transform(val[config.data.label_column])
    deployed_f1 = f1_score(labels, [r["class_id"] for r in predictions], average="macro")
    stop.set()
    thread.join()
    del deployed
    gc.collect()
    reference = NewsClassifier.from_checkpoint(args.reference, device="cpu")
    reference.config.inference.batch_size = 10
    ref_predictions = reference.batch_predict(texts)
    ref_f1 = f1_score(labels, [r["class_id"] for r in ref_predictions], average="macro")
    report = dict(artifact_sha256=artifact_digest(args.model), peak_rss_mb=peak[0] / 1024**2,
                  cold_seconds=cold, warm_p95_seconds=float(np.quantile(times, .95)), batch10_seconds=batch_seconds,
                  validation_macro_f1=float(deployed_f1), reference_macro_f1=float(ref_f1),
                  validation_drop=float(ref_f1 - deployed_f1), test_data_accessed=False)
    report["passed"] = bool(report["peak_rss_mb"] < 1536 and cold < 60 and report["warm_p95_seconds"] < 5
                            and batch_seconds < 60 and report["validation_drop"] <= .001)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    card_path = Path(args.model) / "model_card.json"
    card = json.loads(card_path.read_text())
    card["deployment_benchmark"] = report
    card["artifact_sha256"] = report["artifact_sha256"]
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Deployment gates failed; do not publish this artifact unchanged.")


def prepare_vercel(args):
    project = Path(__file__).resolve().parents[1]
    target = Path(args.output).resolve()
    model_dir = target / "model"
    card = json.loads((model_dir / "model_card.json").read_text())
    fingerprint = artifact_digest(model_dir)
    gate = card.get("deployment_benchmark") or {}
    if not gate.get("passed") or gate.get("artifact_sha256") != fingerprint:
        raise ValueError("Run the CPU benchmark successfully on this exact exported artifact first")
    test = card.get("test_metrics") or {}
    if test.get("artifact_sha256") != fingerprint:
        raise ValueError("Run explicit evaluate on this exact exported artifact before deployment")
    for directory in ("src", "configs", "api"):
        if target == project or project / directory == target / directory:
            raise ValueError("Use an isolated deployment directory")
        shutil.copytree(project / directory, target / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    template = project / "deploy" / "templates"
    for file in template.iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
    (model_dir / "comparison.json").write_text(json.dumps(comparison(args.selection), indent=2), encoding="utf-8")
    manifest = {file.relative_to(model_dir).as_posix(): digest(file) for file in model_dir.rglob("*")
                if file.is_file() and file.name != "manifest.json"}
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Serialized estimators must use the same package versions in production.
    from importlib.metadata import version
    requirements = (target / "requirements.txt").read_text()
    import re
    for package in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "joblib", "nltk"):
        resolved = version(package).split("+")[0]
        if package == "torch":
            resolved += "+cpu"
        requirements = re.sub(r"^" + re.escape(package) + r"[>=<].*$", f"{package}=={resolved}", requirements, flags=re.M)
    (target / "requirements.txt").write_text(requirements, encoding="utf-8")
    with (target / "requirements.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"\ntokenizers=={version('tokenizers')}\n")
    print(f"Prepared {target}. Deploy the web/ and this API directory as separate Vercel projects.")
