"""Resumable, sequential validation-only search with a wall-clock budget."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from configs.config import Config
from src.training_workflow import FAMILIES


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def candidate(config, family, rng, baseline=False):
    c = copy.deepcopy(config)
    if baseline:
        defaults = Config()
        for section in ("classical", "tfidf", "dnn", "sequence", "rnn", "attention", "transformer", "distilbert", "training"):
            setattr(c, section, copy.deepcopy(getattr(defaults, section)))
        c.distilbert.freeze_base = False
        if family == "distilbert":
            c.training.epochs = 5
            c.training.batch_size = 16
            c.training.patience = 2
        c.training.save_last = False
        return c
    c.training.optimizer = "adamw"
    c.training.weight_decay = rng.choice([0.0001, 0.01])
    c.training.use_class_weights = rng.choice([True, False])
    c.training.save_last = False
    c.training.patience = 3
    c.training.learning_rate = rng.choice([0.0003, 0.001])
    c.training.epochs = 15
    c.sequence.trainable_embeddings = True
    c.sequence.max_len = rng.choice([64, 96])
    if family == "logistic_regression":
        c.classical.C = rng.choice([0.25, 1., 4.])
        c.classical.class_weight = rng.choice([None, "balanced"])
        c.tfidf.max_features = 50000
        c.tfidf.use_char = rng.choice([False, True])
    elif family == "dnn":
        c.dnn.residual = True
        c.dnn.dropout = rng.choice([0.1, 0.2, 0.3])
        c.training.learning_rate = rng.choice([0.0001, 0.0003, 0.001])
        c.tfidf.max_features = rng.choice([20000, 50000])
    elif family == "distilbert":
        c.distilbert.freeze_base = False
        c.distilbert.pooling = rng.choice(["cls", "cls_mean"])
        c.distilbert.learning_rate = rng.choice([1e-5, 2e-5, 3e-5])
        c.distilbert.head_lr_multiplier = rng.choice([1., 5.])
        c.distilbert.dropout = rng.choice([0.1, 0.2])
        c.distilbert.max_length = rng.choice([128, 192])
        c.training.batch_size = 16
        c.training.accumulation_steps = 2
        c.training.epochs = 5
        c.training.patience = 2
        c.training.scheduler = "linear_warmup"
        c.training.warmup_ratio = rng.choice([0.06, 0.1])
        c.training.gradient_clip = 1.
    elif family == "transformer":
        c.transformer.normalize_input = True
        c.transformer.pooling = rng.choice(["cls", "mean"])
        c.transformer.num_encoder_layers = rng.choice([2, 4])
        c.transformer.dim_feedforward = rng.choice([600, 1200])
        c.transformer.dropout = rng.choice([0.1, 0.2])
        c.transformer.max_seq_len = c.sequence.max_len
        c.training.learning_rate = rng.choice([0.0001, 0.0003])
    elif family == "attention":
        c.attention.combined_pooling = True
        c.attention.hidden_size = rng.choice([128, 256])
        c.attention.num_layers = rng.choice([1, 2])
        c.attention.attention_dim = rng.choice([64, 128])
        c.attention.dropout = rng.choice([0.1, 0.3])
    else:
        c.rnn.compact_head = True
        c.rnn.hidden_size = rng.choice([128, 256])
        c.rnn.num_layers = rng.choice([1, 2])
        c.rnn.dropout = rng.choice([0.1, 0.3])
        c.rnn.pooling = "mean_max"
    return c


def tune(args):
    if args.trials_per_family < 1:
        raise ValueError("trials-per-family must be positive")
    if args.budget_hours <= 0:
        raise ValueError("budget-hours must be positive")
    config = Config.from_yaml(args.config)
    config.validate()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config.data.split_manifest = str(root / "split_manifest.json")
    from src.training_data import training_split
    _, _, split_hash = training_split(config)
    state_path = root / "study.json"
    signature = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "config_sha256": signature, "split_sha256": split_hash, "trials": {}, "phase_seconds": {}, "test_data_accessed": False}
    if state["config_sha256"] != signature or state["split_sha256"] != split_hash:
        raise ValueError("Existing study uses different data/config. Use a new --output directory.")
    total_budget = args.budget_hours * 3600
    phase_caps = {"screen": total_budget / 3, "tune": total_budget / 3,
                  "confirm": total_budget / 4}
    # The remaining 1/12 is reserved for the user's separate evaluation/report command.
    def completed(family):
        return [v for v in state["trials"].values() if v["status"] == "complete" and v["family"] == family and v["phase"] != "confirm"]

    def run(key, family, c, phase, reference=None):
        if key in state["trials"] and state["trials"][key]["status"] == "complete":
            return
        available = phase_caps[phase] - state["phase_seconds"].get(phase, 0.)
        if available <= 0:
            return
        folder = root / key
        folder.mkdir(parents=True, exist_ok=True)
        for field in ("saved_models_dir", "results_dir", "logs_dir", "checkpoints_dir"):
            setattr(c.paths, field, str(folder / field))
        c.training.checkpoints_dir = c.paths.checkpoints_dir
        c.data.split_manifest = config.data.split_manifest
        c.seed = 42 if phase != "confirm" else c.seed
        c.validate()
        c.save(folder / "config.yaml")
        record = dict(family=family, phase=phase, status="running", config=str(folder / "config.yaml"),
                      reference=reference, seed=c.seed, preprocessing=c.preprocessing.mode)
        state["trials"][key] = record
        atomic_json(state_path, state)
        start = time.monotonic()
        print(f"[{phase}] {key} (remaining {available / 3600:.2f} hours)", flush=True)
        try:
            with (folder / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "main.py", "train", "--config", str(folder / "config.yaml"),
                                         "--model", family], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=available, cwd=Path(__file__).resolve().parents[1])
            if result.returncode:
                record.update(status="failed", exit_code=result.returncode)
            else:
                results = json.loads((Path(c.paths.results_dir) / "all_results.json").read_text())
                record.update(status="complete", metrics=next(iter(results.values())))
        except subprocess.TimeoutExpired:
            record["status"] = "budget_exhausted"
        finally:
            state["phase_seconds"][phase] = state["phase_seconds"].get(phase, 0.) + time.monotonic() - start
            atomic_json(state_path, state)

    for mode in ("raw", "optimum", "extreme"):
        for family in FAMILIES:
            if family == "distilbert" and mode != "raw":
                continue
            c = candidate(config, family, random.Random(42), baseline=True)
            c.preprocessing.mode = mode
            run(f"baseline_{family}_{mode}", family, c, "screen")
    for trial in range(args.trials_per_family):
        for index, family in enumerate(FAMILIES):
            available = completed(family)
            if not available:
                continue
            best = max(available, key=lambda r: r["metrics"]["best_val_f1"])
            c = candidate(config, family, random.Random(42 + trial * 100 + index))
            c.preprocessing.mode = best["preprocessing"]
            run(f"candidate_{family}_{trial}", family, c, "tune")
    winners = {}
    for family in FAMILIES:
        records = completed(family)
        if records:
            winners[family] = max(records, key=lambda r: (r["metrics"]["best_val_f1"], -r["metrics"].get("inference_time_ms", float("inf"))))
    finalists = sorted(winners, key=lambda f: winners[f]["metrics"]["best_val_f1"], reverse=True)[:2]
    for seed in (43, 44):
        for family in finalists:
            c = Config.from_yaml(winners[family]["config"])
            c.seed = seed
            reference = winners[family]["config"]
            reference_id = hashlib.sha256(reference.encode()).hexdigest()[:8]
            run(f"confirm_{family}_{reference_id}_{seed}", family, c, "confirm", reference=reference)
    confirmation = {}
    for family in finalists:
        scores = [winners[family]["metrics"]["best_val_f1"]]
        for record in state["trials"].values():
            if record["status"] == "complete" and record["phase"] == "confirm" and record["family"] == family and record["reference"] == winners[family]["config"]:
                scores.append(record["metrics"]["best_val_f1"])
        import statistics
        confirmation[family] = {"runs": len(scores), "mean": statistics.mean(scores),
                                "std": statistics.pstdev(scores)}
    if finalists:
        # Compare seed means only if both candidates completed the same three seeds.
        confirmed = all(confirmation[f]["runs"] == 3 for f in finalists)
        best_family = (max(finalists, key=lambda f: (confirmation[f]["mean"], -confirmation[f]["std"],
                       -winners[f]["metrics"].get("inference_time_ms", float("inf"))))
                       if confirmed else finalists[0])
        selected = dict(schema_version=1, selection_metric="validation_macro_f1", split_sha256=split_hash,
                        winner=best_family, model_dir=winners[best_family]["metrics"]["model_dir"],
                        families=winners, confirmation=confirmation, confirmation_complete=confirmed,
                        missing_families=[f for f in FAMILIES if f not in winners], test_data_accessed=False)
        atomic_json(root / "selection.json", selected)
    print(f"Study saved to {root}. No test data was read. Run evaluate-selected explicitly when ready.")
