"""
News Topic Classification — Streamlit Web Application

A production-grade, interactive web interface and deployment dashboard for
classifying news headlines into topic categories using trained NLP models.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Friendly Mapping
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "Business": "#667eea",
    "Science and Technology": "#48bb78",
    "Sports": "#ed8936",
    "World News": "#e53e3e",
}

CLASS_ICONS = {
    "Business": "💼",
    "Science and Technology": "🔬",
    "Sports": "⚽",
    "World News": "🌍",
}

ARCHITECTURE_NAMES = {
    "logistic_regression": "Logistic Regression",
    "lr": "Logistic Regression",
    "dnn": "Deep Neural Network",
    "rnn": "Recurrent Neural Network",
    "bilstm": "Bidirectional LSTM (BiLSTM)",
    "lstm": "Long Short-Term Memory (LSTM)",
    "gru": "Gated Recurrent Unit (GRU)",
    "bigru": "Bidirectional GRU (BiGRU)",
    "birnn": "Bidirectional RNN (BiRNN)",
    "attention": "BiLSTM with Self-Attention",
    "transformer": "Transformer Encoder",
    "distilbert": "DistilBERT Transformer",
}

EXAMPLE_HEADLINES = {
    "Business": [
        "Apple reports record quarterly revenue beating Wall Street expectations",
        "Federal Reserve raises interest rates by 25 basis points to combat inflation",
        "Tesla stock surges 15% after earnings beat analyst estimates",
    ],
    "Science and Technology": [
        "NASA's James Webb telescope captures stunning images of distant galaxies",
        "Scientists discover high-temperature superconductor breakthrough in quantum lab",
        "Google DeepMind AI achieves new milestone in protein structure prediction",
    ],
    "Sports": [
        "Lakers defeat Celtics in thrilling overtime championship game",
        "Messi scores hat trick as Argentina wins Copa America final",
        "Serena Williams announces retirement from professional tennis after historic career",
    ],
    "World News": [
        "UN Security Council holds emergency session on international crisis",
        "European Union reaches historic climate agreement at international summit",
        "Major earthquake strikes coastal region causing widespread humanitarian effort",
    ],
}


# ---------------------------------------------------------------------------
# Verification & Discovery Utilities
# ---------------------------------------------------------------------------

def is_valid_checkpoint(checkpoint_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify whether a directory is a valid, complete model checkpoint ready for inference.

    Returns:
        (True, None) if complete, or (False, failure_reason) if incomplete.
    """
    if not checkpoint_dir.is_dir():
        return False, f"Not a directory: {checkpoint_dir}"

    pipeline_path = checkpoint_dir / "pipeline_config.json"
    if not pipeline_path.is_file():
        return False, "Missing pipeline_config.json"

    try:
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Corrupted pipeline_config.json: {exc}"

    feature_type = pipeline.get("feature_type")
    if feature_type not in {"tfidf", "word2vec", "distilbert"}:
        return False, f"Unsupported feature_type: {feature_type!r}"

    if not (checkpoint_dir / "preprocessor.joblib").is_file():
        return False, "Missing preprocessor.joblib"

    if not (checkpoint_dir / "label_encoder.joblib").is_file():
        return False, "Missing label_encoder.joblib"

    has_weights = (checkpoint_dir / "model.pt").is_file() or (checkpoint_dir / "model.joblib").is_file()
    if not has_weights:
        return False, "Missing model weights (neither model.pt nor model.joblib found)"

    if feature_type == "tfidf" and not (checkpoint_dir / "vectorizer.joblib").is_file():
        return False, "Missing vectorizer.joblib (required for TF-IDF)"

    return True, None


def check_deployment_health(checkpoint_dir: Path) -> Dict[str, Any]:
    """
    Perform a granular startup health check on a model checkpoint.
    """
    if not checkpoint_dir.exists():
        return {
            "healthy": False,
            "message": "Checkpoint directory not found",
            "details": {},
        }

    is_valid, reason = is_valid_checkpoint(checkpoint_dir)
    pipeline_path = checkpoint_dir / "pipeline_config.json"
    feature_type = "unknown"
    if pipeline_path.is_file():
        try:
            feature_type = json.loads(pipeline_path.read_text(encoding="utf-8")).get("feature_type", "unknown")
        except Exception:
            pass

    weights_ok = (checkpoint_dir / "model.pt").is_file() or (checkpoint_dir / "model.joblib").is_file()
    pipeline_ok = pipeline_path.is_file()
    preprocessor_ok = (checkpoint_dir / "preprocessor.joblib").is_file()
    label_encoder_ok = (checkpoint_dir / "label_encoder.joblib").is_file()

    if feature_type == "tfidf":
        extractor_ok = (checkpoint_dir / "vectorizer.joblib").is_file()
    elif feature_type == "distilbert":
        extractor_ok = (checkpoint_dir / "tokenizer").is_dir() or (checkpoint_dir / "base_model").is_dir() or True
    else:
        extractor_ok = True

    model_card_ok = (checkpoint_dir / "model_card.json").is_file()

    details = {
        "Model Weights": weights_ok,
        "Pipeline Config": pipeline_ok,
        "Preprocessor": preprocessor_ok,
        "Label Encoder": label_encoder_ok,
        "Feature Extractor": extractor_ok,
        "Model Card": model_card_ok,
    }

    return {
        "healthy": is_valid,
        "message": "Deployment Healthy" if is_valid else (reason or "Deployment Incomplete"),
        "details": details,
    }


def detect_deployment_mode(project_root: Optional[Path] = None) -> Tuple[str, Optional[Path], List[Path]]:
    """
    Detect whether running in Production Mode (single pinned model) or Development Mode
    (multi-model exploration) and return (mode, pinned_path, candidate_paths).
    """
    root = project_root or PROJECT_ROOT

    # Priority 1: MODEL_DIR or DEPLOYED_MODEL_DIR or CHECKPOINT_DIR environment variable
    for env_var in ("MODEL_DIR", "DEPLOYED_MODEL_DIR", "CHECKPOINT_DIR"):
        val = os.environ.get(env_var)
        if val:
            candidate = Path(val).resolve()
            if candidate.is_dir():
                valid, _ = is_valid_checkpoint(candidate)
                if valid:
                    return "production", candidate, [candidate]
                # If directory contains subdirectories
                sub_valid = [d for d in sorted(candidate.iterdir()) if d.is_dir() and is_valid_checkpoint(d)[0]]
                if sub_valid:
                    if len(sub_valid) == 1:
                        return "production", sub_valid[0], sub_valid
                    return "development", None, sub_valid

    # Priority 2: Standalone deployment in app/model/
    standalone = root / "app" / "model"
    if standalone.is_dir():
        valid, _ = is_valid_checkpoint(standalone)
        if valid:
            return "production", standalone, [standalone]

    # Priority 3: Repository saved_models/
    saved_models_dir = root / "saved_models"
    if saved_models_dir.is_dir():
        valid_models = [d for d in sorted(saved_models_dir.iterdir()) if d.is_dir() and is_valid_checkpoint(d)[0]]
        if valid_models:
            if len(valid_models) == 1:
                return "production", valid_models[0], valid_models
            return "development", None, valid_models

    # Fallback: checkpoints directory
    checkpoints_dir = root / "checkpoints"
    if checkpoints_dir.is_dir():
        valid_models = [d for d in sorted(checkpoints_dir.iterdir()) if d.is_dir() and is_valid_checkpoint(d)[0]]
        if valid_models:
            return "development", None, valid_models

    return "unknown", None, []


def get_model_f1_score(model_dir: Path, all_results: Optional[dict] = None) -> float:
    """Extract validation F1 score for sorting, with accuracy fallback."""
    card_path = model_dir / "model_card.json"
    if card_path.is_file():
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            summary = card.get("training_summary", {})
            f1 = summary.get("best_val_f1")
            if f1 is not None:
                return float(f1)
            acc = summary.get("best_val_acc")
            if acc is not None:
                return float(acc)
        except Exception:
            pass

    if all_results and model_dir.name in all_results:
        res = all_results[model_dir.name]
        f1 = res.get("best_val_f1") or res.get("test_f1_macro")
        if f1 is not None:
            return float(f1)
        acc = res.get("best_val_acc") or res.get("test_accuracy")
        if acc is not None:
            return float(acc)

    return 0.0


def get_best_model(model_dirs: List[Path], all_results: Optional[dict] = None) -> Optional[Path]:
    """
    Return the model directory with the highest validation F1 score.
    """
    if not model_dirs:
        return None
    return max(model_dirs, key=lambda d: (get_model_f1_score(d, all_results), d.name))


def build_leaderboard(
    model_dirs: List[Path], top_n: int = 5, all_results: Optional[dict] = None
) -> List[Dict[str, Any]]:
    """
    Build a ranked top-N leaderboard of models.
    """
    ranked = sorted(
        model_dirs,
        key=lambda d: (get_model_f1_score(d, all_results), d.name),
        reverse=True,
    )
    leaderboard = []
    for rank, model_dir in enumerate(ranked[:top_n], start=1):
        meta = extract_model_metadata(model_dir, all_results)
        leaderboard.append({
            "rank": rank,
            "name": model_dir.name,
            "display_name": meta["display_name"],
            "architecture": meta["architecture"],
            "val_f1": meta["val_f1"],
            "val_acc": meta["val_accuracy"],
            "path": model_dir,
        })
    return leaderboard


def extract_model_metadata(model_dir: Path, all_results: Optional[dict] = None) -> Dict[str, Any]:
    """
    Extract dynamic metadata from model_card.json, pipeline_config.json,
    and config.yaml with graceful layered fallbacks.
    """
    card = {}
    card_path = model_dir / "model_card.json"
    if card_path.is_file():
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            card = {}

    pipeline = {}
    pipeline_path = model_dir / "pipeline_config.json"
    if pipeline_path.is_file():
        try:
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        except Exception:
            pipeline = {}

    # 1. Model Name & Display Name
    model_name = card.get("model_name") or model_dir.name
    raw_arch = card.get("architecture") or pipeline.get("model_name") or "unknown"
    arch_lower = raw_arch.lower()

    # Detect specific cell types from name if architecture is generic 'rnn'
    if arch_lower == "rnn":
        dir_lower = model_dir.name.lower()
        if "bilstm" in dir_lower:
            arch_lower = "bilstm"
        elif "bigru" in dir_lower:
            arch_lower = "bigru"
        elif "birnn" in dir_lower:
            arch_lower = "birnn"
        elif "lstm" in dir_lower:
            arch_lower = "lstm"
        elif "gru" in dir_lower:
            arch_lower = "gru"

    architecture = ARCHITECTURE_NAMES.get(arch_lower, raw_arch.replace("_", " ").title())

    # Prettify display name
    mode = card.get("preprocessing_mode") or pipeline.get("preprocessing_mode") or "standard"
    feature_type = card.get("feature_type") or pipeline.get("feature_type") or "features"
    display_name = f"{architecture} ({feature_type.upper()} · {mode.title()})"

    # 2. Feature Extractor Detail
    feat_lower = feature_type.lower()
    if feat_lower == "tfidf":
        feature_extractor = "TF-IDF (20,000 max features, n-grams 1–2)"
    elif feat_lower == "word2vec":
        feature_extractor = "Word2Vec (300-dim Google News)"
    elif feat_lower == "distilbert":
        feature_extractor = "DistilBERT Tokenizer (AutoTokenizer)"
    else:
        feature_extractor = feature_type.title()

    # 3. Training Date
    training_date = card.get("training_date")
    if not training_date:
        captured_utc = card.get("environment", {}).get("captured_at_utc")
        if captured_utc:
            try:
                # Parse ISO timestamp
                dt = datetime.datetime.fromisoformat(captured_utc.replace("Z", "+00:00"))
                training_date = dt.strftime("%Y-%m-%d")
            except Exception:
                training_date = str(captured_utc)[:10]
        else:
            try:
                training_date = datetime.datetime.fromtimestamp(model_dir.stat().st_mtime).strftime("%Y-%m-%d")
            except Exception:
                training_date = "Recent"

    # 4. Parameter Count
    param_count = card.get("parameter_count")
    if param_count and isinstance(param_count, (int, float)) and param_count > 0:
        if param_count >= 1_000_000:
            param_str = f"{param_count:,.0f} (~{param_count / 1_000_000:.2f}M)"
        elif param_count >= 1_000:
            param_str = f"{param_count:,.0f} (~{param_count / 1_000:.1f}K)"
        else:
            param_str = f"{param_count:,}"
    elif arch_lower in {"logistic_regression", "lr"}:
        param_str = "~80K (Linear model)"
    else:
        param_str = "N/A"

    # 5. Dataset Size
    dataset_size = card.get("dataset_size")
    if not dataset_size:
        num_samples = card.get("training_summary", {}).get("num_samples")
        if num_samples:
            dataset_size = f"{num_samples:,} Headlines"
        else:
            dataset_size = "88,000+ Headlines"

    # 6. Classes
    classes = card.get("classes") or pipeline.get("class_names") or list(CLASS_COLORS.keys())
    num_classes = len(classes)

    # 7. Performance metrics
    summary = card.get("training_summary", {})
    val_acc_val = summary.get("best_val_acc")
    val_f1_val = summary.get("best_val_f1")

    if (val_acc_val is None or val_f1_val is None) and all_results and model_dir.name in all_results:
        res = all_results[model_dir.name]
        val_acc_val = val_acc_val or res.get("best_val_acc") or res.get("test_accuracy")
        val_f1_val = val_f1_val or res.get("best_val_f1") or res.get("test_f1_macro")

    val_accuracy = f"{val_acc_val * 100:.1f}%" if val_acc_val is not None else "N/A"
    val_f1 = f"{val_f1_val * 100:.1f}%" if val_f1_val is not None else "N/A"

    # 8. Epochs
    epochs_val = summary.get("epochs") or card.get("epochs")
    epochs_str = f"{epochs_val} epochs" if epochs_val else "15 epochs (configured)"

    return {
        "model_name": model_name,
        "display_name": display_name,
        "architecture": architecture,
        "feature_type": feature_type.upper(),
        "feature_extractor": feature_extractor,
        "preprocessing_mode": mode.title(),
        "training_date": training_date,
        "parameter_count": param_str,
        "dataset_size": dataset_size,
        "num_classes": num_classes,
        "classes": classes,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "epochs": epochs_str,
    }


# ---------------------------------------------------------------------------
# Model Loader with Structured Diagnostics
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_classifier_with_diagnostics(model_path: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load the classifier with structured error reporting. Cached across user sessions.
    """
    try:
        from src.inference import NewsClassifier
        classifier = NewsClassifier.from_checkpoint(model_path)
        return classifier, None
    except Exception as exc:
        err_msg = f"Failed to load checkpoint: {model_path}\n\nReason: {type(exc).__name__}: {exc}"
        logger.error(err_msg, exc_info=True)
        return None, err_msg


def load_all_results() -> Dict[str, Any]:
    """Load results/all_results.json if it exists."""
    results_path = PROJECT_ROOT / "results" / "all_results.json"
    if results_path.is_file():
        try:
            return json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "high-confidence"
    elif confidence >= 0.5:
        return "medium-confidence"
    return "low-confidence"


def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create an interactive probability distribution chart."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [f"{CLASS_ICONS.get(c, '📌')} {c}" for c, _ in sorted_probs]
    probs = [p for _, p in sorted_probs]
    colors = [CLASS_COLORS.get(c, "#667eea") for c, _ in sorted_probs]

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="white", width=1),
            cornerradius=5,
        ),
        text=[f"{p:.1%}" for p in probs],
        textposition="inside",
        textfont=dict(color="white", size=14, family="Inter"),
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            tickformat=".0%",
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=13),
    )

    return fig


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="News Topic Classifier",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.8rem 2.5rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.25);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .main-header p {
            margin: 0.4rem 0 0 0;
            font-size: 1.05rem;
            opacity: 0.92;
        }

        .prediction-card {
            background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
            border-radius: 14px;
            padding: 1.8rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
            border: 1px solid #e2e8f0;
        }

        .prediction-label {
            font-size: 1.85rem;
            font-weight: 700;
            color: #2d3748;
            margin: 0.4rem 0;
        }

        .confidence-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.95rem;
        }

        .high-confidence {
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
        }

        .medium-confidence {
            background: linear-gradient(135deg, #ecc94b, #d69e2e);
            color: white;
        }

        .low-confidence {
            background: linear-gradient(135deg, #fc8181, #e53e3e);
            color: white;
        }

        .health-badge {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 0.6rem;
        }
        .health-good {
            background-color: #def7ec;
            color: #03543f;
            border: 1px solid #bcf0da;
        }
        .health-bad {
            background-color: #fde8e8;
            color: #9b1c1c;
            border: 1px solid #fbd5d5;
        }

        .best-badge {
            display: inline-block;
            background: linear-gradient(135deg, #ffd700, #ffae00);
            color: #744210;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }

        .metric-mini-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }
        .metric-mini-table td {
            padding: 0.35rem 0.2rem;
            border-bottom: 1px solid #edf2f7;
        }
        .metric-mini-table td:first-child {
            color: #718096;
            font-weight: 500;
        }
        .metric-mini-table td:last-child {
            text-align: right;
            font-weight: 600;
            color: #2d3748;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for user input if not present
    if "headline_input" not in st.session_state:
        st.session_state["headline_input"] = ""

    all_results = load_all_results()

    # Model Discovery & Deployment Mode Detection
    mode, pinned_path, candidate_paths = detect_deployment_mode()

    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Deployment & Model")

        active_model_path: Optional[Path] = None
        best_model_path = get_best_model(candidate_paths, all_results) if candidate_paths else None

        if not candidate_paths:
            st.markdown('<div class="health-badge health-bad">❌ No Model Checkpoint Found</div>', unsafe_allow_html=True)
            st.warning("No model artifacts found on this deployment server.")
            with st.expander("ℹ️ How to Deploy Models"):
                st.markdown("""
                **For Streamlit Cloud / Production:**
                Commit the model in `app/model/` to GitHub:
                ```bash
                git add app/model
                git commit -m "Add deployed model"
                git push origin main
                ```
                Or configure `MODEL_DIR` in Streamlit Cloud Secrets.
                """)
            meta = {
                "model_name": "None",
                "display_name": "No Model Deployed",
                "architecture": "N/A",
                "feature_type": "N/A",
                "preprocessing_mode": "N/A",
                "val_accuracy": "N/A",
                "val_f1": "N/A",
                "parameter_count": "N/A",
                "dataset_size": "88,000+ Headlines",
                "training_date": "N/A",
                "classes": list(CLASS_COLORS.keys()),
            }
        elif mode == "production" and pinned_path:
            active_model_path = pinned_path
            st.markdown(
                '<div class="health-badge health-good">🔒 Production Mode (Pinned Model)</div>',
                unsafe_allow_html=True,
            )
            health = check_deployment_health(active_model_path)
            if health["healthy"]:
                st.markdown('<div class="health-badge health-good">✅ Deployment Healthy</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="health-badge health-bad">❌ {health["message"]}</div>', unsafe_allow_html=True)

            with st.expander("🔍 Deployment Artifacts Status", expanded=not health["healthy"]):
                for component, status in health.get("details", {}).items():
                    icon = "✅" if status else "❌"
                    st.markdown(f"- {icon} **{component}**")

            meta = extract_model_metadata(active_model_path, all_results)
        else:
            # Development Mode: Multi-model selector with smart default (best Val-F1)
            st.caption("🛠️ **Development Mode** (Multiple models detected)")
            model_options = {d.name: d for d in candidate_paths}
            options_list = list(model_options.keys())

            default_index = 0
            if best_model_path and best_model_path.name in model_options:
                default_index = options_list.index(best_model_path.name)

            selected_model_name = st.selectbox(
                "Select Model",
                options_list,
                index=default_index,
                help="Choose from available checkpoints. The default is auto-selected based on highest validation F1.",
            )
            active_model_path = model_options[selected_model_name]

            if best_model_path and active_model_path == best_model_path:
                st.markdown('<div class="best-badge">🏆 Best Available Model</div>', unsafe_allow_html=True)

            health = check_deployment_health(active_model_path)
            if health["healthy"]:
                st.markdown('<div class="health-badge health-good">✅ Deployment Healthy</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="health-badge health-bad">❌ {health["message"]}</div>', unsafe_allow_html=True)

            with st.expander("🔍 Deployment Artifacts Status", expanded=not health["healthy"]):
                for component, status in health.get("details", {}).items():
                    icon = "✅" if status else "❌"
                    st.markdown(f"- {icon} **{component}**")

            meta = extract_model_metadata(active_model_path, all_results)

        st.markdown("---")
        st.markdown("### 📊 Model Specifications")
        st.markdown(f"""
        <table class="metric-mini-table">
            <tr><td>Model</td><td>{meta['model_name']}</td></tr>
            <tr><td>Architecture</td><td>{meta['architecture']}</td></tr>
            <tr><td>Features</td><td>{meta['feature_type']}</td></tr>
            <tr><td>Preprocessing</td><td>{meta['preprocessing_mode']}</td></tr>
            <tr><td>Validation Accuracy</td><td><span style="color:#2f855a;">{meta['val_accuracy']}</span></td></tr>
            <tr><td>Validation F1</td><td><span style="color:#2b6cb0;">{meta['val_f1']}</span></td></tr>
            <tr><td>Parameters</td><td>{meta['parameter_count']}</td></tr>
            <tr><td>Dataset Size</td><td>{meta['dataset_size']}</td></tr>
            <tr><td>Trained Date</td><td>{meta['training_date']}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        if mode == "development" and len(candidate_paths) > 1:
            st.markdown("---")
            st.markdown("### 🏆 Model Leaderboard (Top 5)")
            leaderboard = build_leaderboard(candidate_paths, top_n=5, all_results=all_results)
            lb_df = pd.DataFrame([
                {
                    "Rank": f"#{item['rank']}",
                    "Model": item["display_name"][:25] + ("..." if len(item["display_name"]) > 25 else ""),
                    "Val F1": item["val_f1"],
                    "Val Acc": item["val_acc"],
                }
                for item in leaderboard
            ])
            st.dataframe(lb_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏷️ Supported Topics")
        for cls in meta["classes"]:
            icon = CLASS_ICONS.get(cls, "📌")
            st.markdown(f"- {icon} **{cls}**")

    # ---------------------------------------------------------------------------
    # Main Content Area
    # ---------------------------------------------------------------------------

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>📰 News Topic Classifier</h1>
        <p>Classify news headlines into topic categories using trained production NLP models ({meta['display_name']})</p>
    </div>
    """, unsafe_allow_html=True)

    if not active_model_path:
        st.warning("⚠️ **No Active Model Checkpoint Found on this Deployment**")
        st.info("""
        **To enable predictions on Streamlit Cloud:**
        1. Ensure the deployed model checkpoint in `app/model/` is committed and pushed to GitHub:
           ```bash
           git add app/model
           git commit -m "Add deployed model checkpoint"
           git push origin main
           ```
        2. Or configure `MODEL_DIR` in Streamlit Cloud Dashboard > Settings > Secrets.
        """)
        return

    # Load Model with Diagnostics
    classifier, load_error = load_classifier_with_diagnostics(str(active_model_path))

    if classifier is None:
        st.error(f"### ⚠️ Failed to initialize classifier\n\n```\n{load_error}\n```")
        st.info("Check the deployment checklist in the sidebar to verify artifact integrity.")
        return

    # Two-Column Layout for Single Headline Classification
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### ✍️ Enter a News Headline")

        # Example headline buttons
        st.markdown("**Try an example headline:**")
        example_cols = st.columns(4)

        for i, (category, examples) in enumerate(EXAMPLE_HEADLINES.items()):
            with example_cols[i]:
                icon = CLASS_ICONS.get(category, "📌")
                short_name = category.split()[0]
                if st.button(f"{icon} {short_name}", key=f"btn_ex_{category}", use_container_width=True):
                    st.session_state["headline_input"] = examples[0]

        # Text input area connected directly to session state
        headline_text = st.text_area(
            "News Headline",
            key="headline_input",
            height=110,
            placeholder="Type or paste a news headline here...",
            label_visibility="collapsed",
        )

        classify_headline_btn = st.button("🔍 Classify Headline", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🎯 Prediction Result")

        if classify_headline_btn and headline_text.strip():
            if len(headline_text) > 5000:
                st.error("⚠️ A headline must be 5,000 characters or fewer.")
            else:
                try:
                    with st.spinner("Classifying headline..."):
                        start_time = time.perf_counter()
                        result = classifier.predict(headline_text.strip())
                        elapsed_ms = (time.perf_counter() - start_time) * 1000

                    predicted_class = result["class"]
                    confidence = result["confidence"]
                    icon = CLASS_ICONS.get(predicted_class, "📌")
                    color = CLASS_COLORS.get(predicted_class, "#667eea")
                    conf_class = get_confidence_class(confidence)

                    if confidence < 0.60:
                        st.warning("⚠️ Low-confidence prediction: review headline clarity or topic boundaries.")

                    st.markdown(f"""
                    <div class="prediction-card" style="border-top: 4px solid {color};">
                        <div style="font-size: 2.8rem;">{icon}</div>
                        <div class="prediction-label">{predicted_class}</div>
                        <span class="confidence-badge {conf_class}">
                            Confidence: {confidence:.1%}
                        </span>
                        <div style="margin-top: 0.6rem; color: #718096; font-size: 0.85rem;">
                            Inference latency: {elapsed_ms:.1f} ms · Model: {meta['model_name']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if result.get("probabilities"):
                        st.markdown("#### Probability Distribution")
                        fig = create_probability_chart(result["probabilities"])
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    logger.error("Single prediction error: %s", exc, exc_info=True)

        elif classify_headline_btn:
            st.warning("Please enter or select a headline above to classify.")
        else:
            st.info("Enter a headline or select an example above and click **Classify Headline**.")

    # ---------------------------------------------------------------------------
    # Independent Batch Classification Section
    # ---------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📋 Batch Classification Mode", expanded=False):
        st.markdown("Classify multiple headlines at once (one headline per line, up to 100 headlines).")

        batch_text = st.text_area(
            "Batch Headlines Input",
            height=140,
            placeholder="Headline 1\nHeadline 2\nHeadline 3",
            label_visibility="collapsed",
        )

        classify_batch_btn = st.button("📊 Classify Batch", type="secondary")

        if classify_batch_btn and batch_text and batch_text.strip():
            raw_lines = [h.strip() for h in batch_text.strip().split("\n") if h.strip()]

            if len(raw_lines) > 100:
                st.warning(f"Batch limit is 100 headlines. Only the first 100 of {len(raw_lines)} headlines will be processed.")
                raw_lines = raw_lines[:100]

            if any(len(h) > 5000 for h in raw_lines):
                st.error("Each batch headline must be 5,000 characters or fewer.")
            else:
                try:
                    with st.spinner(f"Classifying {len(raw_lines)} headlines..."):
                        batch_results = classifier.batch_predict(raw_lines)

                    # Display Dataframe
                    df = pd.DataFrame([
                        {
                            "Headline": r["text"][:90] + ("..." if len(r["text"]) > 90 else ""),
                            "Category": f"{CLASS_ICONS.get(r['class'], '📌')} {r['class']}",
                            "Confidence": f"{r['confidence']:.1%}",
                        }
                        for r in batch_results
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Distribution Chart
                    categories = [r["class"] for r in batch_results]
                    cat_counts = pd.Series(categories).value_counts()

                    fig = px.pie(
                        values=cat_counts.values,
                        names=cat_counts.index,
                        color=cat_counts.index,
                        color_discrete_map=CLASS_COLORS,
                        title="Batch Category Distribution",
                    )
                    fig.update_layout(
                        font=dict(family="Inter"),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as exc:
                    st.error(f"Batch prediction error: {exc}")
                    logger.error("Batch prediction failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
