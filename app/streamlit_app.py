"""
News Topic Classification — Streamlit Web Application

A professional, interactive web interface for classifying news headlines
into topic categories using trained NLP models.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEPLOYED_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEPLOYED_MODEL_NAME = "TF-IDF Logistic Regression"

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }

    .prediction-card:hover {
        transform: translateY(-2px);
    }

    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0.5rem 0;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
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

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border-left: 4px solid;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-size: 1.05rem;
        padding: 1rem;
        transition: border-color 0.2s ease;
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .sidebar .stSelectbox label, .sidebar .stSlider label {
        font-weight: 600;
        color: #2d3748;
    }

    .example-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: #edf2f7;
        color: #4a5568;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .example-chip:hover {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
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

EXAMPLE_HEADLINES = {
    "Business": [
        "Apple reports record quarterly revenue beating Wall Street expectations",
        "Federal Reserve raises interest rates by 25 basis points",
        "Tesla stock surges 15% after earnings beat estimates",
    ],
    "Science and Technology": [
        "NASA's James Webb telescope captures stunning images of distant galaxies",
        "Scientists discover high-temperature superconductor breakthrough",
        "Google DeepMind AI achieves new milestone in protein structure prediction",
    ],
    "Sports": [
        "Lakers defeat Celtics in thrilling overtime championship game",
        "Messi scores hat trick as Argentina wins Copa America",
        "Serena Williams announces retirement from professional tennis",
    ],
    "World News": [
        "UN Security Council holds emergency session on Middle East crisis",
        "European Union reaches historic climate agreement at summit",
        "Major earthquake strikes coastal region causing widespread damage",
    ],
}


@st.cache_resource
def load_classifier(model_path: str):
    """Load the classifier (cached to avoid reloading on every interaction)."""
    try:
        from src.inference import NewsClassifier
        classifier = NewsClassifier.from_checkpoint(model_path)
        return classifier
    except Exception as e:
        return None


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "high-confidence"
    elif confidence >= 0.5:
        return "medium-confidence"
    return "low-confidence"


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create an interactive probability distribution chart."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [f"{CLASS_ICONS.get(c, '📌')} {c}" for c, _ in sorted_probs]
    probs = [p for _, p in sorted_probs]
    colors = [CLASS_COLORS.get(c, "#718096") for c, _ in sorted_probs]

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
# Main App
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📰 News Topic Classifier</h1>
        <p>Classify news headlines into Business, Science & Tech, Sports, or World News using state-of-the-art NLP models</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Deploy one audited TF-IDF checkpoint only.  Do not expose local
        # experiments or Word2Vec-backed models in the public demo.
        saved_models_dir = DEPLOYED_MODEL_DIR.parent
        available_models = [DEPLOYED_MODEL_DIR.name] if all(
            (DEPLOYED_MODEL_DIR / artifact).exists()
            for artifact in (
                "model.joblib",
                "pipeline_config.json",
                "preprocessor.joblib",
                "label_encoder.joblib",
                "vectorizer.joblib",
            )
        ) else []

        if available_models:
            selected_model = st.selectbox(
                "Deployed Model",
                available_models,
                index=0,
                help="This deployment is pinned to its audited TF-IDF model.",
            )
            model_path = str(DEPLOYED_MODEL_DIR)
            st.caption(f"Deployed model: **{DEPLOYED_MODEL_NAME}**")
            card_path = DEPLOYED_MODEL_DIR / "model_card.json"
            if card_path.exists():
                try:
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                    st.caption(f"{card.get('architecture', selected_model)} · {card.get('parameter_count', '—')} parameters · trained {card.get('training_date', 'unknown')}")
                except (OSError, json.JSONDecodeError):
                    st.caption(selected_model)
        else:
            st.warning("⚠️ No trained models found. Run training first:\n```\npython main.py train\n```")
            model_path = None

        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This demo uses a TF-IDF Logistic Regression model trained on 88,000+
        news headlines to predict the topic of any headline.

        **Categories:**
        - 💼 Business
        - 🔬 Science & Technology
        - ⚽ Sports
        - 🌍 World News

        **Models available:**
        - Logistic Regression
        - Deep Neural Network
        - BiLSTM with Attention
        - Transformer Encoder
        """)

    # Main content
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### ✍️ Enter a News Headline")

        # Example buttons
        st.markdown("**Try an example:**")
        example_cols = st.columns(4)
        selected_example = None

        for i, (category, examples) in enumerate(EXAMPLE_HEADLINES.items()):
            with example_cols[i]:
                icon = CLASS_ICONS.get(category, "📌")
                if st.button(f"{icon} {category.split()[0]}", key=f"ex_{category}"):
                    selected_example = examples[0]

        # Text input
        default_text = selected_example or ""
        user_input = st.text_area(
            "Headline",
            value=default_text,
            height=100,
            placeholder="Type or paste a news headline here...",
            label_visibility="collapsed",
        )

        # Batch mode
        with st.expander("📋 Batch Classification"):
            batch_input = st.text_area(
                "Enter multiple headlines (one per line)",
                height=150,
                placeholder="Headline 1\nHeadline 2\nHeadline 3",
            )

        classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🎯 Prediction Result")

        if classify_btn and user_input.strip():
            if len(user_input) > 5000:
                st.error("A headline must be 5,000 characters or fewer.")
                return
            if model_path is None:
                st.error("No model loaded. Please train a model first.")
            else:
                classifier = load_classifier(model_path)
                if classifier is None:
                    st.error("Failed to load model. Check the model path.")
                else:
                    with st.spinner("Classifying..."):
                        start = time.perf_counter()
                        result = classifier.predict(user_input.strip())
                        elapsed = (time.perf_counter() - start) * 1000

                    # Display result
                    predicted_class = result["class"]
                    confidence = result["confidence"]
                    icon = CLASS_ICONS.get(predicted_class, "📌")
                    color = CLASS_COLORS.get(predicted_class, "#718096")
                    conf_class = get_confidence_class(confidence)

                    if confidence < 0.60:
                        st.warning("Low-confidence prediction: review the input or use this result with caution.")

                    st.markdown(f"""
                    <div class="prediction-card" style="border-top: 4px solid {color};">
                        <div style="font-size: 3rem;">{icon}</div>
                        <div class="prediction-label">{predicted_class}</div>
                        <span class="confidence-badge {conf_class}">
                            Confidence: {confidence:.1%}
                        </span>
                        <div style="margin-top: 0.5rem; color: #718096; font-size: 0.85rem;">
                            Inference time: {elapsed:.1f} ms
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability distribution
                    st.markdown("#### Probability Distribution")
                    if result["probabilities"]:
                        fig = create_probability_chart(result["probabilities"])
                        st.plotly_chart(fig, use_container_width=True)

        elif classify_btn:
            st.warning("Please enter a headline to classify.")
        else:
            st.info("Enter a headline and click **Classify** to see the prediction.")

    # Batch results
    if classify_btn and batch_input and batch_input.strip():
        headlines = [h.strip() for h in batch_input.strip().split("\n") if h.strip()]
        if any(len(headline) > 5000 for headline in headlines):
            st.error("Each batch headline must be 5,000 characters or fewer.")
            return
        if headlines and model_path:
            classifier = load_classifier(model_path)
            if classifier:
                st.markdown("---")
                st.markdown("### 📋 Batch Results")

                with st.spinner(f"Classifying {len(headlines)} headlines..."):
                    batch_results = classifier.batch_predict(headlines)

                # Create results DataFrame
                df = pd.DataFrame([
                    {
                        "Headline": r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
                        "Category": f"{CLASS_ICONS.get(r['class'], '📌')} {r['class']}",
                        "Confidence": f"{r['confidence']:.1%}",
                    }
                    for r in batch_results
                ])

                st.dataframe(df, use_container_width=True, hide_index=True)

                # Category distribution
                categories = [r["class"] for r in batch_results]
                cat_counts = pd.Series(categories).value_counts()

                fig = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    color=cat_counts.index,
                    color_discrete_map=CLASS_COLORS,
                    title="Category Distribution",
                )
                fig.update_layout(
                    font=dict(family="Inter"),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
