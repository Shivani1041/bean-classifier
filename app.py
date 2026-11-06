import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

from utils import TARGET, MODEL_FILES, MODEL_DIR, compute_metrics

st.set_page_config(page_title="Bean Lab | Dry Bean Classifier", page_icon="🫘",
                   layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_artifacts():
    return (joblib.load(f"{MODEL_DIR}/label_encoder.joblib"),
            joblib.load(f"{MODEL_DIR}/feature_names.joblib"))


def show_table(data):
    """Draw a table. st.dataframe needs pyarrow; if the OS blocks that DLL
    (seen on locked-down Windows laptops) fall back to a plain HTML table."""
    try:
        st.dataframe(data)
    except ImportError:
        st.markdown(data.to_html(), unsafe_allow_html=True)


encoder, FEATURES = load_artifacts()
CLASSES = list(encoder.classes_)


def evaluate(model_name, X, y_enc):
    """Load a SAVED model and score it on the uploaded data (no retraining)."""
    model = load_model(MODEL_FILES[model_name])
    pred, proba = model.predict(X), model.predict_proba(X)
    return pred, proba, compute_metrics(y_enc, pred, proba, len(CLASSES))


# ------------------------------------------------------------------ header
st.title("🫘 Bean Lab")
st.caption("Classifying 7 varieties of dry beans from 16 shape measurements. "
           "Upload a test CSV, pick a model, and inspect how it performs.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Upload test data")
    uploaded = st.file_uploader("CSV with the 16 features + 'Class' column", type="csv")
    with open("test_data.csv", "rb") as f:
        st.download_button("⬇ Download sample test_data.csv", f, "test_data.csv", "text/csv")

    st.header("2 · Choose a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES))

    st.markdown("---")
    st.caption("Models are pre-trained and loaded from the `model/` folder. "
               "Nothing is retrained on your upload.")

tab_eval, tab_compare, tab_about = st.tabs(
    ["📊 Evaluate a model", "🏁 Compare all models", "ℹ️ About the data"])

# ------------------------------------------------------------------ data check
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    unknown = sorted(set(df[TARGET].astype(str)) - set(CLASSES)) if TARGET in df.columns else []
    if missing:
        st.error(f"Your CSV is missing these columns: {missing}")
        df = None
    elif unknown:
        st.error(f"Unknown class labels in '{TARGET}': {unknown}. Expected: {CLASSES}")
        df = None

X = y_enc = None
if df is not None:
    X = df[FEATURES]
    y_enc = encoder.transform(df[TARGET].astype(str))

# ------------------------------------------------------------------ tab 1
with tab_eval:
    if X is None:
        st.info("👈 Upload a test CSV in the sidebar to begin "
                "(you can download a ready-made one from the sidebar).")
    else:
        st.subheader(f"{model_name} · {len(df):,} beans")
        pred, proba, m = evaluate(model_name, X, y_enc)

        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("**Confusion matrix**")
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(confusion_matrix(y_enc, pred, labels=range(len(CLASSES))),
                                   display_labels=CLASSES).plot(
                ax=ax, cmap="YlGn", xticks_rotation=45, colorbar=False)
            fig.tight_layout()
            st.pyplot(fig)
        with right:
            st.markdown("**Classification report**")
            rep = pd.DataFrame(classification_report(
                y_enc, pred, labels=range(len(CLASSES)), target_names=CLASSES,
                output_dict=True, zero_division=0)).T.round(3)
            show_table(rep)

        with st.expander("Peek at the predictions"):
            out = df.copy()
            out["Predicted"] = encoder.inverse_transform(pred)
            out["Correct"] = out["Predicted"] == out[TARGET]
            show_table(out[[TARGET, "Predicted", "Correct"] + FEATURES[:4]].head(200))

# ------------------------------------------------------------------ tab 2
with tab_compare:
    if X is None:
        st.info("Upload a test CSV to compare all five models on the same data.")
    else:
        rows = {name: evaluate(name, X, y_enc)[2] for name in MODEL_FILES}
        comp = pd.DataFrame(rows).T.round(4)
        comp.index.name = "ML Model Name"
        show_table(comp.style.highlight_max(axis=0, color="#c8e6c9"))

        metric = st.radio("Chart metric", list(comp.columns), index=0, horizontal=True)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.barh(comp.index, comp[metric], color="#66a36c")
        ax.set_xlim(max(0, comp[metric].min() - 0.05), 1.0)
        ax.set_xlabel(metric)
        for i, v in enumerate(comp[metric]):
            ax.text(v, i, f" {v:.3f}", va="center")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------------------ tab 3
with tab_about:
    st.markdown(f"""
**Dataset:** Dry Bean Dataset, UCI Machine Learning Repository (Koklu & Özkan, 2020)
· 13,611 beans · 16 features · 7 classes

**Classes:** {", ".join(CLASSES)}

**Features:** {", ".join(FEATURES)}

**How it works:** each bean is described by size and shape measurements extracted from
images (area, perimeter, axis lengths, roundness, shape factors...). Five classifiers were
trained on 75% of the data; the remaining 25% was saved as `test_data.csv`.
""")
    try:
        st.markdown("**Test-set results recorded at training time**")
        show_table(pd.read_csv(f"{MODEL_DIR}/results.csv", index_col=0))
    except FileNotFoundError:
        pass