from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from modules.data_processing import MODEL_FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_model.joblib"

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found at models/trained_model.joblib. Export the fitted decision_tree pipeline from the original notebook."
        )
    return joblib.load(MODEL_PATH)

def _ensure_features(frame):
    missing = [c for c in MODEL_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    return frame[MODEL_FEATURES].copy()

def predict_request(model, request_data):
    frame = pd.DataFrame([request_data], columns=MODEL_FEATURES)
    prediction = model.predict(frame)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probs[classes.index(1)])
        elif "1" in classes:
            probability = float(probs[classes.index("1")])
    return prediction, probability

def predict_batch(model, df):
    features = _ensure_features(df)
    result = df.copy()
    result["Predicted_SLA_Breached"] = model.predict(features)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        classes = list(model.classes_)
        if 1 in classes:
            result["Estimated_Breach_Probability"] = probs[:, classes.index(1)]
        elif "1" in classes:
            result["Estimated_Breach_Probability"] = probs[:, classes.index("1")]
    result["Review_Recommendation"] = result["Predicted_SLA_Breached"].apply(
        lambda x: "Staff review recommended" if str(x) in ("1", "1.0", "True") else "Normal monitoring"
    )
    return result
