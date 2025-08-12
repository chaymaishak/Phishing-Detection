import os
import streamlit as st
import pandas as pd
from src.predict import load_artifacts, predict_texts

st.set_page_config(page_title="Phishing Detection", layout="wide")
st.title("📧 Phishing Email Detection & Analysis")

MODEL_PATH = "model.joblib"
VEC_PATH = "vectorizer.joblib"

col1, col2 = st.columns([2,1])

with col1:
    st.header("Predict a single email")
    text = st.text_area("Paste email text here", height=180)
    if st.button("Predict single"):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
            st.error("Model not found. Please run training: python src/train.py")
        else:
            model, vec = load_artifacts(MODEL_PATH,VEC_PATH)
            preds, probs = predict_texts([text], model, vec)
            label = "PHISHING" if preds[0]==1 else "LEGITIMATE"
            st.write(f"**Prediction:** {label}")
            st.write(f"**Phishing probability:** {probs[0]:.2f}")

with col2:
    st.header("Batch predict (CSV)")
    uploaded = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
                st.error("Model not found. Please run training: python src/train.py")
            else:
                model, vec = load_artifacts(MODEL_PATH,VEC_PATH)
                texts = df['text'].astype(str).tolist()
                preds, probs = predict_texts(texts, model, vec)
                df['phishing_prob'] = probs
                df['prediction'] = ['phishing' if p==1 else 'legitimate' for p in preds]
                st.dataframe(df.head(200))
                st.download_button("Download predictions CSV", df.to_csv(index=False).encode('utf-8'), "predictions.csv")

st.markdown("---")
st.markdown("**Notes:** Train model first by running `python src/train.py` (or modify to point to a larger dataset).")
