import streamlit as st
import tensorflow as tf
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from configs.config import MODEL_PATH, DECISION_THRESHOLD
from src.preprocessing import preprocess_for_inference

st.set_page_config(page_title="Tuberculosis Screening Triage", page_icon="🫁")

@st.cache_resource
def load_triage_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}.")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH))

model = load_triage_model()

st.title("🫁 Chest X-Ray Tuberculosis Triage System")
st.warning(
    "**CLINICAL RESEARCH DISCLAIMER**: This application is a research-grade triage screening tool "
    "and **does NOT provide a medical diagnosis**. Predicted probabilities reflect statistical pattern "
    "matching from an academic cohort ($N=800$) and require microbiological confirmation."
)

uploaded_file = st.file_uploader("Upload Posteroanterior (PA) Chest Radiograph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Radiograph", use_container_width=True)

    with st.spinner("Analyzing radiograph feature maps..."):
        try:
            tensor = preprocess_for_inference(bytes_data)
            prediction = float(model.predict(tensor)[0][0])
            is_tb = prediction >= DECISION_THRESHOLD
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    with col2:
        st.subheader("Triage Assessment")
        if is_tb:
            st.error("🚨 **Presumptive Finding: Tuberculosis (TB)**")
        else:
            st.success("✅ **Presumptive Finding: Normal**")

        st.metric(label="Model Confidence Score", value=f"{prediction * 100:.1f}%")
        st.caption(f"Operating Decision Threshold: $\\tau = {DECISION_THRESHOLD:.2f}$")
