# (Paste the complete code block above into this cell)import streamlit as st
import tensorflow as tf
from pathlib import Path
import sys
import time

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from configs.config import MODEL_PATH, DECISION_THRESHOLD
from src.preprocessing import preprocess_for_inference

st.set_page_config(
    page_title="Tuberculosis Detection from Chest X-Ray",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- BACKGROUND STYLING CODE START ---
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                          url("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)
# --- BACKGROUND STYLING CODE END ---

@st.cache_resource(show_spinner=False)
def load_triage_model():
    if not MODEL_PATH.exists():
        st.error(f"System Error: Model artifact missing at {MODEL_PATH}.")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False)

model = load_triage_model()

st.title("🩺VISION-TB: Visual Intelligence System for Identifying Onset of Tuberculosis.")
st.markdown(
    "A custom Convolutional Neural Network (CNN) built from scratch for binary "
    "classification of Chest X-Ray images into TB or Normal."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (PNG/JPG)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("#### Input Image")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.markdown("#### Model Assessment")
        
        with st.spinner("Running inference..."):
            start_time = time.time()
            try:
                tensor = preprocess_for_inference(bytes_data)
                raw_prediction = float(model.predict(tensor, verbose=0)[0][0])
                is_tb = raw_prediction >= DECISION_THRESHOLD
            except Exception as e:
                st.error(f"Inference Error: {e}")
                st.stop()
            inference_time = time.time() - start_time

        if is_tb:
            st.error("🚨 **Predicted Class: TB**")
        else:
            st.success("✅ **Predicted Class: Normal**")
            
        with st.expander("🔬 Model Confidence & Metrics"):
            st.markdown(f"**Probability Score:** `{raw_prediction:.4f}`")
            st.markdown(f"**Decision Threshold ($\tau$):** `{DECISION_THRESHOLD}`")
            st.markdown(f"**Inference Latency:** `{inference_time * 1000:.1f} ms`")
            st.divider()
            st.caption(
                "**Disclaimer:** This is an academic research project and model output "
                "does not constitute a medical diagnosis."
            )

# (Paste the complete code block above into this cell)
