import streamlit as st
import tensorflow as tf
from pathlib import Path
import sys
import time

# Ensure the app can find the src and configs directories
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from configs.config import MODEL_PATH, DECISION_THRESHOLD
from src.preprocessing import preprocess_for_inference

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="PneumoScreen | TB Triage",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS to remove generic Streamlit branding and improve typography
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .stAlert > div {padding-top: 0.5rem; padding-bottom: 0.5rem;}
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading with Caching
@st.cache_resource(show_spinner=False)
def load_triage_model():
    if not MODEL_PATH.exists():
        st.error(f"System Error: Model artifact missing at {MODEL_PATH}.")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH))

model = load_triage_model()

# 4. Header Section
st.title("🩺 PneumoScreen: TB Triage System")
st.markdown(
    "A research-grade convolutional neural network pipeline for the rapid triage of "
    "Posteroanterior (PA) chest radiographs."
)

# 5. Core Interaction Area
uploaded_file = st.file_uploader(
    "Securely upload a PA chest radiograph (PNG/JPG)", 
    type=["png", "jpg", "jpeg"],
    help="Images are processed entirely in memory and are not saved to any database."
)

if uploaded_file is not None:
    # Read bytes once
    bytes_data = uploaded_file.read()
    
    # Create a clean two-column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("#### Input Radiograph")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.markdown("#### Triage Assessment")
        
        # Simulated processing delay for UX (prevents jarring instantaneous flashes)
        with st.spinner("Executing forward pass..."):
            start_time = time.time()
            try:
                tensor = preprocess_for_inference(bytes_data)
                raw_prediction = float(model.predict(tensor, verbose=0)[0][0])
                is_tb = raw_prediction >= DECISION_THRESHOLD
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.stop()
            inference_time = time.time() - start_time

        # Primary Abstraction: Clear, binary clinical outcome
        if is_tb:
            st.error("🚨 **Presumptive Finding: Tuberculosis**\n\nHigh probability of spatial features correlating with TB pathology.")
        else:
            st.success("✅ **Presumptive Finding: Normal**\n\nNo significant spatial features correlating with TB pathology detected.")
            
        # Progressive Disclosure: Technical details hidden by default
        with st.expander("🔬 Technical & Clinical Metrics", expanded=False):
            st.markdown(f"**Calculated Probability:** `{raw_prediction:.4f}`")
            st.markdown(f"**Decision Threshold:** `τ = {DECISION_THRESHOLD}`")
            st.markdown(f"**Inference Latency:** `{inference_time * 1000:.1f} ms`")
            st.divider()
            st.caption(
                "**Architecture:** Custom 4-Block CNN (6.81M params)\n\n"
                "**Disclaimer:** This is a statistical pattern matching tool trained on an academic dataset (N=800). "
                "It does not constitute a medical diagnosis. Microbiological confirmation is required."
            )
