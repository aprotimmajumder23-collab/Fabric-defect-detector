"""
STEP 2 — Launch the web interface
Run:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from fabric_utils import CLASSES, generate_sample, preprocess_for_model

MODEL_PATH = "fabric_model.h5"

st.set_page_config(page_title="Fabric Defect Detector", page_icon="🧵")
st.title("🧵 Fabric Defect Detector")
st.caption("AI-powered fabric inspection — upload a fabric photo or try a demo sample.")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


try:
    model = get_model()
except (OSError, IOError):
    st.error("Model file 'fabric_model.h5' not found. Run `python train_model.py` first.")
    st.stop()

# ---------- Input ----------
tab_upload, tab_demo = st.tabs(["📤 Upload Image", "🎲 Demo Sample"])

image, true_label = None, None

with tab_upload:
    uploaded = st.file_uploader("Drop a fabric image here", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read that image file.")
        else:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

with tab_demo:
    if st.button("🎲 Generate random fabric sample", use_container_width=True):
        st.session_state["demo_img"], st.session_state["demo_label"] = generate_sample()
    if "demo_img" in st.session_state:
        image = st.session_state["demo_img"]
        true_label = st.session_state["demo_label"]

# ---------- Prediction ----------
if image is not None:
    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image, caption="Fabric under inspection", use_container_width=True)

    preds = model.predict(preprocess_for_model(image), verbose=0)[0]
    idx = int(np.argmax(preds))
    result = CLASSES[idx]

    with col_result:
        if result == "Normal":
            st.success(f"### ✅ {result}\nNo defect detected — fabric passed.")
        else:
            st.error(f"### ⚠️ {result} Detected\nDefect found — send for manual review.")

        st.write("**Model confidence:**")
        for i, cls in enumerate(CLASSES):
            st.progress(float(preds[i]), text=f"{cls}: {preds[i]:.0%}")

    if true_label is not None:
        verdict = "agreed ✅" if true_label == result else "disagreed ❌"
        st.info(f"🎲 Demo ground truth: **{true_label}** — model {verdict}")
else:
    st.info("👆 Upload an image or generate a demo sample to get started.")