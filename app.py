"""
STEP 2 — Launch the web interface
Run:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from Fabric_utils import CLASSES, get_real_dataset_sample, preprocess_for_model

MODEL_PATH = "fabric_model.h5"

st.set_page_config(page_title="Fabric Defect Detector", page_icon="🧵", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #f4f8ff 0%, #eefaf6 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .top-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid #dfeaf7;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }
        .result-box {
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #ffffff;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.05);
        }
        .result-safe {
            background: linear-gradient(135deg, #e9fff7 0%, #dffaf0 100%);
            border-color: #8fe2b8;
        }
        .result-risk {
            background: linear-gradient(135deg, #fff0f1 0%, #ffe7ea 100%);
            border-color: #f4b9c1;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #1f2937;
        }
        .small-note {
            color: #475569;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-card">
        <h1 style="margin:0; font-size:2.2rem;">🧵 Fabric Defect Detector</h1>
        <p class="small-note" style="margin:0.5rem 0 0 0;">AI-powered fabric inspection for rapid visual defect detection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Fabric inspection")
st.sidebar.caption("Model classes: " + ", ".join(CLASSES))
st.sidebar.markdown("This tool checks cloth images against the five categories learned from the labeled dataset.")


@st.cache_resource
def get_model():
    model = load_model(MODEL_PATH)
    if model.output_shape[-1] != len(CLASSES):
        raise ValueError(f"Expected {len(CLASSES)} outputs but model has {model.output_shape[-1]}")
    return model


try:
    model = get_model()
except (OSError, IOError, ValueError):
    st.error("Model file 'fabric_model.h5' was not found or is incompatible. Run `python train_model.py` first.")
    st.stop()

if model.output_shape[-1] != len(CLASSES):
    st.error(f"Model mismatch: this app expects {len(CLASSES)} output classes, but the loaded model has {model.output_shape[-1]}. Please retrain with `python train_model.py`.")
    st.stop()

# ---------- Input ----------
tab_upload, tab_demo = st.tabs(["📤 Upload Image", "📷 Real Dataset Sample"])

image, true_label = None, None

with tab_upload:
    uploaded = st.file_uploader("Drop a fabric image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read that image file.")
        else:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

with tab_demo:
    col_demo_1, col_demo_2 = st.columns([2, 1])
    with col_demo_1:
        if st.button("📷 Load a random real sample", use_container_width=True):
            cls = np.random.choice(CLASSES)
            try:
                image, true_label = get_real_dataset_sample(class_name=cls)
                st.session_state["demo_img"] = image
                st.session_state["demo_label"] = true_label
            except FileNotFoundError:
                st.error("No real dataset sample found. Please ensure the dataset folder exists.")
    with col_demo_2:
        st.caption("Real-data demo")
    if "demo_img" in st.session_state:
        image = st.session_state["demo_img"]
        true_label = st.session_state["demo_label"]

# ---------- Prediction ----------
if image is not None:
    col_img, col_result = st.columns([1.1, 0.9])

    with col_img:
        st.image(image, caption="Fabric under inspection", use_container_width=True)

    preds = model.predict(preprocess_for_model(image), verbose=0)[0]
    idx = int(np.argmax(preds))
    result = CLASSES[idx]
    confidence = float(preds[idx])
    ranked_indices = np.argsort(preds)[::-1]
    runner_up = CLASSES[int(ranked_indices[1])]
    runner_up_confidence = float(preds[ranked_indices[1]])

    with col_result:
        if result == "Good":
            st.markdown(
                f"""
                <div class="result-box result-safe">
                    <h3 style="margin:0; color:#0f766e;">✅ {result}</h3>
                    <p style="margin:0.5rem 0 0 0;">No defect detected — fabric passed inspection.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box result-risk">
                    <h3 style="margin:0; color:#b42318;">⚠️ {result}</h3>
                    <p style="margin:0.5rem 0 0 0;">Defect found — manual review recommended.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-title'>Prediction summary</div>", unsafe_allow_html=True)
        st.metric("Predicted class", result, delta=f"{confidence * 100:.1f}% confidence")

        if confidence < 0.60 or confidence - runner_up_confidence < 0.15:
            st.warning(
                f"Low-confidence result. Compare {result} ({confidence:.0%}) "
                f"with {runner_up} ({runner_up_confidence:.0%}) and review manually."
            )

        st.markdown("<div class='section-title'>Class probabilities</div>", unsafe_allow_html=True)
        for i, cls in enumerate(CLASSES):
            st.progress(float(preds[i]), text=f"{cls}: {preds[i]:.0%}")

    if true_label is not None:
        verdict = "agreed ✅" if true_label == result else "disagreed ❌"
        st.info(f"🎲 Demo ground truth: **{true_label}** — model {verdict}")
else:
    st.info("👆 Upload an image or load a real dataset sample to get started.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick guide")
st.sidebar.write("1. Upload a real fabric image or load a real dataset sample.")
st.sidebar.write("2. Check the predicted defect class and confidence.")
st.sidebar.write("3. Send samples with defects for manual review.")