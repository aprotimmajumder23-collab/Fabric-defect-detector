"""STEP 2 — Launch the web interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from fabric_utils import (
    ARTIFACTS_DIR,
    CLASSES,
    GOOD_CLASS,
    PATCH_SIZE,
    colourise_heatmap,
    decode_image_bytes,
    draw_flagged_tiles,
    find_model_path,
    get_real_dataset_sample,
    grad_cam,
    inspect_image,
    list_class_files,
    load_detector,
    predict_probabilities,
    summarise_prediction,
)

DATA_DIR = "dataset"
ACCEPTED_TYPES = ["jpg", "jpeg", "png", "bmp"]

st.set_page_config(page_title="Fabric Defect Detector", page_icon="🧵", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .top-card {
            background: linear-gradient(135deg, #f4f8ff 0%, #eefaf6 100%);
            border: 1px solid #dfeaf7; border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
        }
        .result-box { border-radius: 14px; padding: 1rem 1.2rem; border: 1px solid rgba(15, 23, 42, 0.08); margin-bottom: 0.8rem; }
        .result-safe { background: linear-gradient(135deg, #e9fff7 0%, #dffaf0 100%); border-color: #8fe2b8; }
        .result-risk { background: linear-gradient(135deg, #fff0f1 0%, #ffe7ea 100%); border-color: #f4b9c1; }
        .small-note { color: #475569; font-size: 0.9rem; }
    </style>
    <div class="top-card">
        <h1 style="margin:0; font-size:2rem;">🧵 Fabric Defect Detector</h1>
        <p class="small-note" style="margin:0.4rem 0 0 0;">Patch-level fabric inspection: Good · Hole · Objects · Oil Spot · Thread Error</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model + metadata
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_detector()


@st.cache_data
def get_metrics() -> dict | None:
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def dataset_available() -> bool:
    return any(list_class_files(DATA_DIR, cls) for cls in CLASSES)


try:
    model = get_model()
except (OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

metrics = get_metrics()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("Inspection settings")
    threshold = st.slider(
        "Defect sensitivity",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="A patch is flagged when its total defect probability reaches this value. Lower = catch more defects (more false alarms).",
    )
    mode = st.radio(
        "Inspection mode",
        ["Auto", "Whole image", "Tiled scan"],
        help="The model was trained on 64×64 patches. Large images are scanned tile-by-tile to localise defects; 'Auto' tiles anything bigger than 2 patches.",
    )
    tile_size = st.slider("Tile size (px)", 32, 256, PATCH_SIZE, step=16, help="Match this to the size a 64 px dataset patch would have in your photo.")
    overlap = st.slider("Tile overlap", 0.0, 0.75, 0.5, step=0.25, help="Overlapping tiles give a smoother heatmap at the cost of more inference.")
    use_tta = st.toggle(
        "Test-time augmentation",
        value=False,
        help="Average predictions over 8 flips/rotations of each patch. More stable, ~8× slower.",
    )
    show_cam = st.toggle("Show Grad-CAM explanation", value=True, help="Highlights the regions that drove the whole-image prediction.")

    st.markdown("---")
    st.markdown(f"**Model:** `{find_model_path()}`")
    if metrics:
        test = metrics.get("test", {})
        dd = test.get("defect_detection", {})
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{test.get('accuracy', 0):.1%}")
        c2.metric("Macro F1", f"{test.get('macro_f1', 0):.2f}")
        c1.metric("Defect recall", f"{dd.get('recall', 0):.1%}")
        c2.metric("False-alarm rate", f"{dd.get('false_alarm_rate', 0):.1%}")
        st.caption(f"Held-out source images, defect sensitivity {dd.get('threshold', 0.5):.2f}. ROC-AUC {dd.get('roc_auc', float('nan')):.2f}.")
    else:
        st.caption("Run `python train_model.py` to generate metrics.")

    st.markdown("---")
    st.markdown("### Quick guide")
    st.write("1. Upload a fabric photo (or a batch) or load a labelled sample.")
    st.write("2. Review the verdict, heatmap and flagged tiles.")
    st.write("3. Tune sensitivity to trade missed defects against false alarms.")

stride = max(1, int(round(tile_size * (1 - overlap))))


def should_tile(img: np.ndarray) -> bool:
    if mode == "Whole image":
        return False
    if mode == "Tiled scan":
        return True
    return min(img.shape[:2]) >= 2 * tile_size


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def verdict_box(label: str, is_defective: bool, subtitle: str) -> None:
    css, icon, colour = ("result-risk", "⚠️", "#b42318") if is_defective else ("result-safe", "✅", "#0f766e")
    st.markdown(
        f"""<div class="result-box {css}"><h3 style="margin:0; color:{colour};">{icon} {label}</h3>
        <p style="margin:0.4rem 0 0 0;">{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def probability_bars(probs: dict[str, float]) -> None:
    for cls in CLASSES:
        st.progress(float(min(max(probs[cls], 0.0), 1.0)), text=f"{cls}: {probs[cls]:.0%}")


def render_whole_image(image: np.ndarray) -> dict:
    probs = predict_probabilities(model, [image], tta=use_tta)[0]
    summary = summarise_prediction(probs, threshold)
    col_img, col_res = st.columns([1.1, 0.9])

    with col_img:
        if show_cam:
            try:
                cam = grad_cam(model, image, CLASSES.index(summary["top_class"]))
                st.image(colourise_heatmap(image, cam), caption=f"Grad-CAM for '{summary['top_class']}'", use_container_width=True)
            except (ValueError, tf.errors.OpError) as exc:
                st.image(image, caption="Fabric under inspection", use_container_width=True)
                st.caption(f"Grad-CAM unavailable: {exc}")
        else:
            st.image(image, caption="Fabric under inspection", use_container_width=True)

    with col_res:
        if summary["is_defective"]:
            verdict_box(summary["label"], True, f"Defect probability {summary['defect_score']:.0%} — manual review recommended.")
        else:
            verdict_box(GOOD_CLASS, False, f"Defect probability {summary['defect_score']:.0%} — below the {threshold:.0%} threshold.")
        st.metric("Predicted class", summary["label"], delta=f"{summary['confidence'] * 100:.1f}% confidence")
        if summary["confidence"] < 0.60 or summary["confidence"] - summary["runner_up_confidence"] < 0.15:
            st.warning(
                f"Low-margin result: {summary['label']} ({summary['confidence']:.0%}) vs {summary['runner_up']} ({summary['runner_up_confidence']:.0%})."
            )
        st.markdown("**Class probabilities**")
        probability_bars(summary["probabilities"])
    return summary


def render_tiled(image: np.ndarray) -> dict:
    with st.spinner("Scanning tiles…"):
        result = inspect_image(model, image, tile_size=tile_size, stride=stride, threshold=threshold, tta=use_tta)

    col_img, col_res = st.columns([1.3, 0.7])
    with col_img:
        tab_heat, tab_boxes, tab_raw = st.tabs(["🔥 Defect heatmap", "🟥 Flagged tiles", "🖼️ Original"])
        tab_heat.image(colourise_heatmap(image, result["heatmap"]), use_container_width=True, caption="Red = high defect probability")
        tab_boxes.image(
            draw_flagged_tiles(image, result["flagged"]), use_container_width=True, caption=f"{len(result['flagged'])} of {result['n_tiles']} tiles flagged"
        )
        tab_raw.image(image, use_container_width=True)

    with col_res:
        if result["is_defective"]:
            verdict_box(result["label"], True, f"{len(result['flagged'])} tile(s) flagged — {result['defect_area_fraction']:.0%} of the surface.")
        else:
            verdict_box(GOOD_CLASS, False, f"No tile reached the {threshold:.0%} defect threshold.")
        c1, c2 = st.columns(2)
        c1.metric("Tiles scanned", result["n_tiles"])
        c2.metric("Max defect score", f"{result['max_defect_score']:.0%}")
        st.markdown("**Defect types among flagged tiles**")
        votes = result["class_votes"]
        total = max(sum(votes.values()), 1)
        for cls, n in votes.items():
            st.progress(n / total if result["flagged"] else 0.0, text=f"{cls}: {n} tile(s)")

    if result["flagged"]:
        rows = [
            {
                "x": f["box"][0],
                "y": f["box"][1],
                "size": f["box"][2] - f["box"][0],
                "defect": f["label"],
                "confidence": round(f["confidence"], 3),
                "defect score": round(f["defect_score"], 3),
            }
            for f in sorted(result["flagged"], key=lambda f: -f["defect_score"])
        ]
        with st.expander(f"Flagged tile details ({len(rows)})"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    return result


def render_image(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    if should_tile(image):
        st.caption(f"Tiled scan · {w}×{h} px · tile {tile_size} px, stride {stride} px")
        return render_tiled(image)
    st.caption(f"Whole-image classification · {w}×{h} px (resized to the model input)")
    return render_whole_image(image)


def read_upload(uploaded) -> np.ndarray | None:
    image = decode_image_bytes(uploaded.getvalue())
    if image is None:
        st.error(f"Could not read '{uploaded.name}'.")
    return image


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_single, tab_batch, tab_demo = st.tabs(["📤 Inspect image", "🗂️ Batch inspection", "📷 Labelled sample"])

with tab_single:
    uploaded = st.file_uploader("Drop a fabric image here", type=ACCEPTED_TYPES, key="single_upload", label_visibility="collapsed")
    if uploaded:
        image = read_upload(uploaded)
        if image is not None:
            render_image(image)
    else:
        st.info("👆 Upload a fabric photo. Small crops are classified directly; larger photos are scanned tile-by-tile.")

with tab_batch:
    files = st.file_uploader("Upload several fabric images", type=ACCEPTED_TYPES, accept_multiple_files=True, key="batch_upload")
    if files:
        rows = []
        progress = st.progress(0.0, text="Inspecting…")
        for i, f in enumerate(files):
            image = decode_image_bytes(f.getvalue())
            if image is None:
                rows.append({"file": f.name, "verdict": "unreadable"})
            elif should_tile(image):
                r = inspect_image(model, image, tile_size=tile_size, stride=stride, threshold=threshold, tta=use_tta)
                rows.append(
                    {
                        "file": f.name,
                        "verdict": "DEFECT" if r["is_defective"] else "OK",
                        "class": r["label"],
                        "defect score": round(r["max_defect_score"], 3),
                        "flagged tiles": len(r["flagged"]),
                        "tiles": r["n_tiles"],
                        "size": f"{image.shape[1]}×{image.shape[0]}",
                    }
                )
            else:
                s = summarise_prediction(predict_probabilities(model, [image], tta=use_tta)[0], threshold)
                rows.append(
                    {
                        "file": f.name,
                        "verdict": "DEFECT" if s["is_defective"] else "OK",
                        "class": s["label"],
                        "defect score": round(s["defect_score"], 3),
                        "flagged tiles": int(s["is_defective"]),
                        "tiles": 1,
                        "size": f"{image.shape[1]}×{image.shape[0]}",
                    }
                )
            progress.progress((i + 1) / len(files), text=f"Inspected {i + 1}/{len(files)}")
        progress.empty()

        df = pd.DataFrame(rows)
        n_defect = int((df["verdict"] == "DEFECT").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", len(df))
        c2.metric("Defective", n_defect)
        c3.metric("Pass rate", f"{(len(df) - n_defect) / len(df):.0%}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        st.download_button("⬇️ Download CSV report", buffer.getvalue(), file_name="fabric_inspection_report.csv", mime="text/csv")
    else:
        st.info("Upload multiple images to get a pass/fail table and a downloadable CSV report.")

with tab_demo:
    if not dataset_available():
        st.warning("No labelled dataset found in `dataset/`; the sample browser is disabled.")
    else:
        c1, c2 = st.columns([1, 2])
        choice = c1.selectbox("Class", ["Random", *CLASSES], key="demo_class")
        if c2.button("🎲 Load a random labelled sample", use_container_width=True):
            try:
                img, label, path = get_real_dataset_sample(None if choice == "Random" else choice, DATA_DIR, random.Random())
                st.session_state["demo"] = {"image": img, "label": label, "path": path}
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
        demo = st.session_state.get("demo")
        if demo:
            st.caption(f"Sample: `{demo['path']}`")
            result = render_image(demo["image"])
            predicted = result["label"]
            agreed = predicted == demo["label"]
            (st.success if agreed else st.error)(
                f"Ground truth: **{demo['label']}** — model predicted **{predicted}** ({'agreed ✅' if agreed else 'disagreed ❌'})"
            )
        else:
            st.info("Pick a class (or Random) and load a sample to see how the model performs on labelled data.")
