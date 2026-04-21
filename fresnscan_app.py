"""
FreshScan — Step 1 + Step 2 Streamlit App
==========================================
Step 1: YOLOv8 detect + count ผลไม้
Step 2: EfficientNetB0 classify สด/เน่า แต่ละผล

Run:
    streamlit run fresnscan_app.py
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreshScan",
    page_icon="🍊",
    layout="wide"
)

st.markdown("""
<style>
    .title { font-size: 2.8rem; font-weight: 800; color: #E07B00; text-align: center; }
    .subtitle { font-size: 1rem; color: #888; text-align: center; margin-bottom: 1.5rem; }
    .count-box {
        border-radius: 12px; padding: 1rem;
        text-align: center; margin-bottom: 0.5rem;
    }
    .count-total  { background: #FFF8F0; border: 2px solid #E07B00; }
    .count-fresh  { background: #F0FFF4; border: 2px solid #38A169; }
    .count-rotten { background: #FFF5F5; border: 2px solid #E53E3E; }
    .count-number-orange { font-size: 2.5rem; font-weight: 800; color: #E07B00; }
    .count-number-green  { font-size: 2.5rem; font-weight: 800; color: #38A169; }
    .count-number-red    { font-size: 2.5rem; font-weight: 800; color: #E53E3E; }
    .count-label { font-size: 0.85rem; color: #666; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍊 FreshScan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Fruit Detection · Counting · Quality Assessment</div>', unsafe_allow_html=True)
st.divider()

# ─── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
# 6 classes ตรงกับ training dataset (Kaggle sriramr)
CLASS_NAMES = [
    "freshapples", "freshbanana", "freshoranges",
    "rottenapples", "rottenbanana", "rottenoranges"
]
FRESH_CLASSES  = {"freshapples", "freshbanana", "freshoranges"}
ROTTEN_CLASSES = {"rottenapples", "rottenbanana", "rottenoranges"}
PADDING     = 8

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

@st.cache_resource
def load_classifier(path):
    try:
        model = tf.keras.models.load_model(path)
        return model, True
    except Exception as e:
        return None, False

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Step 1 — Detection")
    yolo_path = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "runs/detect/best.pt"],
    )
    fruit_class = st.selectbox("Fruit Class", ["orange", "apple", "banana"])
    show_thresh = st.toggle("Advanced: Confidence Threshold", value=False)
    if show_thresh:
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    else:
        conf_thresh = 0.5

    st.divider()
    st.subheader("Step 2 — Classification")
    clf_choice = st.radio(
        "Classifier Model",
        ["EfficientNetB0", "MobileNetV2"],
        help="เลือกโมเดลที่ train มาสำหรับ classify สด/เน่า"
    )
    clf_path = "best_efficientnetb0.h5" if clf_choice == "EfficientNetB0" else "best_mobilenetv2.h5"
    st.caption(f"📂 {clf_path}")

    st.divider()
    st.subheader("Display")
    show_conf   = st.toggle("Show Confidence", value=True)
    box_color_fresh  = st.color_picker("Fresh Color",  "#38A169")
    box_color_rotten = st.color_picker("Rotten Color", "#E53E3E")
    box_color_det    = st.color_picker("Detection Only Color", "#E07B00")

# ─── Load Models ───────────────────────────────────────────────────────────────
yolo_model = load_yolo(yolo_path)
clf_model, use_clf = load_classifier(clf_path)

if use_clf:
    st.sidebar.success("✅ Classifier loaded")
else:
    st.sidebar.warning("⚠️ No classifier — detection only")

# ─── Helper ────────────────────────────────────────────────────────────────────
def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def classify_fruit(crop_bgr):
    """
    classify crop ด้วย model 6 classes
    แล้ว map กลับเป็น Fresh / Rotten
    preprocess ตรงกับที่ train: rescale 1/255 (ไม่ใช้ keras preprocess_input)
    """
    if not use_clf or clf_model is None:
        return None, None

    # resize + normalize ตรงกับ training (ImageDataGenerator rescale=1/255)
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    arr     = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    pred      = clf_model.predict(arr, verbose=0)[0]   # shape: (6,)
    idx       = int(np.argmax(pred))
    conf      = float(pred[idx])
    class_name = CLASS_NAMES[idx]

    # map 6 classes → Fresh / Rotten
    label = "Fresh" if class_name in FRESH_CLASSES else "Rotten"

    return label, conf

def process_image(img_bgr):
    h, w    = img_bgr.shape[:2]
    results = yolo_model(img_bgr, conf=conf_thresh, verbose=False)
    detections = results[0]

    counts   = {"total": 0, "fresh": 0, "rotten": 0}
    details  = []
    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()

    c_fresh  = hex_to_rgb(box_color_fresh)
    c_rotten = hex_to_rgb(box_color_rotten)
    c_det    = hex_to_rgb(box_color_det)

    for i, det in enumerate(detections.boxes.data.tolist()):
        xmin, ymin, xmax, ymax, confidence, class_id = det[:6]
        class_name = yolo_model.names[int(class_id)]
        if class_name != fruit_class:
            continue

        counts["total"] += 1
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # crop สำหรับ classify
        px1 = max(0, xmin - PADDING)
        py1 = max(0, ymin - PADDING)
        px2 = min(w, xmax + PADDING)
        py2 = min(h, ymax + PADDING)
        crop = img_bgr[py1:py2, px1:px2]

        quality, conf_q = classify_fruit(crop)

        # เลือกสีตาม quality แต่ label บน box = detect confidence เท่านั้น
        if quality == "Fresh":
            color = c_fresh
            counts["fresh"] += 1
        elif quality == "Rotten":
            color = c_rotten
            counts["rotten"] += 1
        else:
            color = c_det

        # label บน bounding box = ชื่อผลไม้ + YOLO confidence เท่านั้น
        label = f"{fruit_class} {confidence:.0%}" if show_conf else fruit_class

        # วาด bounding box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 3)

        # label background
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (xmin, ymin - lh - 10), (xmin + lw + 4, ymin), color, -1)
        cv2.putText(annotated, label, (xmin + 2, ymin - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # เลขกำกับ
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        cv2.circle(annotated, (cx, cy), 16, (255, 255, 255), -1)
        cv2.putText(annotated, str(counts["total"]), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        details.append({
            "Fruit #": counts["total"],
            "Quality": quality or "—",
            "Confidence (classify)": f"{conf_q:.1%}" if conf_q else "—",
            "Confidence (detect)": f"{confidence:.1%}",
            "W×H (px)": f"{xmax-xmin}×{ymax-ymin}",
        })

    return annotated, counts, details

# ─── Upload (หลายรูป) ─────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📂 Upload Fruit Images (เลือกได้หลายรูป)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 Upload ภาพผลไม้เพื่อเริ่ม detection")
    st.stop()

# ─── วนประมวลผลแต่ละรูป แสดงเป็นบล็อกๆ ────────────────────────────────────────
for file_idx, uploaded in enumerate(uploaded_files):
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.expander(f"🖼️ ภาพที่ {file_idx+1} — {uploaded.name}", expanded=True):

        with st.spinner(f"🔍 กำลัง detect ภาพที่ {file_idx+1}..."):
            annotated, counts, details = process_image(img_bgr)

        # ─── Count Summary ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="count-box count-total">
                <div class="count-number-orange">{counts['total']}</div>
                <div class="count-label">🍊 Total Detected</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="count-box count-fresh">
                <div class="count-number-green">{counts['fresh']}</div>
                <div class="count-label">✅ Fresh</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="count-box count-rotten">
                <div class="count-number-red">{counts['rotten']}</div>
                <div class="count-label">❌ Rotten</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ─── Original + Annotated ───────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.caption("📷 Original")
            st.image(img_rgb, width=300)
        with col2:
            st.caption(f"🔍 Result — {counts['total']} fruit(s) found")
            st.image(annotated, width=300)

        if counts["total"] == 0:
            st.warning(f"ไม่พบ {fruit_class} — ลองลด Confidence Threshold ใน sidebar ค่ะ")
            continue

        # ─── Crop Gallery ───────────────────────────────────────────────────────
        st.divider()
        st.caption(f"🔬 Individual Fruit Analysis ({counts['total']} fruits)")

        h_img, w_img = img_bgr.shape[:2]
        results_raw  = yolo_model(img_bgr, conf=conf_thresh, verbose=False)[0]
        fruit_crops  = []

        for det in results_raw.boxes.data.tolist():
            xmin, ymin, xmax, ymax, confidence, class_id = det[:6]
            if yolo_model.names[int(class_id)] != fruit_class:
                continue
            xmin, ymin = int(xmin), int(ymin)
            xmax, ymax = int(xmax), int(ymax)
            px1 = max(0, xmin - PADDING)
            py1 = max(0, ymin - PADDING)
            px2 = min(w_img, xmax + PADDING)
            py2 = min(h_img, ymax + PADDING)
            crop_bgr = img_bgr[py1:py2, px1:px2]
            if crop_bgr.size == 0:
                continue
            quality, conf_q = classify_fruit(crop_bgr)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            fruit_crops.append({
                "crop_rgb": crop_rgb,
                "quality":  quality,
                "conf_q":   conf_q,
                "conf_det": confidence,
                "x1": xmin, "y1": ymin,
                "x2": xmax, "y2": ymax,
            })

        cols_per_row = 4
        for row_start in range(0, len(fruit_crops), cols_per_row):
            row_items = fruit_crops[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for ci, item in enumerate(row_items):
                fruit_no = row_start + ci + 1
                q  = item["quality"]
                cq = item["conf_q"]
                cd = item["conf_det"]
                x1, y1 = item["x1"], item["y1"]
                x2, y2 = item["x2"], item["y2"]
                w_px = x2 - x1
                h_px = y2 - y1

                if q == "Fresh":
                    badge = "🟢 Fresh"
                    color = "#38A169"
                elif q == "Rotten":
                    badge = "🔴 Rotten"
                    color = "#E53E3E"
                else:
                    badge = "⚪ Unknown"
                    color = "#888888"

                with cols[ci]:
                    st.image(item["crop_rgb"], width=300)
                    st.markdown(
                        f"<div style='text-align:center;font-weight:700;color:{color};font-size:1rem'>"
                        f"#{fruit_no} {badge}</div>"
                        f"<div style='text-align:center;color:#555;font-size:0.75rem;line-height:1.6'>"
                        f"📍 ({x1}, {y1}) → ({x2}, {y2})<br>"
                        f"📐 {w_px}×{h_px} px · 🎯 detect: {cd:.0%}</div>",
                        unsafe_allow_html=True
                    )

        # ─── Detail Table ────────────────────────────────────────────────────────
        if details:
            st.divider()
            df = pd.DataFrame(details)
            st.dataframe(df, width='stretch')

            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "⬇️ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name=f"result_{uploaded.name}",
                    mime="image/png",
                    key=f"dl_img_{file_idx}"
                )
            with col_dl2:
                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"result_{uploaded.name}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{file_idx}"
                )
