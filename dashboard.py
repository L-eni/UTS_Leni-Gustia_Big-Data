# app.py
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import os
import pandas as pd

# -----------------------
# Config page
# -----------------------
st.set_page_config(page_title="👟 Gender & Footwear AI Detection", layout="wide", page_icon="👟")
st.title("👟 Gender & Footwear Recognition App (Debuggable)")

st.markdown("""
Instruksi singkat:
- Mode `Deteksi Gender (YOLO)` hanya untuk mendeteksi `Men`/`Women`.
- Mode `Klasifikasi Alas Kaki (CNN)` hanya berjalan jika gambar lolos *domain check* (YOLO footwear atau heuristik person-check).
- Gunakan tabel debug untuk melihat label/conf/area bbox dan sesuaikan threshold.
""")

# -----------------------
# Paths: ubah sesuai lokasi model
# -----------------------
YOLO_GENDER_PATH = "model/Leni Gustia_Laporan 4.pt"   # model YOLO untuk gender (harus memiliki label Men/Women)
YOLO_SHOE_PATH = "model/footwear_yolo.pt"            # optional: YOLO khusus footwear (jika ada)
CNN_PATH = "model/Leni_Gustia_Laporan_2.h5"          # model klasifikasi alas kaki (Shoe/Sandal/Boot)

# -----------------------
# Load models (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models():
    if not os.path.exists(YOLO_GENDER_PATH):
        st.error(f"❌ File YOLO gender tidak ditemukan: {YOLO_GENDER_PATH}")
        st.stop()
    if not os.path.exists(CNN_PATH):
        st.error(f"❌ File CNN tidak ditemukan: {CNN_PATH}")
        st.stop()

    yolo_gender = YOLO(YOLO_GENDER_PATH)
    yolo_shoe = YOLO(YOLO_SHOE_PATH) if os.path.exists(YOLO_SHOE_PATH) else None
    cnn = tf.keras.models.load_model(CNN_PATH)
    # Pastikan label CNN sesuai urutan pelatihan:
    cnn_labels = ["Boot", "Sandal", "Shoe"]

    # Ambil mapping label YOLO jika tersedia
    yolo_gender_names = getattr(yolo_gender, "model", None)
    # ultralytics: results[0].names biasanya ada pada runtime

    return yolo_gender, yolo_shoe, cnn, cnn_labels

yolo_gender_model, yolo_shoe_model, classifier, class_labels = load_models()

# -----------------------
# Sidebar controls
# -----------------------
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold (YOLO)", 0.1, 1.0, 0.30, 0.05)
person_area_threshold = st.sidebar.slider("Person area threshold (reject if bigger)", 0.05, 0.6, 0.25, 0.05)
cnn_min_confidence = st.sidebar.slider("CNN min confidence (%)", 10, 95, 50, 5)
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.write("Tips tuning:")
st.sidebar.write("- Naikkan `person area threshold` jika YOLO mendeteksi person kecil di gambar sepatu.")
st.sidebar.write("- Kalau punya YOLO footwear, letakkan di `model/footwear_yolo.pt`.")

# -----------------------
# Util: hitung area ratio
# -----------------------
def bbox_area_ratio_xyxy(x1, y1, x2, y2, img_w, img_h):
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return (w * h) / (img_w * img_h)

# -----------------------
# Fungsi: deteksi gender dengan filter area
# -----------------------
def detect_gender_with_debug(img_pil, conf_th):
    """Mengembalikan results list (label, conf, area_ratio, xyxy) dan annotated image."""
    results = yolo_gender_model(img_pil)  # ultralytics menerima PIL
    res0 = results[0]
    img_w, img_h = img_pil.size

    debug_rows = []
    annotated = img_pil.copy()
    draw = ImageDraw.Draw(annotated)

    # names mapping
    names = res0.names if hasattr(res0, "names") else {}

    for box in res0.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = names.get(cls, str(cls))
        # ultralytics box.xyxy is tensor-like; convert
        try:
            xyxy = box.xyxy[0].tolist()
        except Exception:
            # fallback if format berbeda
            xyxy = [float(v) for v in box.xyxy]
        x1, y1, x2, y2 = xyxy
        area_ratio = bbox_area_ratio_xyxy(x1, y1, x2, y2, img_w, img_h)

        debug_rows.append({
            "label": label,
            "confidence": conf,
            "area_ratio": area_ratio,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

        # annotate (only if conf >= 0.05 to visualize)
        if conf >= 0.05:
            # buat kotak dan teks
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            txt = f"{label} {conf:.2f} ar={area_ratio:.2f}"
            draw.text((x1, max(0, y1-15)), txt, fill="black")

    df = pd.DataFrame(debug_rows)
    return df, annotated

# -----------------------
# Fungsi deteksi footwear via YOLO (opsional)
# -----------------------
def detect_footwear_by_yolo(img_pil, conf_th):
    if yolo_shoe_model is None:
        return None  # no model
    results = yolo_shoe_model(img_pil)
    res0 = results[0]
    img_w, img_h = img_pil.size
    names = res0.names if hasattr(res0, "names") else {}
    found = []
    for box in res0.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = names.get(cls, str(cls))
        try:
            xyxy = box.xyxy[0].tolist()
        except Exception:
            xyxy = [float(v) for v in box.xyxy]
        x1, y1, x2, y2 = xyxy
        area_ratio = bbox_area_ratio_xyxy(x1, y1, x2, y2, img_w, img_h)
        if conf >= conf_th and label.lower() in ("shoe", "sandal", "boot"):
            found.append({"label": label, "conf": conf, "area_ratio": area_ratio, "xyxy": xyxy})
    return found

# -----------------------
# Fungsi klasifikasi CNN
# -----------------------
def classify_cnn(img_pil):
    pil = img_pil.convert("RGB")
    h, w = classifier.input_shape[1:3]
    pil_r = pil.resize((w, h))
    x = image.img_to_array(pil_r)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = classifier.predict(x, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = class_labels[idx]
    return label, conf  # conf in [0,1]

# -----------------------
# UI: Mode Deteksi Gender
# -----------------------
if menu == "🧍 Deteksi Gender (YOLO)":
    st.subheader("🧍 Deteksi Gender (Men/Women)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔎 Mendeteksi gender (YOLO) ..."):
            start = time.time()
            df_debug, ann = detect_gender_with_debug(img, conf_threshold)
            duration = time.time() - start

        st.image(ann, caption="Hasil Deteksi (annotated)", use_container_width=True)
        st.success(f"⏱️ Waktu Proses: {duration:.2f} detik")

        if df_debug.empty:
            st.warning("⚠️ Tidak ada object terdeteksi oleh YOLO gender.")
        else:
            # Tampilkan hanya deteksi yang memenuhi conf_threshold
            df_show = df_debug[df_debug["confidence"] >= conf_threshold].copy()
            df_show["confidence_pct"] = (df_show["confidence"] * 100).round(2)
            df_show["area_ratio_pct"] = (df_show["area_ratio"] * 100).round(2)
            if df_show.empty:
                st.warning("⚠️ Tidak ada deteksi gender di atas confidence threshold.")
            else:
                st.write("Deteksi yang melewati threshold:")
                st.dataframe(df_show[["label", "confidence_pct", "area_ratio_pct", "x1","y1","x2","y2"]])

                # filter untuk person besar
                # hanya ambil label Men/Women yang area_ratio >= person_area_threshold
                df_person_big = df_show[(df_show["label"].isin(["Men", "Women"])) & (df_show["area_ratio"] >= person_area_threshold)]
                if not df_person_big.empty:
                    for _, row in df_person_big.iterrows():
                        st.write(f"**{row['label']}** — Confidence: {row['confidence']*100:.2f}%, Area: {row['area_ratio']*100:.2f}%")
                else:
                    st.info("⚠️ Tidak ada person besar terdeteksi (atau semua di bawah area threshold).")

    else:
        st.info("📤 Silakan unggah gambar untuk mendeteksi gender.")

# -----------------------
# UI: Mode Klasifikasi Alas Kaki
# -----------------------
elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
    st.subheader("👞 Klasifikasi Alas Kaki (Shoe/Sandal/Boot) — dengan Domain Check")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔍 Memeriksa domain gambar (YOLO footwear / heuristik person)..."):
            start = time.time()
            # 1) coba YOLO footwear bila ada
            yolo_foot_found = detect_footwear_by_yolo(img, conf_threshold)
            # 2) deteksi gender (person) debug
            df_debug, ann_gender = detect_gender_with_debug(img, conf_threshold)
            # check if there's a large person
            large_person_exists = False
            if not df_debug.empty:
                # cari Men/Women dengan area >= person_area_threshold & conf >= conf_threshold
                cond = (df_debug["label"].isin(["Men", "Women"])) & (df_debug["confidence"] >= conf_threshold) & (df_debug["area_ratio"] >= person_area_threshold)
                large_person_exists = df_debug[cond].shape[0] > 0

            duration = time.time() - start

        st.caption(f"⏱️ Waktu pemeriksaan domain: {duration:.2f} detik")
        st.image(ann_gender, caption="Hasil Deteksi Gender (debug)", use_container_width=True)

        # Logic keputusan:
        if yolo_foot_found is False:
            st.error("❌ YOLO footwear ada, tetapi tidak menemukan objek alas kaki (cek model footwear jika ingin presisi lebih baik).")
        elif yolo_foot_found is not None and len(yolo_foot_found) > 0:
            # Ada deteksi footwear oleh YOLO -> lanjutkan ke CNN (opsional crop)
            st.success("✔️ YOLO footwear menemukan objek alas kaki — melanjutkan ke CNN.")
            # (opsional) crop berdasarkan bbox terbesar
            # Kita lakukan klasifikasi pada keseluruhan gambar untuk sederhana
            label, conf = classify_cnn(img)
            conf_pct = conf * 100
            if conf_pct < cnn_min_confidence:
                st.error(f"⚠️ CNN kurang yakin ({conf_pct:.2f}%) — gambar mungkin tidak sesuai domain.")
            else:
                st.success(f"✅ Jenis Alas Kaki: **{label}** ({conf_pct:.2f}%)")
        else:
            # Tidak ada YOLO footwear model (None) -> pakai heuristik person-check
            if yolo_shoe_model is None:
                if large_person_exists:
                    st.error("❌ Gambar tampak foto orang (person besar terdeteksi) — klasifikasi alas kaki dibatalkan.")
                    # tampilkan tabel debug untuk inspeksi
                    st.dataframe(df_debug[["label","confidence","area_ratio"]])
                else:
                    st.info("⚠️ Tidak terdeteksi person besar — melanjutkan ke CNN (fallback).")
                    label, conf = classify_cnn(img)
                    conf_pct = conf * 100
                    if conf_pct < cnn_min_confidence:
                        st.error(f"⚠️ CNN kurang yakin ({conf_pct:.2f}%) — gambar kemungkinan tidak sesuai domain alas kaki.")
                    else:
                        st.success(f"✅ Jenis Alas Kaki: **{label}** ({conf_pct:.2f}%)")
            else:
                # yolo_shoe_model ada tapi tidak menemukan => reject
                st.error("❌ YOLO footwear tidak menemukan object alas kaki (padahal model footwear tersedia).")
                st.dataframe(df_debug[["label","confidence","area_ratio"]])

    else:
        st.info("📤 Silakan unggah gambar untuk klasifikasi alas kaki.")
