import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

# ==================== LOAD MODEL ====================
# YOLO Model for Gender Detection
yolo_model = YOLO("model_gender_yolo.pt")

# CNN Model for Shoe vs Sandal vs Boot Classification
cnn_model = tf.keras.models.load_model("model_alas_kaki.h5")
cnn_labels = ["shoe", "sandal", "boot"]  # urut sesuai pelatihan model

# ==================== UI ====================
st.set_page_config(page_title="Deteksi Gender & Alas Kaki", layout="centered")
st.title("🧠 Deteksi Gender & Klasifikasi Alas Kaki")

mode = st.radio("Pilih Mode:", ["Deteksi Gender (YOLO)", "Klasifikasi Alas Kaki (CNN)"])
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.01)
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
    img_array = np.array(image)

    # ==================== MODE YOLO ====================
    if mode == "Deteksi Gender (YOLO)":
        results = yolo_model.predict(img_array, conf=confidence_threshold)
        boxes = results[0].boxes

        # Jika tidak ada objek terdeteksi
        if boxes is None or len(boxes) == 0:
            st.warning("❌ Tidak ada gender terdeteksi (mungkin bukan gambar orang)")
        else:
            # Ambil box dengan confidence tertinggi
            confs = boxes.conf.numpy()
            best_idx = int(np.argmax(confs))
            conf = float(confs[best_idx])
            cls_id = int(boxes.cls[best_idx])
            label = yolo_model.names[cls_id]

            if conf >= confidence_threshold:
                st.success(f"✅ Gender: {label} — Confidence: {conf*100:.2f}%")
                st.image(results[0].plot(), caption="Hasil Deteksi", use_column_width=True)
            else:
                st.warning("❌ Confidence terlalu rendah / tidak sesuai")

    # ==================== MODE CNN ====================
    elif mode == "Klasifikasi Alas Kaki (CNN)":
        # Resize gambar sesuai input CNN
        img_resized = image.resize((150, 150))  # sesuaikan ukuran input model
        img_norm = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        prediction = cnn_model.predict(img_input)[0]
        best_conf = np.max(prediction)
        best_idx = np.argmax(prediction)
        best_label = cnn_labels[best_idx]

        if best_conf >= confidence_threshold:
            st.success(f"👟 Jenis Alas Kaki: {best_label} — Confidence: {best_conf*100:.2f}%")
        else:
            st.warning("❌ Bukan gambar alas kaki atau confidence terlalu rendah")
