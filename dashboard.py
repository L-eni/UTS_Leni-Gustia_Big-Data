import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Model deteksi gender (YOLO)
    yolo_model = YOLO("model/Leni_Gustia_Laporan_4.pt")
    # Model klasifikasi alas kaki (CNN)
    classifier = tf.keras.models.load_model("model/Leni_Gustia_Laporan_2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Fungsi Deteksi & Klasifikasi
# ==========================
def detect_objects(img, conf_threshold=0.3):
    results = yolo_model(img)
    annotated_img = results[0].plot()
    detected_objects = []

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf >= conf_threshold:
            cls = int(box.cls)
            label = results[0].names[cls]
            detected_objects.append({"label": label, "confidence": round(conf * 100, 2)})

    return annotated_img, detected_objects


def classify_image(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = classifier.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    class_labels = ["Shoe", "Sandal", "Boot"]
    class_name = class_labels[class_index]
    confidence = np.max(prediction)
    return class_name, round(confidence * 100, 2)

# ==========================
# UI (Streamlit)
# ==========================
st.set_page_config(page_title="👟 Gender & Footwear AI Detection", layout="wide")
st.title("👟 Gender & Footwear Recognition App")
st.markdown("Aplikasi ini menggunakan **YOLOv8** untuk deteksi gender dan **CNN (TensorFlow)** untuk klasifikasi alas kaki.")

# Sidebar menu
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MODE 1: Deteksi Gender (YOLO)
# ==========================
if menu == "🧍 Deteksi Gender (YOLO)":
    st.subheader("🧍 Deteksi Gender (Men/Women) menggunakan YOLOv8")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔎 Mendeteksi objek..."):
            start_time = time.time()
            annotated_img, detections = detect_objects(img, conf_threshold)
            end_time = time.time()

        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
        st.success(f"⏱️ Waktu Proses: {end_time - start_time:.2f} detik")

        if detections:
            st.subheader("📋 Objek Terdeteksi:")
            for i, det in enumerate(detections):
                st.write(f"**{i+1}. {det['label']}** — Confidence: {det['confidence']}%")
        else:
            st.warning("Tidak ada objek terdeteksi di gambar ini.")

# ==========================
# MODE 2: Klasifikasi Alas Kaki (CNN)
# ==========================
elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
    st.subheader("👞 Klasifikasi Alas Kaki (Shoe / Sandal / Boot) menggunakan CNN")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🧠 Mengklasifikasikan jenis alas kaki..."):
            start_time = time.time()
            class_name, confidence = classify_image(img)
            end_time = time.time()

        st.success(f"✅ Jenis Alas Kaki: **{class_name}** ({confidence}%)")
        st.caption(f"⏱️ Waktu Proses: {end_time - start_time:.2f} detik")

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart AI Vision — Leni Gustia 👩‍💻 | YOLOv8 + TensorFlow CNN")
