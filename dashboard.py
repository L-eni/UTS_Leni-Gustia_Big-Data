import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/Leni_Gustia_Laporan_4.pt"
    cnn_path = "model/Leni_Gustia_Laporan_2.h5"

    # Cek keberadaan file model
    if not os.path.exists(yolo_path):
        st.error(f"❌ File YOLO tidak ditemukan: {yolo_path}")
        st.stop()
    if not os.path.exists(cnn_path):
        st.error(f"❌ File CNN tidak ditemukan: {cnn_path}")
        st.stop()

    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(cnn_path)

    # Ambil class label dari model
    if hasattr(classifier, "classes"):
        class_labels = [cls for cls in classifier.classes]
    else:
        # fallback default, bisa diganti sesuai model training
        class_labels = ["Boot", "Sandal", "Shoe"]

    return yolo_model, classifier, class_labels

yolo_model, classifier, class_labels = load_models()

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
    try:
        img = img.convert("RGB")
        input_shape = classifier.input_shape[1:3]  # (height, width)
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = classifier.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        class_name = class_labels[class_index]
        confidence = np.max(prediction)
        return class_name, round(confidence * 100, 2)
    except Exception as e:
        st.error(f"⚠️ Terjadi error pada klasifikasi: {e}")
        return None, None

# ==========================
# UI (Streamlit)
# ==========================
st.set_page_config(page_title="👟 Gender & Footwear AI Detection", layout="wide")
st.title("👟 Gender & Footwear Recognition App")
st.markdown("Aplikasi ini menggunakan **YOLOv8** untuk deteksi gender dan **CNN (TensorFlow)** untuk klasifikasi alas kaki.")

# Sidebar
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MODE 1: Deteksi Gender (YOLO)
# ==========================
if menu == "🧍 Deteksi Gender (YOLO)":
    st.subheader("🧍 Deteksi Gender (Men/Women) menggunakan YOLOv8")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
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
    st.subheader("👞 Klasifikasi Alas Kaki menggunakan CNN")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🧠 Mengklasifikasikan jenis alas kaki..."):
            start_time = time.time()
            class_name, confidence = classify_image(img)
            end_time = time.time()

        if class_name:
            st.success(f"✅ Jenis Alas Kaki: **{class_name}** ({confidence}%)")
            st.caption(f"⏱️ Waktu Proses: {end_time - start_time:.2f} detik")

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart AI Vision — Leni Gustia 👩‍💻 | YOLOv8 + TensorFlow CNN")
