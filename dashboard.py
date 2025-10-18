import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="👟 Gender & Footwear AI Detection",
    layout="wide",
    page_icon="👟"
)

st.title("👟 Gender & Footwear Recognition App")
st.markdown("""
Aplikasi ini menggunakan **YOLOv8** untuk deteksi gender dan **CNN (TensorFlow)** untuk klasifikasi alas kaki.  
Hanya dua domain model yang digunakan:
- 🧍 **YOLO:** Men / Women
- 👞 **CNN:** Shoe / Sandal / Boot
""")

# ==========================
# Load Models
# ==========================
@st.cache_resource(show_spinner=False)
def load_models():
    # Ganti nama file sesuai yang benar (pastikan ekstensi .pt untuk YOLO)
    yolo_path = "model/Leni Gustia_Laporan 4.pt"
    cnn_path = "model/Leni_Gustia_Laporan_2.h5"

    if not os.path.exists(yolo_path):
        st.error(f"❌ File YOLO tidak ditemukan: {yolo_path}")
        st.stop()

    if not os.path.exists(cnn_path):
        st.error(f"❌ File CNN tidak ditemukan: {cnn_path}")
        st.stop()

    # Load model
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(cnn_path)
    class_labels = ["Boot", "Sandal", "Shoe"]
    return yolo_model, classifier, class_labels

yolo_model, classifier, class_labels = load_models()

# ==========================
# Fungsi Deteksi YOLO
# ==========================
def detect_objects(img, conf_threshold=0.3):
    """Deteksi gender (Men/Women)."""
    results = yolo_model(img)
    annotated_img = results[0].plot()

    detected_objects = []
    valid_labels = ["Men", "Women"]

    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = results[0].names[cls]

        if conf >= conf_threshold:
            # Jika hasil bukan Men/Women → tandai objek tidak sesuai
            if label not in valid_labels:
                detected_objects.append({
                    "label": "Objek tidak sesuai dengan model gender",
                    "confidence": round(conf * 100, 2)
                })
            else:
                detected_objects.append({
                    "label": label,
                    "confidence": round(conf * 100, 2)
                })

    return annotated_img, detected_objects

# ==========================
# Fungsi Klasifikasi CNN
# ==========================
def classify_image(img):
    """Klasifikasi alas kaki (Shoe/Sandal/Boot)."""
    try:
        img = img.convert("RGB")
        input_shape = classifier.input_shape[1:3]
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        class_name = class_labels[class_index]
        confidence = np.max(prediction)

        valid_labels = ["Shoe", "Sandal", "Boot"]
        if class_name not in valid_labels:
            class_name = "Model atau objek ini tidak sesuai"

        return class_name, round(confidence * 100, 2)
    except Exception as e:
        st.error(f"⚠️ Terjadi error saat klasifikasi: {e}")
        return "Model atau objek ini tidak sesuai", 0

# ==========================
# Sidebar
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MODE 1: YOLO
# ==========================
if menu == "🧍 Deteksi Gender (YOLO)":
    st.subheader("🧍 Deteksi Gender (Men/Women)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔎 Mendeteksi objek..."):
            start_time = time.time()
            annotated_img, detections = detect_objects(img, conf_threshold)
            duration = time.time() - start_time

        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
        st.success(f"⏱️ Waktu Proses: {duration:.2f} detik")

        if detections:
            st.subheader("📋 Hasil Deteksi:")
            for i, det in enumerate(detections):
                st.write(f"**{i+1}. {det['label']}** — Confidence: {det['confidence']}%")
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi atau bukan domain gender.")
    else:
        st.info("📤 Silakan unggah gambar untuk memulai deteksi.")

# ==========================
# MODE 2: CNN
# ==========================
elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
    st.subheader("👞 Klasifikasi Alas Kaki (Shoe/Sandal/Boot)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🧠 Mengklasifikasikan..."):
            start_time = time.time()
            class_name, confidence = classify_image(img)
            duration = time.time() - start_time

        if class_name == "Model atau objek ini tidak sesuai":
            st.error("⚠️ Gambar ini tidak sesuai untuk model alas kaki.")
        else:
            st.success(f"✅ Jenis Alas Kaki: **{class_name}** ({confidence}%)")
            st.caption(f"⏱️ Waktu Proses: {duration:.2f} detik")
    else:
        st.info("📤 Silakan unggah gambar untuk klasifikasi alas kaki.")

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart AI Vision — Leni Gustia 👩‍💻 | YOLOv8 + TensorFlow CNN")
