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
st.set_page_config(page_title="👟 Gender & Footwear AI Detection", layout="wide", page_icon="👟")

st.title("👟 Gender & Footwear Recognition App")
st.markdown("""
Aplikasi ini menggunakan **YOLOv8** untuk deteksi gender dan **CNN (TensorFlow)** 
untuk klasifikasi alas kaki.  

🧍 YOLO mendeteksi objek gender (Men/Women).  
👞 CNN mengklasifikasi jenis alas kaki (Shoe/Sandal/Boot) — **hanya jika objek alas kaki terdeteksi**.
""")

# ==========================
# Load Models
# ==========================
@st.cache_resource(show_spinner=False)
def load_models():
    yolo_gender_path = "model/Leni Gustia_Laporan 4.pt"  # model YOLO Gender
    yolo_shoe_path = "model/footwear_yolo.pt"             # <— model YOLO khusus alas kaki (kalau ada)
    cnn_path = "model/Leni_Gustia_Laporan_2.h5"

    if not os.path.exists(yolo_gender_path):
        st.error(f"❌ File YOLO Gender tidak ditemukan: `{yolo_gender_path}`")
        st.stop()
    if not os.path.exists(cnn_path):
        st.error(f"❌ File CNN tidak ditemukan: `{cnn_path}`")
        st.stop()

    yolo_gender_model = YOLO(yolo_gender_path)
    # Jika tidak punya YOLO alas kaki, bagian ini bisa dikomentari
    yolo_shoe_model = YOLO(yolo_shoe_path) if os.path.exists(yolo_shoe_path) else None
    classifier = tf.keras.models.load_model(cnn_path)
    class_labels = ["Boot", "Sandal", "Shoe"]

    return yolo_gender_model, yolo_shoe_model, classifier, class_labels

yolo_gender_model, yolo_shoe_model, classifier, class_labels = load_models()

# ==========================
# Fungsi Deteksi Gender
# ==========================
def detect_gender(img, conf_threshold=0.3):
    results = yolo_gender_model(img)
    annotated_img = results[0].plot()
    detected_objects = []
    valid_labels = ["Men", "Women"]

    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = results[0].names[cls]
        if conf >= conf_threshold:
            if label in valid_labels:
                detected_objects.append({"label": label, "confidence": round(conf * 100, 2)})
    return annotated_img, detected_objects

# ==========================
# Fungsi Deteksi Alas Kaki (Filter Domain CNN)
# ==========================
def detect_shoe(img, conf_threshold=0.3):
    if yolo_shoe_model is None:
        # Kalau YOLO alas kaki tidak ada → skip deteksi
        return True  
    results = yolo_shoe_model(img)
    valid_labels = ["Shoe", "Sandal", "Boot"]
    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = results[0].names[cls]
        if conf >= conf_threshold and label in valid_labels:
            return True
    return False

# ==========================
# Fungsi Klasifikasi CNN
# ==========================
def classify_footwear(img):
    img = img.convert("RGB")
    input_shape = classifier.input_shape[1:3]
    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = classifier.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]
    confidence = np.max(prediction)

    return class_name, round(confidence * 100, 2)

# ==========================
# Sidebar
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["🧍 Deteksi Gender (YOLO)", "👞 Klasifikasi Alas Kaki (CNN)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# MODE YOLO GENDER
# ==========================
if menu == "🧍 Deteksi Gender (YOLO)":
    st.subheader("🧍 Deteksi Gender (Men/Women)")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        with st.spinner("🔎 Mendeteksi objek gender..."):
            start_time = time.time()
            annotated_img, detections = detect_gender(img, conf_threshold)
            duration = time.time() - start_time

        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
        st.success(f"⏱️ Waktu Proses: {duration:.2f} detik")
        if detections:
            for i, det in enumerate(detections):
                st.write(f"**{i+1}. {det['label']}** — Confidence: {det['confidence']}%")
        else:
            st.warning("⚠️ Tidak ada objek gender terdeteksi.")
    else:
        st.info("📤 Silakan unggah gambar untuk mulai deteksi.")

# ==========================
# MODE CNN FOOTWEAR
# ==========================
elif menu == "👞 Klasifikasi Alas Kaki (CNN)":
    st.subheader("👞 Klasifikasi Alas Kaki (Shoe/Sandal/Boot)")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        # 🔎 Filter domain pakai YOLO alas kaki
        with st.spinner("🔍 Mengecek apakah gambar sesuai domain alas kaki..."):
            is_footwear = detect_shoe(img, conf_threshold)

        if not is_footwear:
            st.error("❌ Gambar ini tidak mengandung alas kaki — klasifikasi CNN dibatalkan.")
        else:
            with st.spinner("🧠 Mengklasifikasikan alas kaki..."):
                start_time = time.time()
                class_name, confidence = classify_footwear(img)
                duration = time.time() - start_time

            st.success(f"✅ Jenis Alas Kaki: **{class_name}** ({confidence}%)")
            st.caption(f"⏱️ Waktu Proses: {duration:.2f} detik")
    else:
        st.info("📤 Silakan unggah gambar alas kaki untuk klasifikasi.")
