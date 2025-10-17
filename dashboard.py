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
    yolo_model = YOLO("model/Leni Gustia_Laporan 4.pt")  # Model deteksi gender
    classifier = tf.keras.models.load_model("model/Leni_Gustia_Laporan 2.h5")  # Model klasifikasi alas kaki
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Fungsi Prediksi Gabungan
# ==========================
def combined_detection_and_classification(img, conf_threshold=0.3):
    results = yolo_model(img)
    annotated_img = results[0].plot()
    detections = results[0].boxes

    combined_results = []

    # Loop tiap objek terdeteksi
    for det in detections:
        conf = float(det.conf)
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0])
        label = results[0].names[int(det.cls)]
        cropped_img = img.crop((x1, y1, x2, y2))

        # Klasifikasi alas kaki
        img_resized = cropped_img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = classifier.predict(img_array)
        class_index = np.argmax(pred)
        prob = np.max(pred)

        class_labels = ["Shoe", "Sandal", "Boot"]
        shoe_label = class_labels[class_index]

        combined_results.append({
            "gender": label,
            "footwear": shoe_label,
            "prob": round(prob * 100, 2),
            "conf": round(conf * 100, 2)
        })

    return annotated_img, combined_results

# ==========================
# UI
# ==========================
st.set_page_config(page_title="👟 Gender & Footwear Detection", layout="wide")
st.title("👟 Smart Gender & Footwear Recognition System")

st.markdown("""
Aplikasi ini menggabungkan **deteksi gender** (Men/Women) menggunakan YOLO 
dan **klasifikasi alas kaki** (Shoe/Sandal/Boot) menggunakan CNN.
""")

menu = st.sidebar.radio("Pilih Mode:", ["🔍 Deteksi & Klasifikasi", "📊 Statistik & Informasi"])

if menu == "🔍 Deteksi & Klasifikasi":
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔎 Menganalisis gambar..."):
            start_time = time.time()
            annotated_img, results_combined = combined_detection_and_classification(img, conf_threshold)
            end_time = time.time()

        st.image(annotated_img, caption="Hasil Deteksi & Klasifikasi", use_container_width=True)
        st.success(f"⏱️ Waktu Proses: {end_time - start_time:.2f} detik")

        if results_combined:
            st.subheader("📋 Hasil Analisis")
            for i, res in enumerate(results_combined):
                st.markdown(f"""
                **Objek {i+1}:**
                - Gender: 🧍‍♂️ *{res['gender']}*
                - Jenis Alas Kaki: 👞 *{res['footwear']}*
                - Confidence Deteksi: {res['conf']}%
                - Probabilitas Klasifikasi: {res['prob']}%
                ---
                """)
        else:
            st.warning("Tidak ada objek yang terdeteksi di atas ambang batas confidence.")

elif menu == "📊 Statistik & Informasi":
    st.header("📊 Statistik Model & Informasi Teknis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Kelas Klasifikasi", "3", "Shoe, Sandal, Boot")
        st.metric("Jenis Model Deteksi", "YOLOv8 Custom", "Men/Women")
    with col2:
        st.metric("Framework CNN", "TensorFlow Keras")
        st.metric("Framework Deteksi", "Ultralytics YOLO")

    st.markdown("""
    ### 📘 Deskripsi Teknis
    - Model YOLO digunakan untuk **mendeteksi manusia dan menentukan gender**.
    - Model CNN digunakan untuk **mengenali jenis alas kaki** pada gambar terdeteksi.
    - Gambar diproses dalam pipeline:
      1. YOLO → Deteksi gender & bounding box  
      2. Crop hasil deteksi  
      3. CNN → Prediksi jenis alas kaki  
    """)

    st.markdown("""
    ### 💡 Ide Pengembangan
    - Tambahkan fitur **video live detection (webcam)**.
    - Simpan hasil deteksi ke database (CSV/SQLite).
    - Gunakan Streamlit tabs untuk pemisahan hasil & laporan.
    - Tambahkan **visualisasi statistik hasil deteksi**.
    """)

# ==========================
# Footer
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart Vision Project — Leni Gustia 👩‍💻 | Powered by YOLOv8 & TensorFlow")
