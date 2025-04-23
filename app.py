import streamlit as st
import os
import uuid
import io
import tensorflow as tf
from PIL import Image
from mtcnn.mtcnn import MTCNN
import tempfile
from skimage.feature import local_binary_pattern, hog
from deepface import DeepFace
import cv2
import numpy as np


from predict_similiarity import predict_similarity
# Nanti kamu tambahkan import predict_ethnicity dan predict_age_emotion

# Folder uploads
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Session state
if "reset" not in st.session_state:
    st.session_state.reset = False

if "uploaded_file1" not in st.session_state:
    st.session_state.uploaded_file1 = None
if "uploaded_file2" not in st.session_state:
    st.session_state.uploaded_file2 = None

def upload_or_camera_input(mode="single"):
    """
    mode: 'single' untuk upload/capture 1 gambar
          'double' untuk upload/capture 2 gambar (seperti face similarity)
    """
    uploaded = False
    images = []

    input_mode = st.radio("Pilih metode input:", ("Upload Gambar", "Ambil dari Webcam"))

    if input_mode == "Upload Gambar":
        if mode == "single":
            uploaded_file = st.file_uploader("Upload file gambar", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)
                images.append(uploaded_file)
                uploaded = True
        elif mode == "double":
            uploaded_file1 = st.file_uploader("Upload file gambar 1", type=['jpg', 'jpeg', 'png'])
            uploaded_file2 = st.file_uploader("Upload file gambar 2", type=['jpg', 'jpeg', 'png'])

            if uploaded_file1 and uploaded_file2:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file1, caption="Gambar 1", use_container_width=True)
                with col2:
                    st.image(uploaded_file2, caption="Gambar 2", use_container_width=True)
                images.append(uploaded_file1)
                images.append(uploaded_file2)
                uploaded = True

    else:  # Ambil dari Webcam
        if mode == "single":
            captured_image = st.camera_input("Ambil gambar dari webcam")
            if captured_image:
                st.image(captured_image, caption="Gambar dari Webcam", use_container_width=True)
                images.append(captured_image)
                uploaded = True
        elif mode == "double":
            st.subheader("Ambil gambar 1 dari webcam")
            captured_image1 = st.camera_input("Ambil gambar dari webcam (Gambar 1)")

            st.subheader("Ambil gambar 2 dari webcam")
            captured_image2 = st.camera_input("Ambil gambar dari webcam (Gambar 2)")

            if captured_image1 and captured_image2:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(captured_image1, caption="Gambar 1", use_container_width=True)
                with col2:
                    st.image(captured_image2, caption="Gambar 2", use_container_width=True)
                images.append(captured_image1)
                images.append(captured_image2)
                uploaded = True

    return uploaded, images

def similarity_page():
    st.header("🔎 Face Similarity Detection")
    uploaded, images = upload_or_camera_input(mode="double")

    if uploaded:
        with st.spinner('Sedang mendeteksi wajah...'):
            result, score = predict_similarity(images[0], images[1])
            st.subheader("Hasil Face Similarity")
            st.metric(label="Similarity Score", value=f"{score:.4f}")
            st.metric(label="Prediction", value=result)



from predict_cnn import predict_ethnicity, get_visualization_image
from predict_ensemble import predict_ethnicity_ensemble # Tambahkan ini
from predict_custom import predict_ethnicity_custom

def ethnicity_page():
    st.header("🌎 Ethnicity Detection")
    algorithm_choose = st.selectbox("Pilih algoritma yang ingin digunakan", ["CNN(Convolution Neural Network)", "Ensemble Method (Recomended)", "Custom Feature Engineering"])
    uploaded, images = upload_or_camera_input(mode="single")

    if uploaded:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(images[0].read())
        temp_file.close()
        if algorithm_choose == "CNN(Convolution Neural Network)":
            # Simpan gambar sementara
            # Inisialisasi detektor dan label
            detector = MTCNN()
            class_labels = ['Batak', 'Jawa', 'Padang', 'Sunda']
            model = tf.keras.models.load_model('model/ethnicity_classification_best_model.h5')

            # Panggil fungsi prediksi
            predicted_label, y_pred_proba, image, box, confidence = predict_ethnicity(temp_file.name, model, detector, class_labels)

            if predicted_label:
                with st.spinner('Sedang mendeteksi etnis...'):
                    st.subheader("Hasil Deteksi Etnis")
                    st.success(f"Etnis wajah terdeteksi: **{predicted_label}** dengan confidence **{confidence:.2f}**")
                    # Tampilkan visualisasi
                    visualization_image = get_visualization_image(image, predicted_label, confidence, y_pred_proba, class_labels, box)
                    if visualization_image:
                        st.image(visualization_image, caption="Visualisasi Prediksi", use_container_width =True)
            else:
                st.error("Gagal mendeteksi wajah atau memprediksi etnis.")

            # Hapus file sementara
        elif algorithm_choose == "Ensemble Method (Recomended)":
            try:
                with st.spinner('Sedang mendeteksi etnis...'):
                    result, confidence, detected_face_img, chart_img = predict_ethnicity_ensemble(temp_file.name)
                    st.subheader("Hasil Deteksi Etnis (Ensemble)")
                    st.success(f"Etnis wajah terdeteksi: **{result}**  dengan confidence **{confidence:.2f}**")

                    st.image(detected_face_img, caption="Hasil Deteksi Wajah", use_container_width =True)
                    st.image(chart_img, caption="Grafik Probabilitas Prediksi", use_container_width =True)

            except Exception as e:
                st.error(f"Error pada prediksi ensemble: {e}")

        elif algorithm_choose == "Custom Feature Engineering":
            try:
                with st.spinner('Sedang mendeteksi etnis...'):
                    predicted_label, confidence, y_pred_proba, image_with_label_rgb = predict_ethnicity_custom(temp_file.name)
                    
                    st.subheader("Hasil Deteksi Etnis (Custom Feature Engineering)")
                    st.success(f"Etnis wajah terdeteksi: **{predicted_label}** dengan confidence **{confidence:.2f}**")

                    st.image(image_with_label_rgb, caption=f"Prediksi: {predicted_label} (Confidence: {confidence:.2f})", use_container_width=True)

                    st.markdown("### Probabilitas Prediksi")
                    probability_dict = {label: prob for label, prob in zip(y_pred_proba['labels'], y_pred_proba['probs'])}
                    for label, prob in probability_dict.items():
                        st.write(f"- **{label}**: {prob:.2f}")

                    st.bar_chart(probability_dict)
            except Exception as e:
                st.error(f"Gagal melakukan prediksi dengan Custom Feature Engineering: {e}")

        else:
            st.info("Metode ini belum diimplementasikan.")
        os.remove(temp_file.name)

from cekgenderage import analyze_faces
def age_emotion_page():
    st.header("🎭 Age, Gender & Emotion Detection")
    uploaded, images = upload_or_camera_input(mode="single")

    if uploaded:
        with st.spinner('Sedang mendeteksi etnis...'):
            file = images[0]
            image_bytes = file.read()
            image_pil = Image.open(io.BytesIO(image_bytes))

            try:
                results = analyze_faces(image_pil)
                st.subheader("Hasil Deteksi")

                col1, col2, col3 = st.columns(3)
                col_names = ["Haar Cascade", "MTCNN", "RetinaFace"]
                for i, (col, name) in enumerate(zip([col1, col2, col3], col_names)):
                    with col:
                        st.image(results[i]['image'], caption=name, use_container_width=True)

            except Exception as e:
                st.error(f"Gagal analisis: {e}")

def live_detect():
    st.title("Live Face Detection")
    run = st.button('Start Camera')

    FRAME_WINDOW = st.image([])

    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    frame_count = 0
    analysis_cache = []

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membaca frame dari webcam")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        if frame_count % 10 == 0:
            analysis_cache = []
            for result in results:
                x, y, w, h = result['box']
                x, y = max(x, 0), max(y, 0)
                face_crop = frame[y:y+h, x:x+w]

                try:
                    face_resized = cv2.resize(face_crop, (224, 224))
                    result_analysis = DeepFace.analyze(
                        img_path=face_resized,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=False
                    )

                    gender_dict = result_analysis[0]['gender']
                    gender = max(gender_dict, key=gender_dict.get)
                    age = result_analysis[0]['age']
                    emotion = max(result_analysis[0]['emotion'], key=result_analysis[0]['emotion'].get)

                    analysis_cache.append({
                        "box": (x, y, w, h),
                        "label": f"Gender: {gender}\nAge: {age} thn\nEmotion: {emotion}"
                    })
                except Exception as e:
                    print(f"Gagal analisis wajah: {e}")

        for data in analysis_cache:
            x, y, w, h = data["box"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            for idx, line in enumerate(data["label"].split('\n')):
                cv2.putText(frame, line, (x, y - 10 - (idx * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        frame_count += 1

    cap.release() 

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Face Similarity", "Ethnicity Detection", "Age & Emotion Detection", "Live Detection"])

# Judul besar
st.title("PCD 2024 - Face Recognition App")

# Routing page
if page == "Face Similarity":
    similarity_page()
elif page == "Ethnicity Detection":
    ethnicity_page()
elif page == "Age & Emotion Detection":
    age_emotion_page()
elif page == "Live Detection":
    live_detect()
