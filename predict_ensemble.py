# predict_ensemble.py

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
import joblib
import matplotlib.pyplot as plt
import io
from PIL import Image



# Load model hanya sekali saat import
model_resnet = load_model('model/resnet50_etnis_model.keras')
model_mobilenet = load_model('model/mobilenetv2_etnis_model.keras') 
model_efficientnet = load_model('model/efficientnetb3_etnis_model.keras')
label_encoder = joblib.load('model/label_encoder.pkl')

weights = {'resnet50': 0.3, 'mobilenetv2': 0.2, 'efficientnetb3': 0.5}

# Fungsi untuk ekstrak wajah
def extract_face(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if not faces:
        return None, None, None

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (224, 224))

    return face_resized, image, (x, y, w, h)

# Fungsi utama prediksi ensemble
def predict_ethnicity_ensemble(image_path):
    detector = MTCNN()
    face_resized, image, (x, y, w, h) = extract_face(image_path, detector)
    
    if face_resized is None:
        raise ValueError("Wajah tidak terdeteksi.")

    # Preprocessing
    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_resized = np.expand_dims(face_resized, axis=0)

    face_resnet = preprocess_resnet(face_resized.copy())
    face_mobilenet = preprocess_mobilenet(face_resized.copy())
    face_efficientnet = preprocess_efficientnet(face_resized.copy())

    # Prediksi
    y_pred_proba_resnet = model_resnet.predict(face_resnet)
    y_pred_proba_mobilenet = model_mobilenet.predict(face_mobilenet)
    y_pred_proba_efficientnet = model_efficientnet.predict(face_efficientnet)

    y_pred_proba = (weights['resnet50'] * y_pred_proba_resnet +
                    weights['mobilenetv2'] * y_pred_proba_mobilenet +
                    weights['efficientnetb3'] * y_pred_proba_efficientnet)
    y_pred_proba /= sum(weights.values())

    y_pred = np.argmax(y_pred_proba, axis=1)
    predicted_label = label_encoder.inverse_transform(y_pred)[0]
    confidence = y_pred_proba[0][y_pred[0]]

    # =========================
    # 5. Visualisasi
    # =========================
    image_with_label = image.copy()
    cv2.rectangle(image_with_label, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image_with_label, f"{predicted_label} ({confidence:.2f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    image_with_label_rgb = cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_label_rgb)
    plt.title(f"Prediksi: {predicted_label} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(label_encoder.classes_, y_pred_proba[0], color=['blue', 'orange', 'green', 'red'])
    plt.title('Probabilitas Prediksi')
    plt.xlabel('Kelas')
    plt.ylabel('Probabilitas')
    plt.ylim(0, 1)
    for i, v in enumerate(y_pred_proba[0]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()

        # Konversi image_with_label untuk Streamlit
    image_with_label_rgb = cv2.cvtColor(image_with_label, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_with_label_rgb)

    # Gambar chart probabilitas ke objek buffer
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_encoder.classes_, y_pred_proba[0], color=['blue', 'orange', 'green', 'red'])
    ax.set_title('Probabilitas Prediksi')
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Probabilitas')
    ax.set_ylim(0, 1)
    for i, v in enumerate(y_pred_proba[0]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    chart_buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(chart_buf, format="png")
    chart_buf.seek(0)
    plt.close(fig)
    chart_image = Image.open(chart_buf)

    return predicted_label, confidence, pil_image, chart_image

    # return predicted_label, confidence

