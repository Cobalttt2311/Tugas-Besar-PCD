import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# --- Custom CosineSimilarity Layer ---
class CosineSimilarity(Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        dot_product = K.sum(x * y, axis=1, keepdims=True)
        norm_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
        norm_y = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True))
        return dot_product / (norm_x * norm_y + K.epsilon())

# --- Load Model ---
model_path = 'model/siamese_cosine_model_mantap.h5'
model = load_model(model_path, custom_objects={'CosineSimilarity': CosineSimilarity})
print("[INFO] Model loaded.")

# --- Crop wajah dari gambar menggunakan MTCNN ---
def detect_and_crop_face(uploaded_file):
    uploaded_file.seek(0)  # reset pointer ke awal file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    if file_bytes.size == 0:
        raise ValueError("[ERROR] File kosong atau gagal dibaca.")

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("[ERROR] Gagal decode gambar. Format mungkin tidak didukung.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    results = detector.detect_faces(image_rgb)

    if len(results) == 0:
        raise ValueError("[ERROR] Tidak ada wajah terdeteksi.")

    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = image_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32") / 255.0
    return face



# --- Prediksi similarity dua gambar ---
def predict_similarity(img1_path, img2_path, threshold=0.4187):
    face1 = detect_and_crop_face(img1_path)
    face2 = detect_and_crop_face(img2_path)

    face1 = np.expand_dims(face1, axis=0)
    face2 = np.expand_dims(face2, axis=0)

    score = model.predict([face1, face2])[0][0]
    print(f"[RESULT] Similarity Score: {score:.4f}")

    if score >= threshold:
        result = "SAME Person"
        print("[INFO] Prediction: SAME person")
    else:
        result = "DIFFERENT Person"
        print("[INFO] Prediction: DIFFERENT person")
    return result, score

# # --- Input Paths ---
# img1_path = "nalen_1.jpg"
# img2_path = "nalen_2.jpg"

# predict_similarity(img1_path, img2_path)

