import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

# =========================
# 1. Fungsi untuk Ekstraksi Wajah
# =========================
def extract_face(image_path, detector, output_size=(224, 224)):
    """
    Ekstrak wajah dari gambar menggunakan MTCNN.
    Mengembalikan wajah yang di-resize, gambar asli, dan koordinat kotak wajah.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Gagal memuat gambar: {image_path}")
            return None, None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        
        if not faces:
            print(f"[ERROR] Tidak ada wajah terdeteksi pada gambar: {image_path}")
            return None, None, None
        
        # Ambil wajah pertama
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face = image_rgb[y:y+h, x:x+w]
        
        # Resize wajah untuk model
        face_resized = cv2.resize(face, output_size)
        face_resized = face_resized / 255.0  # Normalisasi ke [0, 1]
        
        return face_resized, image, (x, y, w, h)
    except Exception as e:
        print(f"[ERROR] Gagal mengekstrak wajah dari gambar {image_path}: {e}")
        return None, None, None

# =========================
# 2. Load Model
# =========================
try:
    model = tf.keras.models.load_model('model/ethnicity_classification_best_model.h5')
    print("[INFO] Model berhasil dimuat dari ethnicity_classification_best_model.h5")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")
    sys.exit(1)

# Kelas etnis
class_labels = ['Batak', 'Jawa', 'Padang', 'Sunda']

# =========================
# 3. Fungsi untuk Prediksi
# =========================
def predict_ethnicity(image_path, model, detector, class_labels):
    """
    Prediksi etnis dari gambar menggunakan model MobileNetV2.
    Mengembalikan label prediksi, probabilitas, gambar asli, dan koordinat kotak wajah.
    """
    # Ekstrak wajah
    face_resized, image, box = extract_face(image_path, detector)
    if face_resized is None:
        return None, None, None, None
    
    # Siapkan input untuk model
    face_input = np.expand_dims(face_resized, axis=0)
    
    # Prediksi
    try:
        y_pred_proba = model.predict(face_input, verbose=0)[0]
        y_pred = np.argmax(y_pred_proba)
        predicted_label = class_labels[y_pred]
        confidence = y_pred_proba[y_pred]
        
        return predicted_label, y_pred_proba, image, box, confidence
    except Exception as e:
        print(f"[ERROR] Gagal melakukan prediksi untuk {image_path}: {e}")
        return None, None, None, None

# =========================
# 4. Fungsi untuk Visualisasi
# =========================
def visualize_prediction(image, predicted_label, confidence, y_pred_proba, class_labels, box):
    """
    Visualisasikan gambar dengan kotak wajah dan probabilitas prediksi.
    """
    if image is None or box is None:
        print("[ERROR] Tidak dapat memvisualisasikan karena gambar atau kotak wajah tidak valid.")
        return
    
    x, y, w, h = box
    image_with_label = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Gambar kotak wajah dan label
    cv2.rectangle(image_with_label, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image_with_label, f"{predicted_label} ({confidence:.2f})", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Plot gambar dan probabilitas
    plt.figure(figsize=(12, 5))
    
    # Tampilkan gambar
    plt.subplot(1, 2, 1)
    plt.imshow(image_with_label)
    plt.title(f"Prediksi: {predicted_label} (Confidence: {confidence:.2f})")
    plt.axis('off')
    
    # Plot probabilitas
    plt.subplot(1, 2, 2)
    plt.bar(class_labels, y_pred_proba, color=['blue', 'orange', 'green', 'red'])
    plt.title('Probabilitas Prediksi')
    plt.xlabel('Kelas')
    plt.ylabel('Probabilitas')
    plt.ylim(0, 1)
    for i, v in enumerate(y_pred_proba):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


from io import BytesIO
from PIL import Image

def get_visualization_image(image, predicted_label, confidence, y_pred_proba, class_labels, box):
    """
    Mengembalikan visualisasi dalam bentuk gambar PIL untuk ditampilkan di Streamlit.
    """
    if image is None or box is None:
        print("[ERROR] Tidak dapat memvisualisasikan karena gambar atau kotak wajah tidak valid.")
        return None

    x, y, w, h = box
    image_with_label = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Gambar kotak dan label
    cv2.rectangle(image_with_label, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image_with_label, f"{predicted_label} ({confidence:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Buat plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Gambar dengan bounding box
    ax[0].imshow(image_with_label)
    ax[0].set_title(f"Prediksi: {predicted_label} (Confidence: {confidence:.2f})")
    ax[0].axis('off')

    # Bar chart probabilitas
    ax[1].bar(class_labels, y_pred_proba, color=['blue', 'orange', 'green', 'red'])
    ax[1].set_title('Probabilitas Prediksi')
    ax[1].set_ylim([0, 1])
    for i, v in enumerate(y_pred_proba):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()

    # Simpan ke buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =========================
# 5. Fungsi Utama
# =========================
def main(image_paths, model_path='ethnicity_classification_best_model.h5', class_labels=['Batak', 'Jawa', 'Padang', 'Sunda']):
    """
    Fungsi utama untuk memprediksi etnis dari daftar gambar.
    """
    # Inisialisasi MTCNN
    detector = MTCNN()
    
    # Muat model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Model berhasil dimuat dari {model_path}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        sys.exit(1)
    
    # Proses setiap gambar
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
            continue
        
        print(f"\n[INFO] Memproses gambar: {image_path}")
        predicted_label, y_pred_proba, image, box = predict_ethnicity(image_path, model, detector, class_labels)
        
        if predicted_label is not None:
            confidence = y_pred_proba[np.argmax(y_pred_proba)]
            print(f"[HASIL PREDIKSI] Etnis: {predicted_label} (Confidence: {confidence:.2f})")
            print(f"[PROBABILITAS] {', '.join([f'{cls}: {prob:.2f}' for cls, prob in zip(class_labels, y_pred_proba)])}")
            visualize_prediction(image, predicted_label, confidence, y_pred_proba, class_labels, box)
        else:
            print(f"[ERROR] Gagal memprediksi etnis untuk gambar: {image_path}")

if __name__ == "__main__":
    # Contoh penggunaan
    image_paths = [
        "nalen_1.jpg",  # Ganti dengan path gambar Anda
        # Tambahkan path gambar lain jika diperlukan
    ]
    
    main(image_paths)