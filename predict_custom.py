import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from skimage.feature import local_binary_pattern, hog
import joblib
import tensorflow_addons as tfa
import os

# Load models and encoders once
models = {
    'resnet50': load_model('model/resnet50_etnis_model_v2.keras', custom_objects={'AdamW': tfa.optimizers.AdamW}),
    'efficientnetb0': load_model('model/efficientnetb0_etnis_model_v2.keras', custom_objects={'AdamW': tfa.optimizers.AdamW}),
    'densenet121': load_model('model/densenet121_etnis_model_v2.keras', custom_objects={'AdamW': tfa.optimizers.AdamW}),
    'xception': load_model('model/xception_etnis_model_v2.keras', custom_objects={'AdamW': tfa.optimizers.AdamW}),
}
scaler = joblib.load('model/feature_scaler_v2.pkl')
label_encoder = joblib.load('model/label_encoder_v2.pkl')

preprocess_functions = {
    'resnet50': preprocess_resnet,
    'efficientnetb0': preprocess_efficientnet,
    'densenet121': preprocess_densenet,
    'xception': preprocess_xception
}
weights = {
    'resnet50': 0.3,
    'efficientnetb0': 0.3,
    'densenet121': 0.2,
    'xception': 0.2
}
detector = MTCNN()

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp1 = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp2 = local_binary_pattern(gray, P=16, R=2, method='uniform')
    hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, 18), range=(0, 18))
    hist1 = hist1.astype("float") / (hist1.sum() + 1e-7)
    hist2 = hist2.astype("float") / (hist2.sum() + 1e-7)
    return np.concatenate([hist1, hist2])

def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    hog_features, _ = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return hog_features

def extract_color_histograms(image):
    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    features = np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v])
    return features.astype("float") / (np.sum(features) + 1e-7)

def preprocess_and_extract_features(image):
    img = cv2.resize(image, (224, 224))
    lbp_features = extract_lbp(img)
    hog_features = extract_hog(img)
    color_features = extract_color_histograms(img)
    all_features = np.concatenate([lbp_features, hog_features[:100], color_features[:50]])
    return img, all_features

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

def draw_prediction_box(image, box, label, confidence):
    x, y, w, h = box
    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(annotated, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # Convert for Streamlit


def predict_ethnicity_custom(image_path):
    face_resized, image, box = extract_face(image_path, detector)
    if face_resized is None:
        return None, None, "Wajah tidak terdeteksi", None

    _, features = preprocess_and_extract_features(face_resized)
    features = scaler.transform([features])
    face_resized_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_resized_rgb = np.expand_dims(face_resized_rgb, axis=0)

    y_pred_proba = np.zeros((1, len(label_encoder.classes_)))

    for model_name, model in models.items():
        face_input = preprocess_functions[model_name](face_resized_rgb.copy())
        pred = model.predict([face_input, features], verbose=0)
        y_pred_proba += weights[model_name] * pred

    y_pred_proba /= sum(weights.values())
    y_pred = np.argmax(y_pred_proba, axis=1)
    predicted_label = label_encoder.inverse_transform(y_pred)[0]
    confidence = y_pred_proba[0][y_pred[0]]

    label_list = label_encoder.classes_
    prob_dict = {
        'labels': label_list.tolist(),
        'probs': y_pred_proba[0].tolist()
    }

    # Anotasi gambar
    annotated_image = draw_prediction_box(image, box, predicted_label, confidence)

    return predicted_label, confidence, prob_dict, annotated_image
