import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import pandas as pd
from sklearn.model_selection import train_test_split


class CosineSimilarity(Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        dot_product = K.sum(x * y, axis=1, keepdims=True)
        norm_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
        norm_y = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True))
        return dot_product / (norm_x * norm_y + K.epsilon())

# --- Model Embedding ---
def create_embedding_model():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(128)(x)  # embedding size
    model = Model(inputs, outputs)
    return model

# --- Model Siamese dengan Cosine Similarity ---
def create_siamese_model(base_model):
    inputA = Input(shape=(224, 224, 3))
    inputB = Input(shape=(224, 224, 3))
    embeddingA = base_model(inputA)
    embeddingB = base_model(inputB)
    similarity = CosineSimilarity()([embeddingA, embeddingB])
    model = Model(inputs=[inputA, inputB], outputs=similarity)
    return model

# --- Inisialisasi detektor MTCNN ---
detector = MTCNN()

# --- Fungsi Deteksi dan Crop Wajah (224x224) ---
def detect_and_crop_face(img_path):
    img = load_img(img_path)
    img_array = img_to_array(img)
    faces = detector.detect_faces(img_array)
    
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        cropped = img.crop((x, y, x + w, y + h)).resize((224, 224))
        return img_to_array(cropped) / 255.0
    return None

# --- Load Dataset dengan Deteksi dan Crop Wajah ---
def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['path', 'name', 'ethnicity'])

    images, labels_name, labels_ethnicity = [], [], []

    for index, row in df.iterrows():
        img_path = row['path']
        name = row['name']
        ethnicity = row['ethnicity'].strip().strip(';')

        if os.path.exists(img_path):
            cropped_img = detect_and_crop_face(img_path)
            if cropped_img is not None:
                images.append(cropped_img)
                labels_name.append(name)
                labels_ethnicity.append(ethnicity)
                print(f"Loaded and cropped: {img_path}")
            else:
                print(f"Face not detected: {img_path}")
        else:
            print(f"File not found: {img_path}")

    return np.array(images), np.array(labels_name), np.array(labels_ethnicity)

# --- Fungsi Batch Generator ---
def create_pairs_batch(images, labels, batch_size=16):
    n = len(images)
    pairs = []
    pair_labels = []

    for i in range(0, n, batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        for i in range(len(batch_images)):
            for j in range(i + 1, len(batch_images)):
                pairs.append([batch_images[i], batch_images[j]])
                pair_labels.append(1 if batch_labels[i] == batch_labels[j] else 0)

                if len(pairs) >= batch_size:
                    yield np.array(pairs), np.array(pair_labels)
                    pairs, pair_labels = [], []

# Load all data
all_df = pd.read_csv("data.csv", header=None, names=["path", "name", "ethnicity"])

# Hitung jumlah kemunculan setiap nama
name_counts = all_df["name"].value_counts()

# Hanya ambil data yang nama-nya muncul minimal 2 kali
filtered_df = all_df[all_df["name"].isin(name_counts[name_counts >= 2].index)]

# Stratified split menjadi 70% train dan 30% sementara (val + test)
train_df, temp_df = train_test_split(filtered_df, test_size=0.30, random_state=42, stratify=filtered_df["name"])

# Split 30% tadi menjadi 15% val dan 15% test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["name"])

# Simpan ke CSV
train_df.to_csv("data_training.csv", index=False, header=False)
val_df.to_csv("data_validate.csv", index=False, header=False)
test_df.to_csv("data_test.csv", index=False, header=False)

# --- Load Data dari Tiga CSV ---
images_train, labels_name_train, _ = load_dataset_from_csv('data_training.csv')
images_val, labels_name_val, _ = load_dataset_from_csv('data_validate.csv')
images_test, labels_name_test, _ = load_dataset_from_csv('data_test.csv')

# --- Train Model ---
embedding_model = create_embedding_model()
siamese_model = create_siamese_model(embedding_model)
siamese_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

for batch_pairs, batch_labels in create_pairs_batch(images_train, labels_name_train, batch_size=16):
    siamese_model.fit([batch_pairs[:, 0], batch_pairs[:, 1]], batch_labels, epochs=1, batch_size=16)

embeddings = embedding_model.predict(images_test)
labels_name = labels_name_test
images = images_test

# --- t-SNE Visualisasi ---
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
for name in np.unique(labels_name):
    idx = np.where(labels_name == name)
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=name)
plt.legend(loc='upper right', fontsize=6)
plt.title("t-SNE Visualisasi Berdasarkan Nama Individu")
plt.savefig("tsne_visualization_based_on_names.png")
plt.show()

# --- ROC Curve dan EER ---
distances = []
all_labels = []

for batch_pairs, batch_labels in create_pairs_batch(images_test, labels_name_test, batch_size=16):
    batch_similarities = siamese_model.predict([batch_pairs[:, 0], batch_pairs[:, 1]]).ravel()
    distances.extend(batch_similarities)
    all_labels.extend(batch_labels)

distances = np.array(distances)
all_labels = np.array(all_labels)

fpr, tpr, thresholds = roc_curve(all_labels, distances)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve_face_verification_cosine.png")
plt.show()

# --- Threshold Optimal (EER) ---
fnr = 1 - tpr
eer_index = np.nanargmin(np.abs(fnr - fpr))
eer_threshold = thresholds[eer_index]

print(f"EER Threshold: {eer_threshold:.4f}")

def classify_by_threshold(distances, threshold):
    return (distances >= threshold).astype(int)

y_pred = classify_by_threshold(distances, eer_threshold)
precision = precision_score(all_labels, y_pred, zero_division=0)
recall = recall_score(all_labels, y_pred, zero_division=0)
f1 = f1_score(all_labels, y_pred, zero_division=0)
accuracy = accuracy_score(all_labels, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"ROC_AUC_Score_OvR: {roc_auc:.4f}")
print(f"EER: {fpr[eer_index]:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"TAR: {tpr[eer_index]:.4f}")
print(f"FAR: {fpr[eer_index]:.4f}")
print(f"FRR: {fnr[eer_index]:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1_Score: {f1:.4f}")
print(f"EER_Threshold: {eer_threshold:.4f}")
print(f"Accuracy: {accuracy:.4f}")

embedding_model.save('embedding_model.h5')
siamese_model.save('siamese_cosine_model.h5')
with open('threshold.txt', 'w') as f:
    f.write(str(eer_threshold))
