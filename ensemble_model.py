import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
import joblib

# =========================
# 1. Load dan Preprocessing
# =========================
df = pd.read_csv('data2.csv')

# Encode label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['etnis'])
num_classes = len(label_encoder.classes_)

X = []
y = []
groups = []

for idx, row in df.iterrows():
    img = cv2.imread(row['path'])
    if img is None:
        print(f"[WARNING] Gagal memuat gambar: {row['path']}")
        continue
    img = cv2.resize(img, (224, 224))
    X.append(img)
    y.append(row['label'])
    groups.append(row['name'])

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

# Oversampling
ros = RandomOverSampler(random_state=42)
X_reshaped = X.reshape(X.shape[0], -1)
X_resampled, y_resampled = ros.fit_resample(X_reshaped, y)

# Perbarui groups berdasarkan indeks oversampling
# RandomOverSampler mengembalikan X_resampled dan y_resampled dengan indeks baru
# Kita perlu menyesuaikan groups dengan indeks yang sama
# Karena RandomOverSampler menggandakan sampel, kita ambil groups berdasarkan indeks asli
indices = ros.sample_indices_  # Indeks sampel yang dipilih oleh oversampler
groups_resampled = groups[indices]

X = X_resampled.reshape(-1, 224, 224, 3)
y = y_resampled
groups = groups_resampled
y_categorical = to_categorical(y, num_classes)

# Simpan label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# =========================
# 2. Augmentasi Data
# =========================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    channel_shift_range=10
)

# =========================
# 3. Definisi Model
# =========================
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess_resnet

def create_mobilenetv2_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess_mobilenet

def create_efficientnetb0_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess_efficientnet

# Inisialisasi model
models = [
    ("resnet50", create_resnet50_model()),
    ("mobilenetv2", create_mobilenetv2_model()),
    ("efficientnetb0", create_efficientnetb0_model())
]

# Bobot untuk soft voting
weights = {'resnet50': 0.3, 'mobilenetv2': 0.2, 'efficientnetb0': 0.5}

# =========================
# 4. Group K-Fold Cross-Validation
# =========================
kfold = GroupKFold(n_splits=5)
fold_no = 1
accuracies = []
all_y_true = []
all_y_pred = []
all_y_pred_proba = []

# Periksa distribusi individu per fold
print("Distribusi individu per fold:")
for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    print(f"Fold {fold+1} - Train: {train_groups}, Test: {test_groups}")

for train_idx, test_idx in kfold.split(X, y, groups):
    print(f"\n[INFO] Training Fold {fold_no}...")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]
    y_train_labels = y[train_idx]

    # Class weighting
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Prediksi ensemble per fold
    y_pred_proba_fold = np.zeros((len(test_idx), num_classes))
    for model_name, (base_model, preprocess_fn) in models:
        print(f"[INFO] Training {model_name} for Fold {fold_no}...")
        X_train_preprocessed = preprocess_fn(X_train.copy())
        X_test_preprocessed = preprocess_fn(X_test.copy())

        model = base_model
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            datagen.flow(X_train_preprocessed, y_train, batch_size=32),
            validation_data=(X_test_preprocessed, y_test),
            epochs=20,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=1
        )

        # Simpan model per fold
        model.save(f"{model_name}_fold_{fold_no}.keras")

        # Prediksi untuk fold ini
        y_pred_proba = model.predict(X_test_preprocessed)
        y_pred_proba_fold += weights[model_name] * y_pred_proba

    # Normalisasi probabilitas
    y_pred_proba_fold /= sum(weights.values())
    y_pred_classes = np.argmax(y_pred_proba_fold, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Hitung akurasi per fold
    fold_accuracy = np.mean(y_pred_classes == y_true)
    print(f"[INFO] Akurasi Fold {fold_no} (Ensemble): {fold_accuracy*100:.2f}%")
    accuracies.append(fold_accuracy)

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred_classes)
    all_y_pred_proba.extend(y_pred_proba_fold)

    fold_no += 1

# =========================
# 5. Evaluasi Keseluruhan
# =========================
print(f"\n[INFO] Akurasi Rata-rata Cross-Validation (Ensemble): {np.mean(accuracies)*100:.2f}%")

print("\n[INFO] Classification Report (Keseluruhan):")
class_report = classification_report(all_y_true, all_y_pred, target_names=label_encoder.classes_)
print(class_report)

cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Keseluruhan)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

all_y_pred_proba = np.array(all_y_pred_proba)
all_y_true_categorical = to_categorical(all_y_true, num_classes)

print("\n[INFO] ROC-AUC Scores (One-vs-Rest):")
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    roc_auc = roc_auc_score(all_y_true_categorical[:, i], all_y_pred_proba[:, i])
    print(f"Kelas {label_encoder.classes_[i]}: {roc_auc:.2f}")
    
    fpr, tpr, _ = roc_curve(all_y_true_categorical[:, i], all_y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f'ROC Curve {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve (One-vs-Rest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 6. Simpan Model Terakhir
# =========================
for model_name, (base_model, _) in models:
    base_model.save(f"{model_name}_etnis_model.keras")
    print(f"[OK] Model {model_name} berhasil disimpan sebagai '{model_name}_etnis_model.keras'")