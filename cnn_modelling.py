import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from mtcnn import MTCNN
import os
from itertools import product

# 1. Preprocessing Gambar Wajah dengan MTCNN
def preprocess_face(image_path, output_size=(224, 224)):
    """
    Preprocess gambar: deteksi wajah dengan MTCNN, crop, resize, dan normalisasi.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak ditemukan: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah dengan MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(image_rgb)
        if len(faces) == 0:
            raise ValueError(f"Tidak ada wajah ditemukan di: {image_path}")
        
        x, y, w, h = faces[0]['box']
        face_image = image_rgb[y:y+h, x:x+w]
        
        face_image = cv2.resize(face_image, output_size)
        face_image = face_image / 255.0
        
        return face_image
    except Exception as e:
        print(f"Error preprocessing {image_path}: {str(e)}")
        return None

# 2. Fungsi untuk membangun model
def build_model(dropout_rate=0.5, learning_rate=1e-3):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Membekukan lapisan base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Fine-tuning model
def fine_tune_model(model, learning_rate=1e-5):
    model.trainable = True
    for layer in model.layers[:50]:  # Bekukan 50 lapisan pertama
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 4. Memuat dataset
data = pd.read_csv('data2.csv')
NUM_CLASSES = len(data['etnis'].unique())  # Jumlah suku (4: Sunda, Jawa, Batak, Padang)

# Periksa distribusi kelas
print("Distribusi kelas di dataset:")
print(data['etnis'].value_counts())

# Periksa path file
invalid_paths = [path for path in data['path'] if not os.path.exists(path)]
if invalid_paths:
    print(f"Invalid paths: {invalid_paths}")

# 5. Parameter gambar
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16  # Kurangi batch size untuk stabilitas

# 6. ImageDataGenerator untuk augmentasi
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator()

# 7. K-fold Cross Validation dan Grid Search
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

param_grid = {
    'dropout_rate': [0.3, 0.5],
    'learning_rate': [1e-3, 5e-4],
    'epochs': [10, 20]
}

best_val_acc = 0
best_model = None
best_params = {}
results = []

# Split data awal untuk test set
train_val_data, test_data = train_test_split(data, test_size=0.1, stratify=data['etnis'], random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_data)):
    print(f"\nFold {fold + 1}/{n_splits}")
    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]

    print("Distribusi kelas di train set:")
    print(train_data['etnis'].value_counts())
    print("Distribusi kelas di validation set:")
    print(val_data['etnis'].value_counts())

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='path',
        y_col='etnis',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        preprocessing_function=preprocess_face
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_data,
        x_col='path',
        y_col='etnis',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        preprocessing_function=preprocess_face
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    for dropout_rate, learning_rate, epochs in product(param_grid['dropout_rate'], 
                                                      param_grid['learning_rate'], 
                                                      param_grid['epochs']):
        print(f"Training with dropout_rate={dropout_rate}, learning_rate={learning_rate}, epochs={epochs}")
        
        model = build_model(dropout_rate=dropout_rate, learning_rate=learning_rate)
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=len(train_data) // BATCH_SIZE,
            validation_steps=len(val_data) // BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Fine-tuning
        model = fine_tune_model(model, learning_rate=1e-5)
        history_fine = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10,  # Fine-tune untuk 10 epoch
            steps_per_epoch=len(train_data) // BATCH_SIZE,
            validation_steps=len(val_data) // BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        val_acc = max(history_fine.history['val_accuracy'])
        print(f"Fold {fold + 1} Validation Accuracy: {val_acc:.4f}")

        results.append({
            'fold': fold + 1,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'val_accuracy': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_params = {'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'epochs': epochs}

        # Plot akurasi dan loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

print(f"\nBest Parameters: {best_params}, Best Validation Accuracy: {best_val_acc:.4f}")

# 8. Evaluasi model terbaik pada test set
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='path',
    y_col='etnis',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    preprocessing_function=preprocess_face
)

test_generator.reset()
y_pred = best_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 9. Hitung metrik evaluasi
accuracy = np.mean(y_pred_classes == y_true)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
print(classification_report(y_true, y_pred_classes, target_names=class_labels))
print(f"Macro-average F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Weighted-average F1-Score: {report['weighted avg']['f1-score']:.4f}")

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)
roc_auc_scores = []
plt.figure(figsize=(8, 6))

for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.show()

print("\nROC-AUC Scores per Class:")
for label, score in zip(class_labels, roc_auc_scores):
    print(f"{label}: {score:.4f}")
print(f"Mean ROC-AUC: {np.mean(roc_auc_scores):.4f}")

# 10. Simpan model terbaik
best_model.save('ethnicity_classification_best_model.h5')