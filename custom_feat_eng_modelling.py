import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate, Input, Conv2D, MaxPooling2D, Flatten, Add, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0, DenseNet121, InceptionV3, Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
import joblib
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1_l2
import tensorflow_addons as tfa

# Set memory growth for GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"[INFO] GPU is available: {len(physical_devices)} devices")
else:
    print("[INFO] No GPU detected, using CPU")

# Create directories for saving models and results
os.makedirs('models_v2', exist_ok=True)
os.makedirs('results_v2', exist_ok=True)

# =========================
# 1. Enhanced Feature Engineering
# =========================
def extract_lbp(image):
    """Extract uniform LBP features from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use multiple LBP radii for multi-scale analysis
    lbp1 = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp2 = local_binary_pattern(gray, P=16, R=2, method='uniform')
    
    # Histogram features for each scale
    hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, 18), range=(0, 18))
    
    # Normalize histograms
    hist1 = hist1.astype("float") / (hist1.sum() + 1e-7)
    hist2 = hist2.astype("float") / (hist2.sum() + 1e-7)
    
    # Concatenate features
    return np.concatenate([hist1, hist2])

def extract_hog(image):
    """Extract HOG features from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))  # Smaller size for HOG
    
    # Calculate HOG features
    hog_features, hog_image = hog(
        resized, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True,
        block_norm='L2-Hys'
    )
    
    return hog_features

def extract_color_histograms(image):
    """Extract color histograms from various color spaces"""
    # RGB histograms
    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
    
    # HSV histograms
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    
    # Normalize histograms
    features = np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v])
    features = features.astype("float") / (np.sum(features) + 1e-7)
    
    return features

def preprocess_and_extract_features(image_path):
    """Process image and extract multiple features"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Resize image
    img = cv2.resize(img, (224, 224))
    
    # Extract multiple features
    lbp_features = extract_lbp(img)
    hog_features = extract_hog(img)
    color_features = extract_color_histograms(img)
    
    # Combine all features
    all_features = np.concatenate([lbp_features, hog_features[:100], color_features[:50]])
    
    return img, all_features

# =========================
# 2. Advanced Data Loading and Preprocessing
# =========================
print("[INFO] Loading dataset...")
df = pd.read_csv('data2.csv')

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['etnis'])
num_classes = len(label_encoder.classes_)
print(f"[INFO] Found {num_classes} classes: {label_encoder.classes_}")

# Prepare data structures
X_images = []
X_features = []
y = []
groups = []
failed_images = []

for idx, row in df.iterrows():
    if idx % 100 == 0:
        print(f"[INFO] Processing image {idx}/{len(df)}")
    
    img, custom_features = preprocess_and_extract_features(row['path'])
    if img is None or custom_features is None:
        failed_images.append(row['path'])
        continue
    
    X_images.append(img)
    X_features.append(custom_features)
    y.append(row['label'])
    groups.append(row['name'])

if failed_images:
    print(f"[WARNING] Failed to process {len(failed_images)} images")

X_images = np.array(X_images)
X_features = np.array(X_features)
y = np.array(y)
groups = np.array(groups)

# Standardize features
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)
joblib.dump(scaler, 'models_v2/feature_scaler_v2.pkl')

# Advanced resampling using SMOTE-Tomek links
print("[INFO] Performing advanced class balancing...")
X_images_reshaped = X_images.reshape(X_images.shape[0], -1)
smt = SMOTETomek(random_state=42)
X_images_resampled, y_resampled = smt.fit_resample(X_images_reshaped, y)
X_features_resampled, _ = smt.fit_resample(X_features, y)

# Keep track of which samples were used for resampling
try:
    indices = smt.sample_indices_
    groups_resampled = groups[indices]
except:
    # If indices are not available, use original groups (this is a fallback)
    groups_resampled = np.array([f"synthetic_{i}" if i >= len(groups) else groups[i] for i in range(len(y_resampled))])

X_images = X_images_resampled.reshape(-1, 224, 224, 3)
X_features = X_features_resampled
y = y_resampled
groups = groups_resampled
y_categorical = to_categorical(y, num_classes)

# Save preprocessed data and label encoder
joblib.dump(label_encoder, 'models_v2/label_encoder_v2.pkl')
print(f"[INFO] Data preprocessing complete. Images: {X_images.shape}, Features: {X_features.shape}")

# =========================
# 3. Enhanced Data Augmentation
# =========================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    channel_shift_range=10,
    fill_mode='reflect',
    rescale=None  # We handle preprocessing separately
)

# =========================
# 4. Define Advanced Models with Feature Fusion
# =========================
def create_resnet50_model(feature_dim):
    """Create a ResNet50-based model with custom feature fusion"""
    image_input = Input(shape=(224, 224, 3))
    feature_input = Input(shape=(feature_dim,))
    
    # Use a more recent pretrained ResNet50 with fine-tuning
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Gradually unfreeze layers for fine-tuning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Process handcrafted features with dedicated layers
    f = Dense(128, activation='relu')(feature_input)
    f = BatchNormalization()(f)
    f = Dropout(0.4)(f)
    
    # Fusion strategy: concatenate CNN features with handcrafted features
    combined = Concatenate()([x, f])
    
    # Multi-layer fusion network
    combined = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    
    combined = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    
    predictions = Dense(num_classes, activation='softmax')(combined)
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    return model, preprocess_resnet

def create_efficientnetb0_model(feature_dim):
    """Create an EfficientNetB0-based model with custom feature fusion"""
    image_input = Input(shape=(224, 224, 3))
    feature_input = Input(shape=(feature_dim,))
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Fine-tune only the top layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Process handcrafted features
    f = Dense(128, activation='relu')(feature_input)
    f = BatchNormalization()(f)
    f = Dropout(0.4)(f)
    
    # Combine features
    combined = Concatenate()([x, f])
    
    # Deep fusion network
    combined = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    
    combined = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    
    predictions = Dense(num_classes, activation='softmax')(combined)
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    return model, preprocess_efficientnet

def create_densenet_model(feature_dim):
    """Create a DenseNet121-based model with custom feature fusion"""
    image_input = Input(shape=(224, 224, 3))
    feature_input = Input(shape=(feature_dim,))
    
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Fine-tune only the top layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Process handcrafted features
    f = Dense(128, activation='relu')(feature_input)
    f = BatchNormalization()(f)
    f = Dropout(0.4)(f)
    
    # Combine features
    combined = Concatenate()([x, f])
    
    # Deep fusion network
    combined = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    
    combined = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    
    predictions = Dense(num_classes, activation='softmax')(combined)
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    return model, preprocess_densenet

def create_xception_model(feature_dim):
    """Create an Xception-based model with custom feature fusion"""
    image_input = Input(shape=(224, 224, 3))
    feature_input = Input(shape=(feature_dim,))
    
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Fine-tune only the top layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Process handcrafted features
    f = Dense(128, activation='relu')(feature_input)
    f = BatchNormalization()(f)
    f = Dropout(0.4)(f)
    
    # Combine features
    combined = Concatenate()([x, f])
    
    # Deep fusion network
    combined = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    
    combined = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    
    predictions = Dense(num_classes, activation='softmax')(combined)
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    return model, preprocess_xception

# Initialize models
feature_dim = X_features.shape[1]
print(f"[INFO] Feature dimension: {feature_dim}")

models = [
    ("resnet50", create_resnet50_model(feature_dim)),
    ("efficientnetb0", create_efficientnetb0_model(feature_dim)),
    ("densenet121", create_densenet_model(feature_dim)),
    ("xception", create_xception_model(feature_dim))
]

# Weights for weighted ensemble voting (can be tuned)
weights = {
    'resnet50': 0.3, 
    'efficientnetb0': 0.3, 
    'densenet121': 0.2, 
    'xception': 0.2
}

# =========================
# 5. Advanced Training Setup
# =========================
# Advanced callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=0  # Reduced verbosity
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=0  # Reduced verbosity
)

# =========================
# 6. Group K-Fold Cross-Validation with Improved Training
# =========================
kfold = GroupKFold(n_splits=5)
fold_no = 1
accuracies = []
all_y_true = []
all_y_pred = []
all_y_pred_proba = []

print("\n[INFO] Starting cross-validation...")
print("Distribution of individuals per fold:")
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_images, y, groups)):
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    print(f"Fold {fold+1} - Train: {len(train_groups)} subjects, Test: {len(test_groups)} subjects")

# Each fold training
for train_idx, test_idx in kfold.split(X_images, y, groups):
    print(f"\n[INFO] Training Fold {fold_no}...")
    X_train_images, X_test_images = X_images[train_idx], X_images[test_idx]
    X_train_features, X_test_features = X_features[train_idx], X_features[test_idx]
    y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]
    y_train_labels = y[train_idx]

    # Compute class weights for handling imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Print class distribution
    print(f"[INFO] Class distribution in training set: {np.bincount(y_train_labels)}")

    # Ensemble predictions per fold
    y_pred_proba_fold = np.zeros((len(test_idx), num_classes))
    
    # Train each model in the ensemble
    for model_name, (model_builder, preprocess_fn) in models:
        print(f"[INFO] Training {model_name} for Fold {fold_no}/{kfold.n_splits}...")
        model_checkpoint = ModelCheckpoint(
            f"models_v2/{model_name}_fold_{fold_no}_v2.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0  # Reduced verbosity
        )
        
        # Preprocess images according to the model's requirements
        X_train_preprocessed = preprocess_fn(X_train_images.copy())
        X_test_preprocessed = preprocess_fn(X_test_images.copy())

        # Build and compile model
        model = model_builder
        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=1e-4, 
                weight_decay=1e-5
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Train the model with reduced output verbosity
        history = model.fit(
            datagen.flow([X_train_preprocessed, X_train_features], y_train, batch_size=32),
            validation_data=([X_test_preprocessed, X_test_features], y_test),
            epochs=50,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=0  # Reduced verbosity - no per-epoch output
        )

        # Load the best model from checkpoint
        model = load_model(f"models_v2/{model_name}_fold_{fold_no}_v2.keras")

        # Evaluate the model
        scores = model.evaluate([X_test_preprocessed, X_test_features], y_test, verbose=0)
        print(f"[INFO] {model_name} - Fold {fold_no} Accuracy: {scores[1]*100:.2f}%")

        # Predict for this fold
        y_pred_proba = model.predict([X_test_preprocessed, X_test_features], verbose=0)
        
        # Apply model weight
        y_pred_proba_fold += weights[model_name] * y_pred_proba
        
        # Plot training history for this model
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} - Training and Validation Accuracy (Fold {fold_no})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Training and Validation Loss (Fold {fold_no})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results_v2/{model_name}_fold_{fold_no}_training_curves_v2.png')
        plt.close()

    # Normalize probabilities for ensemble
    y_pred_proba_fold /= sum(weights.values())
    y_pred_classes = np.argmax(y_pred_proba_fold, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate fold accuracy
    fold_accuracy = np.mean(y_pred_classes == y_true)
    print(f"[INFO] Fold {fold_no} Accuracy (Ensemble): {fold_accuracy*100:.2f}%")
    accuracies.append(fold_accuracy)

    # Save fold results
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred_classes)
    all_y_pred_proba.extend(y_pred_proba_fold)

    # Fold-specific evaluation - with fixed classification report handling
    print(f"[INFO] Classification Report (Fold {fold_no}):")
    # Get the classes that are actually present in the test data
    present_classes = np.unique(y_true)
    present_class_names = [label_encoder.classes_[i] for i in present_classes]
    
    # Generate report using only the classes present in this fold
    fold_report = classification_report(y_true, y_pred_classes, 
                                     labels=present_classes,
                                     target_names=present_class_names)
    print(fold_report)
    
    # Fold-specific confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes, labels=present_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_class_names, 
                yticklabels=present_class_names)
    plt.title(f'Confusion Matrix (Fold {fold_no})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'results_v2/confusion_matrix_fold_{fold_no}_v2.png')
    plt.close()

    fold_no += 1

# =========================
# 7. Overall Evaluation
# =========================
print(f"\n[INFO] Average Cross-Validation Accuracy (Ensemble): {np.mean(accuracies)*100:.2f}%")

print("\n[INFO] Classification Report (Overall):")
# Get unique classes in all predictions
unique_classes = np.unique(all_y_true)
present_class_names = [label_encoder.classes_[i] for i in unique_classes]

class_report = classification_report(all_y_true, all_y_pred, 
                                  labels=unique_classes,
                                  target_names=present_class_names)
print(class_report)

# Save report to file
with open('results_v2/classification_report_v2.txt', 'w') as f:
    f.write(f"Average Cross-Validation Accuracy: {np.mean(accuracies)*100:.2f}%\n\n")
    f.write(class_report)

# Overall confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=present_class_names, 
            yticklabels=present_class_names)
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('results_v2/confusion_matrix_overall_v2.png')
plt.close()

# ROC curve analysis
all_y_pred_proba = np.array(all_y_pred_proba)
all_y_true_categorical = to_categorical(all_y_true, num_classes)

print("\n[INFO] ROC-AUC Scores (One-vs-Rest):")
plt.figure(figsize=(10, 8))
auc_scores = []

# Only iterate over classes that are actually present in the test data
for i in unique_classes:
    roc_auc = roc_auc_score(all_y_true_categorical[:, i], all_y_pred_proba[:, i])
    auc_scores.append(roc_auc)
    print(f"Class {label_encoder.classes_[i]}: {roc_auc:.4f}")
    
    fpr, tpr, _ = roc_curve(all_y_true_categorical[:, i], all_y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f'ROC {label_encoder.classes_[i]} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curves (One-vs-Rest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('results_v2/roc_curves_v2.png')
plt.close()

# Save mean AUC score
mean_auc = np.mean(auc_scores)
print(f"[INFO] Mean ROC-AUC: {mean_auc:.4f}")
with open('results_v2/roc_auc_scores_v2.txt', 'w') as f:
    f.write(f"Mean ROC-AUC: {mean_auc:.4f}\n\n")
    for i, class_idx in enumerate(unique_classes):
        f.write(f"Class {label_encoder.classes_[class_idx]}: {auc_scores[i]:.4f}\n")

# =========================
# 8. Save Final Models
# =========================
print("\n[INFO] Training final models on all data...")

# Check dimensions to ensure they match
print(f"[INFO] X_images shape: {X_images.shape}, X_features shape: {X_features.shape}, y_categorical shape: {y_categorical.shape}")

# Make sure arrays have the same first dimension
if X_images.shape[0] != X_features.shape[0]:
    # Find the minimum length to use for consistency
    min_len = min(X_images.shape[0], X_features.shape[0])
    X_images = X_images[:min_len]
    X_features = X_features[:min_len]
    y_categorical = y_categorical[:min_len]
    y = y[:min_len]
    print(f"[INFO] Adjusted data shapes - X_images: {X_images.shape}, X_features: {X_features.shape}")

# Train a final model on all data for each architecture
for model_name, (model_builder, preprocess_fn) in models:
    print(f"[INFO] Training final {model_name} model...")
    
    # Preprocess all images
    X_images_preprocessed = preprocess_fn(X_images.copy())
    
    # Build and compile model
    model = model_builder
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate class weights for the entire dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Train the model on all data with reduced verbosity
    history = model.fit(
        [X_images_preprocessed, X_features], y_categorical,  # Direct input, not using datagen
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1  # Increased verbosity to see progress
    )
    
    # Save the final model
    model.save(f"models_v2/{model_name}_etnis_model_v2.keras")
    print(f"[OK] Model {model_name} saved as '{model_name}_etnis_model_v2.keras'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title(f'{model_name} - Final Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'{model_name} - Final Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results_v2/{model_name}_final_training_curves_v2.png')
    plt.close()