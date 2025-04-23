from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from PIL import Image
import uuid

# Folder untuk menyimpan face crop temporer
TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)

def draw_text_lines(img, x, y, lines, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def simplify_gender(gender):
    if isinstance(gender, dict):
        return max(gender, key=gender.get)
    elif isinstance(gender, str):
        return "Man" if gender.lower() == "man" else "Woman"
    return "Unknown"

def analyze_faces(image_pil):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results_all = {
        'haar': [],
        'mtcnn': [],
        'retinaface': []
    }

    # ----------------------------
    # 1. MTCNN
    # ----------------------------
    mtcnn_img = img_bgr.copy()
    detector = MTCNN()
    mtcnn_faces = detector.detect_faces(img_rgb)

    for i, face in enumerate(mtcnn_faces):
        x, y, w, h = face['box']
        x, y = max(x, 0), max(y, 0)
        cropped = mtcnn_img[y:y+h, x:x+w]
        face_path = os.path.join(TEMP_DIR, f"mtcnn_{uuid.uuid4()}.jpg")
        cv2.imwrite(face_path, cv2.resize(cropped, (224, 224)))

        try:
            analysis = DeepFace.analyze(face_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
            gender = simplify_gender(analysis['gender'])
            age = analysis['age']
            emotion = max(analysis['emotion'], key=analysis['emotion'].get)

            cv2.rectangle(mtcnn_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            draw_text_lines(mtcnn_img, x, y+h+10, [f"Age: {age}", f"Gender: {gender}", f"Emotion: {emotion}"])

            results_all['mtcnn'].append({'age': age, 'gender': gender, 'emotion': emotion})

        except Exception as e:
            print(f"[MTCNN] Error: {e}")

    # ----------------------------
    # 2. Haar Cascade
    # ----------------------------
    haar_img = img_bgr.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    haar_faces = face_cascade.detectMultiScale(haar_img, scaleFactor=1.1, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(haar_faces):
        cropped = haar_img[y:y+h, x:x+w]
        face_path = os.path.join(TEMP_DIR, f"haar_{uuid.uuid4()}.jpg")
        cv2.imwrite(face_path, cv2.resize(cropped, (224, 224)))

        try:
            analysis = DeepFace.analyze(face_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
            gender = simplify_gender(analysis['gender'])
            age = analysis['age']
            emotion = max(analysis['emotion'], key=analysis['emotion'].get)

            cv2.rectangle(haar_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            draw_text_lines(haar_img, x, y+h+10, [f"Age: {age}", f"Gender: {gender}", f"Emotion: {emotion}"])

            results_all['haar'].append({'age': age, 'gender': gender, 'emotion': emotion})

        except Exception as e:
            print(f"[Haar] Error: {e}")

    # ----------------------------
    # 3. RetinaFace
    # ----------------------------
    retina_img = img_bgr.copy()

    try:
        detections = DeepFace.extract_faces(img_path=np.array(image_pil), detector_backend='retinaface', enforce_detection=False)

        for i, face_data in enumerate(detections):
            region = face_data['facial_area']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            x, y = max(x, 0), max(y, 0)

            face_img = face_data['face']
            face_path = os.path.join(TEMP_DIR, f"retina_{uuid.uuid4()}.jpg")
            cv2.imwrite(face_path, cv2.resize(face_img, (224, 224)))

            try:
                analysis = DeepFace.analyze(face_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
                gender = simplify_gender(analysis['gender'])
                age = analysis['age']
                emotion = max(analysis['emotion'], key=analysis['emotion'].get)

                cv2.rectangle(retina_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                draw_text_lines(retina_img, x, y+h+10, [f"Age: {age}", f"Gender: {gender}", f"Emotion: {emotion}"])

                results_all['retinaface'].append({'age': age, 'gender': gender, 'emotion': emotion})
            except Exception as e:
                print(f"[RetinaFace] Analysis error: {e}")

    except Exception as e:
        print(f"[RetinaFace] Detection error: {e}")

    # Convert result images ke RGB utk ditampilkan di Streamlit
        # Siapkan hasil masing-masing deteksi
    haar_result = results_all['haar'][0] if results_all['haar'] else {"age": "-", "gender": "-", "emotion": "-"}
    mtcnn_result = results_all['mtcnn'][0] if results_all['mtcnn'] else {"age": "-", "gender": "-", "emotion": "-"}
    retina_result = results_all['retinaface'][0] if results_all['retinaface'] else {"age": "-", "gender": "-", "emotion": "-"}

    return [
        {
            "image": cv2.cvtColor(haar_img, cv2.COLOR_BGR2RGB),
            "age": haar_result["age"],
            "gender": haar_result["gender"],
            "emotion": haar_result["emotion"],
        },
        {
            "image": cv2.cvtColor(mtcnn_img, cv2.COLOR_BGR2RGB),
            "age": mtcnn_result["age"],
            "gender": mtcnn_result["gender"],
            "emotion": mtcnn_result["emotion"],
        },
        {
            "image": cv2.cvtColor(retina_img, cv2.COLOR_BGR2RGB),
            "age": retina_result["age"],
            "gender": retina_result["gender"],
            "emotion": retina_result["emotion"],
        },
    ]

