

from deepface import DeepFace
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import os

# Inisialisasi detektor MTCNN
detector = MTCNN()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka webcam")
    exit()

print("Tekan 'q' untuk keluar")

frame_count = 0
analysis_cache = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    # Setiap 10 frame baru lakukan analisis
    if frame_count % 10 == 0:
        analysis_cache = []  # reset cache
        for i, result in enumerate(results):
            x, y, w, h = result['box']
            x, y = max(x, 0), max(y, 0)

            face_crop = frame[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_crop, (224, 224))
                cv2.imwrite("temp_face.jpg", face_resized)

                result_analysis = DeepFace.analyze(
                    img_path="temp_face.jpg",
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
                print(f"Gagal analisis wajah {i+1}: {e}")

    # Gambar bounding box dan label dari cache
    for data in analysis_cache:
        x, y, w, h = data["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for idx, line in enumerate(data["label"].split('\n')):
            cv2.putText(frame, line, (x, y - 10 - (idx * 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    cv2.imshow("Live Deteksi Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
