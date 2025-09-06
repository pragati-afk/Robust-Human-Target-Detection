import cv2
import os
import pickle
import numpy as np
import face_recognition
from deepface import DeepFace
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ---------------- Load Criminal Database ----------------
CRIMINALS_FOLDER = r"C:\Users\praga\Documents\previous databas\criminals"
EMBEDDINGS_FILE = "criminal_encodings.pkl"

# Load YOLOv8 Face Detection Model
face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(face_model_path)

# Load YOLOv8 Weapon Detection Model
model = YOLO('weapon-detection/runs/detect/train2/weights/best.pt')

def load_criminal_encodings():
    """Loads or computes criminal face encodings."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    
    print("[INFO] Computing criminal face encodings...")
    encodings = {}
    for person_name in os.listdir(CRIMINALS_FOLDER):
        person_path = os.path.join(CRIMINALS_FOLDER, person_name)
        if not os.path.isdir(person_path):
            continue
        encodings[person_name] = []
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)
            face_enc = face_recognition.face_encodings(image)
            if face_enc:
                encodings[person_name].append(face_enc[0])
            else:
                print(f"[WARNING] No face found in {img_path}")
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)
    return encodings

# Load criminal encodings
criminal_encodings = load_criminal_encodings()
print(f"✅ Loaded {len(criminal_encodings)} criminals from folder.")

# ---------------- Real-Time Detection ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces using YOLOv8
    face_results = face_model(frame)
    results = model.predict(frame, stream=True)
    
    for result in face_results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face_roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue
            
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(face_rgb)
            if not face_encodings:
                continue
            face_encoding = face_encodings[0]
            
            # Face Matching
            identity = "Unknown"
            is_criminal = False  
            for person, embeddings in criminal_encodings.items():
                matches = face_recognition.compare_faces(embeddings, face_encoding, tolerance=0.5)
                if True in matches:
                    identity = person
                    is_criminal = True
                    break
            
            # Emotion Detection
            try:
                emotion_result = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                emotion = emotion_result[0]["dominant_emotion"]
            except:
                emotion = "Unknown"
            
            color = (0, 0, 255) if is_criminal else (0, 255, 0)
            label = f"{identity} ({emotion})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            print(f"[DEBUG] Face detected! Matched: {identity} | Emotion: {emotion}")
    
     # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the output
    # cv2.imshow('Weapon Detection', frame)
    # Show Output
    cv2.imshow("Real-Time Security System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
