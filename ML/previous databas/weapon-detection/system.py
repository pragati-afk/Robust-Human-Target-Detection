import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os
import time

def recognize_face(frame, known_faces):
    try:
        results = DeepFace.find(frame, db_path="criminals_db", enforce_detection=False)
        if results:
            return results[0]['identity'], True  # Return matched identity and True for criminal found
    except:
        pass
    return None, False

def detect_emotions(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result:
            return result[0]['dominant_emotion']
    except:
        pass
    return "Unknown"

# Load the YOLO weapon detection model
weapon_model = YOLO('runs/detect/train/weights/best.pt')

# Open webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces and recognize criminals
    face_identity, is_criminal = recognize_face(frame, "criminals_db")
    emotion = detect_emotions(frame)
    
    # Detect weapons
    weapon_results = weapon_model.predict(frame, stream=True)
    
    for result in weapon_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{weapon_model.names[cls]}: {conf:.2f}"
            
            # Draw bounding box for weapon detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Overlay face recognition & emotion analysis results
    if is_criminal:
        cv2.putText(frame, f"CRIMINAL: {face_identity}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Emotion: {emotion}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Show the output
    cv2.imshow('Integrated Security System', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
