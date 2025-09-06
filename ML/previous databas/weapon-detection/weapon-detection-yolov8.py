import os
import cv2
import numpy as np
import wandb
from ultralytics import YOLO

# Install required packages (run these in a separate cell if needed)
# !pip install ultralytics wandb tensorflow==2.13.1 opencv-python

# Login to Weights & Biases manually (replace 'your_wandb_api_key' with actual key)
wandb.login(key='9366a6e617f664f3953010d6986ccf96405cee3d')

# Define dataset path
data_yaml_path = 'C:\\Users\\praga\\Downloads\\weapon-detection\\data.yaml'

# Load YOLO model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data=data_yaml_path, epochs=20, imgsz=640)

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()

# Open webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model.predict(frame)
    
    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('YOLO Live Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Export the model
model.export(format='tflite')