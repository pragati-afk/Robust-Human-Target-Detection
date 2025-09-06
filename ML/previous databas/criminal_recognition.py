import cv2
import face_recognition
import os
import numpy as np

# Path to the criminals dataset
CRIMINALS_FOLDER = r"C:\Users\praga\Documents\previous databas\criminals"


# Load all criminal images and encode faces
criminal_encodings = []
criminal_names = []

for person in os.listdir(CRIMINALS_FOLDER):
    person_path = os.path.join(CRIMINALS_FOLDER, person)
    if os.path.isdir(person_path):  # Ensure it's a folder
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                criminal_encodings.append(encodings[0])
                criminal_names.append(person)  # Use folder name as the person's name

print(f"✅ Loaded {len(criminal_encodings)} criminals from folder.")

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(criminal_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = criminal_names[match_index]

        # Draw a rectangle around the face
        color = (0, 0, 255) if name != "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Criminal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
