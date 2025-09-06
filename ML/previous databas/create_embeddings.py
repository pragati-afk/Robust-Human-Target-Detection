import os
import pickle
import face_recognition

# Path to the criminals dataset
CRIMINALS_FOLDER = r"C:\Users\praga\Documents\previous databas\criminals"
EMBEDDINGS_FILE = r"C:\Users\praga\Documents\previous databas\criminal_encodings.pkl"

# Dictionary to store encodings
criminal_encodings = {}
print("[INFO] Generating criminal face encodings...")

# Process each folder (person)
for person in os.listdir(CRIMINALS_FOLDER):
    person_path = os.path.join(CRIMINALS_FOLDER, person)
    if os.path.isdir(person_path):  
        encodings = []

        # Process each image
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)
            face_enc = face_recognition.face_encodings(image)

            if face_enc:
                encodings.append(face_enc[0])  # Store first face encoding
        
        if encodings:
            criminal_encodings[person] = encodings

# Save encodings to a pickle file
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(criminal_encodings, f)

print(f"✅ Saved {len(criminal_encodings)} criminals' encodings in '{EMBEDDINGS_FILE}'.")
