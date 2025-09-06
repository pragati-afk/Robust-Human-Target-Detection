import cv2
import smtplib
import time

import pyttsx3
from email.message import EmailMessage

import os
import tempfile

# Set a custom temp directory
custom_temp_dir = "C:/Users/praga/Documents/Restrict/temp_audio"
os.makedirs(custom_temp_dir, exist_ok=True)
tempfile.tempdir = custom_temp_dir  # Tell Python to use this temp folder

# Set FFmpeg path manually
os.environ["FFMPEG_BINARY"] = "C:/Users/praga/Documents/Restrict/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe"
os.environ["PATH"] += os.pathsep + "C:/Users/praga/Documents/Restrict/ffmpeg-7.1-essentials_build/bin"

from pydub import AudioSegment
from pydub.playback import play



# # Set FFmpeg Path for pydub
# os.environ["PATH"] += os.pathsep + "C:/Users/praga/Documents/Restrict/ffmpeg-7.1-essentials_build/bin"

# Load Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Email Configuration (Replace with your details)
EMAIL_ADDRESS = "model.alert.test@gmail.com"  # Your Gmail
EMAIL_PASSWORD = "hiij fuyr mnid bbtb"  # Use App Password, not real password!
TO_EMAIL = "hackblackpearl@gmail.com"  # Recipient email

# Alert Sound File
alert_sound = "race-start-beeps-125125.mp3"

def send_email(image_path):
    """Send an email with the detected face image."""
    msg = EmailMessage()
    msg["Subject"] = "⚠️ Face Detected Alert!"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content("A face was detected! See the attached image.")

    with open(image_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="detected_face.jpg")

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Email error: {e}")

from playsound import playsound

def play_alert():
    """Play the alert sound using playsound instead of pydub."""
    safe_path = r"C:/Users/praga/Documents/Restrict/alert_sound.wav"

    # Convert MP3 to WAV (Only Needed Once)
    if not os.path.exists(safe_path):  
        sound = AudioSegment.from_file("race-start-beeps-125125.mp3", format="mp3")
        sound.export(safe_path, format="wav")

    # Play sound without temp file issues
    playsound(safe_path)



# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Play alert sound
        play_alert()

        # Text-to-speech alert
        engine = pyttsx3.init()
        engine.say("Face detected")
        engine.runAndWait()

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Face Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save image and send email
        img_path = "detected_face.jpg"
        cv2.imwrite(img_path, frame)
        send_email(img_path)

        # Wait before resuming detection
        time.sleep(5)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
