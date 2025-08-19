import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/emotion_model.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48)) / 255.0
        roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

