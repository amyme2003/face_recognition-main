import cv2
import numpy as np
import sqlite3

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def get_profile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM face WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        continue  # Skip the iteration if no faces are detected

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        profile = get_profile(id)
        
        # Print confidence score to the console
        print(f"ID: {id}, Confidence: {conf}")

        if profile is not None:
            cv2.putText(img, "Name: " + str(profile[1]), org=(x, y + h + 20), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 127), thickness=2)
            cv2.putText(img, "Age: " + str(profile[2]), org=(x, y + h + 45), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 127), thickness=2)
            # Display confidence on the video feed
            cv2.putText(img, f"Confidence: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

    cv2.imshow("FACE", img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
