import cv2
import numpy as np
import sqlite3
import sys

# Load the Haar cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insert_or_update(Id, Name, age):
    conn = sqlite3.connect("sqlite.db")
    cmd = "SELECT * FROM face WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        # Update existing record in the "face" table
        conn.execute("UPDATE face SET NAME=?, AGE=? WHERE ID=?", (Name, age, Id))
    else:
        # Insert new record into the "face" table
        conn.execute("INSERT INTO face (ID, NAME, AGE) VALUES (?, ?, ?)", (Id, Name, age))
    conn.commit()
    conn.close()

# Command-line arguments for ID, Name, and Age
if len(sys.argv) != 4:
    print("Usage: python dataset_creator.py <ID> <Name> <Age>")
    sys.exit(1)

Id = sys.argv[1]
Name = sys.argv[2]
age = sys.argv[3]

insert_or_update(Id, Name, age)

sampleNum = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"dataset/user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)

    cv2.imshow("Face", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break
    if sampleNum >= 100:  # Capture up to 100 samples
        break

cam.release()
cv2.destroyAllWindows()