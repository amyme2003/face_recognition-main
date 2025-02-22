from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import os
import subprocess
import cv2
import sqlite3

app = Flask(__name__)
app.secret_key = 'Hi'


os.makedirs("dataset", exist_ok=True)
os.makedirs("recognizer", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
   if request.method == 'POST':
       user_id = request.form.get('id')
       name = request.form.get('name')
       age = request.form.get('age')


       if not user_id or not name or not age:
           flash("All fields are required!", "danger")
           return redirect(url_for('add_user'))


       # Call dataset_creator.py with arguments
       command = f'python dataset_creator.py {user_id} "{name}" {age}'
       subprocess.run(command, shell=True)


       flash(f"User {name} added successfully!", "success")
       return redirect(url_for('home'))
  
   return render_template('add_user.html')


@app.route('/train', methods=['GET'])
def train():
    command = 'python trainer.py'
    subprocess.run(command, shell=True)
    return jsonify({"status": "success", "message": "Training completed successfully!"})

# Real-time face detection video stream
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")
cam = cv2.VideoCapture(0)

def get_profile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM face WHERE id=?", (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

def generate_frames():
    while True:
        success, img = cam.read()
        if not success:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            profile = get_profile(id)

            if profile is not None:
                cv2.putText(img, f"Name: {profile[1]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {conf:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/detect', methods=['GET'])
def detect():
    try:
        result = subprocess.run(['python', 'detect.py'], capture_output=True, text=True)
        output = result.stdout
        return jsonify({"status": "success", "output": output})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
