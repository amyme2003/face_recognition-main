from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import subprocess

app = Flask(__name__)
app.secret_key = 'Hi'

# Ensure required directories exist
os.makedirs("dataset", exist_ok=True)
os.makedirs("recognizer", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # Renders HTML home page

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
        command = f'python dataset_creator.py {user_id} \"{name}\" {age}'
        subprocess.run(command, shell=True)

        flash(f"User {name} added successfully!", "success")
        return redirect(url_for('home'))
    
    return render_template('add_user.html')  # Form to add a user

@app.route('/train')
def train():
    # Call trainer.py
    command = 'python trainer.py'
    subprocess.run(command, shell=True)
    flash("Training completed successfully!", "success")
    return redirect(url_for('home'))

@app.route('/detect')
def detect():
    # Call detect.py
    command = 'python detect.py'
    subprocess.run(command, shell=True)
    return redirect(url_for('home'))

# API for real-time face detection (for Flutter app)
@app.route('/detect_faces', methods=['GET'])
def detect_faces():
    try:
        result = subprocess.run(['python', 'detect.py'], capture_output=True, text=True)
        output = result.stdout  # Get detect.py's output

        return jsonify({"status": "success", "output": output})  # Send output to Flutter
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
