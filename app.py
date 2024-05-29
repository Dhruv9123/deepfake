from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import dlib
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

detector = dlib.get_frontal_face_detector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces, img

def extract_face(img, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    return img[y:y+h, x:x+w]

def swap_faces(img, face1, face2):
    face1_img = extract_face(img, face1)
    face2_img = extract_face(img, face2)

    face1_resized = cv2.resize(face1_img, (face2.width(), face2.height()))
    face2_resized = cv2.resize(face2_img, (face1.width(), face1.height()))

    img[face1.top():face1.top()+face1.height(), face1.left():face1.left()+face1.width()] = face2_resized
    img[face2.top():face2.top()+face2.height(), face2.left():face2.left()+face2.width()] = face1_resized

    return img

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('process_file', filename=filename))
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process/<filename>')
def process_file(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    faces, img = detect_faces(image_path)

    if len(faces) < 2:
        return "Need at least two faces to swap", 400

    swapped_img = swap_faces(img, faces[0], faces[1])
    
    result_filename = 'result_' + filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, swapped_img)

    return redirect(url_for('uploaded_file', filename=result_filename))

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
