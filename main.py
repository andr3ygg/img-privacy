from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
import pytesseract
import re
from urllib.request import urlretrieve
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_DIR = "models"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def download_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(PROTOTXT_PATH):
        urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.isfile(MODEL_PATH):
        urlretrieve(MODEL_URL, MODEL_PATH)

def detect_and_blur_faces(image, net):
    h0, w0 = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            k = max(15, ((x2 - x1) // 3) | 1)
            image[y1:y2, x1:x2] = cv2.GaussianBlur(face, (k, k), 30)
    return image

def blur_sensitive_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    patterns = [
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        r"\+?\d[\d\s\-]{8,}",
        r"\d{1,4}\s\w+\s(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road)",
        r"\b\d{8,12}\b"
    ]

    for i in range(len(data['text'])):
        word = data['text'][i]
        if any(re.search(p, word) for p in patterns):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (15, 15), 30)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        blur_faces = 'blur_faces' in request.form
        blur_info = 'blur_info' in request.form

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            download_model_files()
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

            if blur_faces:
                image = detect_and_blur_faces(image, net)
            if blur_info:
                image = blur_sensitive_text(image)

            result_path = os.path.join(RESULT_FOLDER, filename)
            cv2.imwrite(result_path, image)
            return redirect(url_for('result', filename=filename))

    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
