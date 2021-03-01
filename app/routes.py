from app import app
from flask import render_template
from flask import request
from flask import jsonify
from app.prediction.face_detector import FaceDetector
import urllib.parse

# import cv2
# import numpy as np
# import base64


face_detector = FaceDetector()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    req_data = request.get_json()

    # image_data = urllib.parse.unquote(req_data['image_data'])
    image_data = req_data['image_data']

    print("CALLING FACE DETECTION")

    faces = face_detector.detect_faces(image_data)

    print(faces)

    # encoded_data = image_data.split(',')[1]

    # nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # # sbuf = io.StringIO()
    # # sbuf.write(base64.b64decode(encoded_data).decode("utf-8"))
    # # pimg = Image.open(sbuf)
    # # img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    # cv2.imwrite('tmp.png', img)
    # print(image_data)

    return jsonify(faces)