import cv2
import face_recognition as fr
import pickle
import numpy as np
from imutils import paths
import os
from flask import Flask, render_template, request, Response

app = Flask(__name__)

data = {}
dir = os.getcwd()
cap = cv2.VideoCapture(0)
save_frame = None

if not os.path.exists(dir+r'/Images'):
    os.mkdir(dir+r'/Images')
def enrollment (n):
    #print(n)
    path = rf"Images/{n}"
    known_names = []
    known_encodings = []
    for image_filepath in paths.list_images(path):
        #print(image_filepath)
        bgr_img = cv2.imread(image_filepath)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        faces = fr.face_locations(rgb_img)
        encodings = fr.face_encodings(rgb_img, faces)
        for encoding in encodings:
            known_names.append(n)
            known_encodings.append(encoding)
    #face_data = {'names': known_names, 'encodings': known_encodings}
    #print(face_data)

    if os.path.exists(dir + r"/face_encodings"):
        with open(dir + r"/face_encodings", 'rb') as f:
            face_data = pickle.loads(f.read())
            face_data['encodings'].extend(known_encodings)
            face_data['names'].extend(known_names)

        with open(dir + r"/face_encodings", 'wb') as f:
            f.write(pickle.dumps(face_data))

    else:
        face_data = {'names': known_names, 'encodings': known_encodings}
        with open(dir + r"/face_encodings", 'wb') as f:
            f.write(pickle.dumps(face_data))

    return "Enrollment successfully completed"


def generate_frames():
    global save_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            save_frame = frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('enrollmentpage.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enroll', methods = ['POST'])
def enroll():
    if request.form['name']:
        name = request.form['name']
        os.mkdir(f'Images/{name}')
        for i in range(20):
            cv2.imwrite(f"Images/{name}/{name}-{i}.jpg", save_frame)

        message = enrollment(name)
    else:
        message = "Please enter a name for enrollment"
    return render_template('enrollmentpage.html', msg = message)

if __name__ == '__main__':
    app.run(debug=True)
