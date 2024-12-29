from flask import Flask, request, jsonify, render_template, Response
import cv2
import base64
import pickle
import face_recognition as fr
import numpy as np
import os
import pandas as pd
import datetime

app = Flask(__name__)
dir = os.getcwd()
attendance_df = pd.DataFrame(data = None, columns = ['S/N','Name', 'Time_in', 'Time_out'])
today_date = datetime.datetime.now().strftime("%Y-%m-%d")
names = []
save_frame = None

if not os.path.exists(dir + r"/ATTENDANCE_SHEETS"):
    os.mkdir(dir + r"/ATTENDANCE_SHEETS")

if not os.path.exists(dir + r"/LOGS"):
    os.mkdir(dir + r"/LOGS")

#change the path here to the face_encodings path generated from the enrollment project
with open(r'C:\Users\USER\PycharmProjects\FaceEnrollmentApp\face_encodings', 'rb') as f:
    data = pickle.loads(f.read())

def recognise(frame):
    #global save_frame
    reg_names = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = fr.face_locations(rgb_frame)
    encodings = fr.face_encodings(rgb_frame, bboxes)
    for encoding in encodings:
        distances = fr.face_distance(encoding, data["encodings"])
        min_dist = np.min(distances)

        if min_dist > 0.4:
            name = 'unknown'
        else:
            idx = np.argmin(distances)
            name = data['names'][idx]
        reg_names.append(name)

    for (top, right, bottom, left), reg_name in zip(bboxes, reg_names):
        #regz_name = reg_name
        frame = cv2.rectangle(frame, (left, top),
                              (right, bottom), (255, 0, 0))
        frame = cv2.putText(frame, reg_name, (left - 5, top - 5),
                            1, 1, (0, 255, 0))
    #save_frame = frame
    #print(save_frame)
    return frame, reg_names

def sign_in(regz_name):
    global attendance_df
    if regz_name not in attendance_df['Name'].values.tolist():
        i = len(attendance_df.index)
        attendance_df.loc[i] = [i + 1, regz_name, datetime.datetime.now().strftime("%H:%M"),
                                ""]
        attendance_df.to_csv(rf"ATTENDANCE_SHEETS/{today_date}.csv", index=False)
        message = f"Welcome {regz_name}. You have successfully signed in"
    else:
        message = f"Hello {regz_name}, You are already signed in."
    return message

def sign_out(regz_name):
    global attendance_df
    if (regz_name in attendance_df['Name'].values.tolist()) and (
    not (attendance_df.loc[attendance_df['Name'] == regz_name, 'Time_out'].values[0])):
        attendance_df.loc[attendance_df['Name'] == regz_name, 'Time_out'] = datetime.datetime.now().strftime("%H:%M")
        attendance_df.to_csv(rf"ATTENDANCE_SHEETS/{today_date}.csv", index=False)
        message = f"Good bye {regz_name}. You have successfully signed out"
    elif (regz_name not in attendance_df['Name'].values.tolist()):
        message = f"Hello {regz_name}, You have not signed in before."
    else:
        message = f"Hello {regz_name}, You have been signed out already."
    return message

@app.route('/')
def home():
    global attendance_df
    if os.path.exists(rf"ATTENDANCE_SHEETS/{today_date}.csv"):
        attendance_df = pd.read_csv(rf"ATTENDANCE_SHEETS/{today_date}.csv", keep_default_na=False, na_values=['NaN'])
    return render_template('CameraDisplay.html', data=attendance_df, msg=None)

    #return render_template('CameraDisplay.html')

@app.route('/sign', methods = ['POST'])
def sign():
    #print('not_me' in request.form)

        global today_date
        if not os.path.exists(rf"LOGS\{today_date}"):
            os.mkdir(rf"LOGS\{today_date}")
        if request.form['action'] == 'sign_in':
            if request.form['name']:
                reg_name = request.form['name']
                message = sign_in(reg_name)
                if "Welcome" in message:
                    #print(save_frame)
                    cv2.imwrite(rf"LOGS\{today_date}\{reg_name}.jpg", save_frame)
            else:
                if names:
                    for name in names:
                        if name == 'unknown':
                            message = "Unknown face detected. Please enter your name"
                        else:
                            reg_name = name
                            message = sign_in(reg_name)
                else:
                    message = "No face detected. Please, enter a name to sign in!!! "


        elif request.form['action'] == 'sign_out':
            if request.form['name']:
                reg_name = request.form['name']
                message = sign_out(reg_name)
            else:
                if names:
                    for name in names:
                        if name == 'unknown':
                            message = "Unknown face detected. Please enter your name"
                        else:
                            reg_name = name
                            message = sign_out(reg_name)
                else:
                    message = "No face detected. Please, enter a name to sign out!!! "

        return render_template('index.html', data = attendance_df, msg = message)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        # Get the frame data from the request
        result = request.json
        data_url = result.get('frame_data')
        data = data_url.split(',')[1]

        # Decode the base64 data into a binary format
        image_data = base64.b64decode(data)

        # Convert the binary image data to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Read the image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        #recognise(image)

        # Process the frame data as needed (e.g., save it, perform image analysis)
        # Respond with a success message or any relevant data
        return jsonify({'message': 'Frame received successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)
