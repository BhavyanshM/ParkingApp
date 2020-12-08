#!/usr/bin/env python
import os
import time

import cv2
import numpy as np
from flask import Flask, render_template, Response, redirect
from flask_bootstrap import Bootstrap

desktopVideo = "../Videos/ParkingLotKCropped.mp4"
ultra96Video = "/home/xilinx/jupyter_notebooks/pynq-dpu/video/parking_lot_k_01.mp4"
resultVideo = "../output/result.avi"

ultra96Skip = 4
desktopSkip = 1

video = ultra96Video
skip = 1

app = Flask(__name__)
Bootstrap(app)

result = None
yolo3_output = np.array([0, 0])
ready = False
yolo3_ready = False
process = False
count = 0
finish = False


@app.route('/')
def index():
    images = os.listdir(app.static_folder)
    return render_template('index.html', images=images)


@app.route('/signup', methods=['POST'])
def signup():
    global finish
    print("POST: Received")
    finish = True
    return redirect('/')


def gen(cap):
    global result, ready
    count = 0
    while True:
        count += 1
        ret, frame = cap.read()

        if not (ret):
            continue

        if not (count % 4 == 0):
            continue

        boxes = np.loadtxt("../output/text/" + str(count) + ".txt", dtype=np.int32, delimiter=",")

        frame = frame[400:800, 400:1000]
        for box in boxes:
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 255), 3)

        time.sleep(0.02)
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        success, img_enc = cv2.imencode('.jpg', frame)
        result = img_enc.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(cv2.VideoCapture(video)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
