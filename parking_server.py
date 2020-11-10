#!/usr/bin/env python
from flask import Flask, render_template, Response
from flask_bootstrap import Bootstrap
import cv2
import time
import threading
import numpy as np


desktopVideo = "../Videos/ParkingLotKCropped.mp4"
ultra96Video = "/home/xilinx/jupyter_notebooks/pynq-dpu/video/ParkingLotKCropped.mp4"

ultra96Skip = 4
desktopSkip = 1

video = desktopVideo
skip = desktopSkip

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

result = None
yolo3_output = np.array([0,0])
ready = False
yolo3_ready = False
process = False
count = 0


def gen():
	global result, ready, yolo3_ready, yolo3_output
	while True:
		if ready:
			ready = False
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def detect():
	global result, ready, process, frame, count, yolo3_ready
	cap = cv2.VideoCapture(video)
	while True:
		count += 1
		ret, frame = cap.read()
		if not(ret):
			continue
		frame = cv2.pyrDown(frame)
		if count % skip == 0:
			frame = cv2.circle(frame, (yolo3_output[0], yolo3_output[1]), 5, (0,255,0), -1)
			success, img_enc = cv2.imencode('.jpg',frame)
			result = img_enc.tobytes()
			ready = True
		if count % 10 == 0:
			yolo3_thread = threading.Thread(target=yolo3_detect, args=(frame,))
			yolo3_thread.start()	



def yolo3_detect(frame):
	global yolo3_output, yolo3_ready, count
	yolo3_output = np.array([int(100 + 50 * np.sin(count*0.01)), int(100 + 50 * np.cos(count*0.01))])
	yolo3_ready = True



if __name__ == '__main__':
	detector_thread = threading.Thread(target=detect)
	detector_thread.start()




	app.run(host='0.0.0.0', debug=True)
