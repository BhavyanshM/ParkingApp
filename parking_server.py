#!/usr/bin/env python
from flask import Flask, render_template, Response
from flask_bootstrap import Bootstrap
import cv2
import time


desktopVideo = "../Videos/ParkingLotKCropped.mp4"
ultra96Video = "/home/xilinx/jupyter_notebooks/pynq-dpu/video/ParkingLotKCropped.mp4"

ultra96Skip = 4
desktopSkip = 1

video = ultra96Video
skip = ultra96Skip

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')



def gen(cap):
	count = 0
	while True:
		count += 1
		ret, frame = cap.read()
		frame = cv2.pyrDown(frame)
		if count % skip == 0:
			time.sleep(0.03*skip)
			success, img_enc = cv2.imencode('.jpg',frame)
			final = img_enc.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cv2.VideoCapture(video)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)
