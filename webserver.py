#!/usr/bin/env python
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')



def gen(cap):
	while True:
		ret, frame = cap.read()
		success, img_enc = cv2.imencode('.jpg',frame)
		final = img_enc.tobytes()
		yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cv2.VideoCapture("../Videos/ParkingLotKCropped.mp4")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)