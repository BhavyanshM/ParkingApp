#!/usr/bin/env python
from flask import Flask, render_template, Response, redirect
from flask_bootstrap import Bootstrap
import time
import threading
import numpy as np
import cv2
import lsb_release
import sys


desktopVideo = "../Videos/ParkingLotKCropped.mp4"
ultra96Video = "/home/xilinx/jupyter_notebooks/pynq-dpu/video/ParkingLotKCropped.mp4"

ultra96Skip = 4
desktopSkip = 1

video = desktopVideo
skip = desktopSkip

app = Flask(__name__)
Bootstrap(app)


result = None
yolo3_output = np.array([0,0])
ready = False
yolo3_ready = False
process = False
count = 0

finish = False


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/signup', methods = ['POST'])
def signup():
	global finish
	print("POST: Received")
	finish = True	

	return redirect('/')



def gen():
	global result, ready
	while True:
		# print("Request:", ready)
		ready = False
		yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def get_output():
	import socket

	HOST = ''                 # Symbolic name meaning all available interfaces
	PORT = 50008              # Arbitrary non-privileged port
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind((HOST, PORT))
		s.listen(1)
		conn, addr = s.accept()
		with conn:
			print('Connected by', addr)
			while True:
				data = conn.recv(1024)

				print(str(data))

				if not data: continue
				conn.sendall(data)


def detect():
	global result, ready, count, finish, task, kernel
	cap = cv2.VideoCapture(video)


	
	while not(finish):
		count += 1
		ret, frame = cap.read()

		# print("Process")

		if not(ret):
			continue
		frame = cv2.pyrDown(frame)

		# print("Detect")
		# print(count)
		if count % skip == 0:



			success, img_enc = cv2.imencode('.jpg',frame)
			result = img_enc.tobytes()
			ready = True




if __name__ == '__main__':
	detector_thread = threading.Thread(target=detect)
	detector_thread.start()

	output_thread = threading.Thread(target=get_output)
	output_thread.start()


	app.run(host='0.0.0.0', debug=True)
