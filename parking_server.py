#!/usr/bin/env python
from flask import Flask, render_template, Response, redirect
from flask_bootstrap import Bootstrap
import time
import threading
import numpy as np
import cv2
import lsb_release
import sys


from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
overlay.load_model("dpu_tf_yolov3.elf")

import random
import colorsys
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
#%matplotlib inline
from pynq_dpu.edge.dnndk.tf_yolov3_voc_py.tf_yolov3_voc import *

anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
anchor_float = [float(x) for x in anchor_list]
anchors = np.array(anchor_float).reshape(-1, 2)

classes_path = "files/voc_classes.txt"
class_names = get_class(classes_path)

num_classes = len(class_names)
hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: 
                  (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), 
                  colors))
random.seed(0)
random.shuffle(colors)
random.seed(None)

KERNEL_CONV="tf_yolov3"
CONV_INPUT_NODE="conv2d_1_convolution"
CONV_OUTPUT_NODE1="conv2d_59_convolution"
CONV_OUTPUT_NODE2="conv2d_67_convolution"
CONV_OUTPUT_NODE3="conv2d_75_convolution"

def draw_boxes(image, boxes, scores, classes):
    _, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(boxes):
        [top, left, bottom, right] = bbox
        width, height = right - left, bottom - top
        center_x, center_y = left + width*0.5, top + height*0.5
        score, class_index = scores[i], classes[i]
        label = '{}: {:.4f}'.format(class_names[class_index], score) 
        color = tuple([color/255 for color in colors[class_index]])
        ax.add_patch(Rectangle((left, top), width, height,
                               edgecolor=color, facecolor='none'))
        ax.annotate(label, (center_x, center_y), color=color, weight='bold', 
                    fontsize=12, ha='center', va='center')
    return ax
    
def evaluate(yolo_outputs, image_shape, class_names, anchors):
    score_thresh = 0.2
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32

    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(
            yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), 
            input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)

    mask = box_scores >= score_thresh
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np) 
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    boxes_ = np.concatenate(boxes_, axis = 0)
    scores_ = np.concatenate(scores_, axis = 0)
    classes_ = np.concatenate(classes_, axis = 0)

    return boxes_, scores_, classes_


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

def detect():
	global result, ready, count, finish
	cap = cv2.VideoCapture(video)


	n2cube.dpuOpen()
	kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
	task = n2cube.dpuCreateTask(kernel, 0)
	
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




			image_size = frame.shape[:2]
			image_data = np.array(pre_process(frame, (416, 416)), dtype=np.float32)


			input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
			n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, image_data, input_len)

			n2cube.dpuRunTask(task)

			conv_sbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1)
			conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, 
	                                                       conv_sbbox_size)
			conv_out1 = np.reshape(conv_out1, (1, 13, 13, 75))

			conv_mbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2)
			conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, 
	                                                       conv_mbbox_size)
			conv_out2 = np.reshape(conv_out2, (1, 26, 26, 75))

			conv_lbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3)
			conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, 
	                                                       conv_lbbox_size)
			conv_out3 = np.reshape(conv_out3, (1, 52, 52, 75))

			yolo_outputs = [conv_out1, conv_out2, conv_out3]    


			boxes, scores, classes = evaluate(yolo_outputs, image_size, class_names, anchors)
	                                          
	                                          
	        #_ = draw_boxes(image, boxes, scores, classes)

			success, img_enc = cv2.imencode('.jpg',frame)
			result = img_enc.tobytes()
			ready = True




	n2cube.dpuDestroyTask(task)
	n2cube.dpuDestroyKernel(kernel)



if __name__ == '__main__':
	detector_thread = threading.Thread(target=detect)
	detector_thread.start()




	app.run(host='0.0.0.0', debug=True)
