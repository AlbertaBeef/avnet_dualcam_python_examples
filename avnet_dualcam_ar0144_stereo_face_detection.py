'''
Copyright 2022 Avnet Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# Reference:
#    https://www.hackster.io/AlbertaBeef/stereo-neural-inference-with-the-dual-camera-mezzanine-418aca

# USAGE
# python stereo_face_detection.py [--width 640] [--height 480] [--detthreshold 0.55] [--nmsthreshold 0.35] [--fps 1] [--output 0]

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys
import argparse
import sys

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from avnet_dualcam.dualcam import DualCam
from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facelandmark import FaceLandmark
from vitis_ai_vart.utils import get_child_subgraph_dpu


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-W", "--width", required=False,
	help = "input width (default = 640)")
ap.add_argument("-H", "--height", required=False,
	help = "input height (default = 480)")
ap.add_argument("-d", "--detthreshold", required=False,
	help = "face detector softmax threshold (default = 0.55)")
ap.add_argument("-n", "--nmsthreshold", required=False,
	help = "face detector NMS threshold (default = 0.35)")
ap.add_argument("-f", "--fps", required=False, default=False, action="store_true",
	help = "fps overlay (default = off")
ap.add_argument("-o", "--output", required=False,
	help = "output display (0==imshow(default) | 1==kmssink)")
ap.add_argument("-b", "--brightness", required=False,
	help = "brightness in 0-65535 range (default = 256)")
args = vars(ap.parse_args())

if not args.get("width",False):
  width = 640
else:
  width = int(args["width"])
if not args.get("height",False):
  height = 480
else:
  height = int(args["height"])
print('[INFO] input resolution = ',width,'X',height)

if not args.get("detthreshold",False):
  detThreshold = 0.55
else:
  detThreshold = float(args["detthreshold"])
print('[INFO] face detector - softmax threshold = ',detThreshold)

if not args.get("nmsthreshold",False):
  nmsThreshold = 0.35
else:
  nmsThreshold = float(args["nmsthreshold"])
print('[INFO] face detector - NMS threshold = ',nmsThreshold)

if not args.get("fps",False):
  fps_overlay = False 
else:
  fps_overlay = True
print('[INFO] fps overlay =  ',fps_overlay)

if not args.get("output",False):
  output_select = 0 
else:
  output_select = int(args["output"])
print('[INFO] output select =  ',output_select)

if not args.get("brightness",False):
  brightness = 256
else:
  brightness = int(args["brightness"])
print('[INFO] brightness = ',brightness)

# Initialize Vitis-AI/DPU based face detector
densebox_xmodel = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel"
densebox_graph = xir.Graph.deserialize(densebox_xmodel)
densebox_subgraphs = get_child_subgraph_dpu(densebox_graph)
assert len(densebox_subgraphs) == 1 # only one DPU kernel
densebox_dpu = vart.Runner.create_runner(densebox_subgraphs[0],"run")
dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
dpu_face_detector.start()

bLandmarksAvailable = False
bUseLandmarks = False
nLandmarkId = 2

# Initialize Vitis-AI/DPU based face landmark
landmark_xmodel = "/usr/share/vitis_ai_library/models/face_landmark/face_landmark.xmodel"
landmark_graph = xir.Graph.deserialize(landmark_xmodel)
landmark_subgraphs = get_child_subgraph_dpu(landmark_graph)
if len(landmark_subgraphs) == 1: # only one DPU kernel
  bLandmarksAvailable = True
if bLandmarksAvailable == True:
  landmark_dpu = vart.Runner.create_runner(landmark_subgraphs[0],"run")
  dpu_face_landmark = FaceLandmark(landmark_dpu)
  dpu_face_landmark.start()

# init the real-time FPS display
rt_fps_count = 0;
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = 10
rt_fps_y = height-10

# Initialize the capture pipeline
print("[INFO] Initializing the capture pipeline ...")
dualcam = DualCam('ar0144_dual',width,height)
dualcam.set_brightness(brightness)  


# inspired from cvzone.Utils.py
def cornerRect( img, bbox, l=20, t=5, rt=1, colorR=(255,0,255), colorC=(0,255,0)):

	#x1,y1,w,h = bbox
	#x2,y2 = x+w, y+h
	x1,y1,x2,y2 = bbox
	x1 = int(x1)
	x2 = int(x2)
	y1 = int(y1)
	y2 = int(y2)

	if rt != 0:
		cv2.rectangle(img,(x1,y1),(x2,y2),colorR,rt)

	# Top Left x1,y1
	cv2.line(img, (x1,y1), (x1+l,y1), colorC, t)
	cv2.line(img, (x1,y1), (x1,y1+l), colorC, t)
	# Top Right x2,y1
	cv2.line(img, (x2,y1), (x2-l,y1), colorC, t)
	cv2.line(img, (x2,y1), (x2,y1+l), colorC, t)
	# Top Left x1,y2
	cv2.line(img, (x1,y2), (x1+l,y2), colorC, t)
	cv2.line(img, (x1,y2), (x1,y2-l), colorC, t)
	# Top Left x2,y2
	cv2.line(img, (x2,y2), (x2-l,y2), colorC, t)
	cv2.line(img, (x2,y2), (x2,y2-l), colorC, t)

	return img

def initVideoWriter(output_select):
  pipeline_out = ""
  
  if output_select == 1:
	  pipeline_out = "appsrc ! video/x-raw, width=640,height=480, format=BGR ! kmssink bus-id=fd4a0000.display fullscreen-overlay=true sync=false"
  if output_select == 2:
    #pipeline_out = "appsrc ! video/x-raw, format=BGR ! jpegenc ! udpsink host=10.0.0.1 port=5000 sync=false"
    pipeline_out = "appsrc ! video/x-raw ! jpegenc ! multipartmux ! tcpserversink name=out-sink host=0.0.0.0 port=5000 sync=false"
    #decode on PC with
    # gst-launch-1.0 tcpclientsrc host=U96V2_IP port=5000 ! jpegdec ! videoconvert ! autovideosink sync=false

  return cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, 0, 25.0, (640,480))

out = []
if output_select > 0:
  out = initVideoWriter(output_select)

# loop over the frames from the video stream
while True:
	# Update the real-time FPS counter
	if rt_fps_count == 0:
		rt_fps_time = cv2.getTickCount()

	# Capture image from camera
	left_frame,right_frame = dualcam.capture_dual()

	# Make copies of left/right images for graphical annotations and display
	frame1 = left_frame.copy()
	frame2 = right_frame.copy()

	# Vitis-AI/DPU based face detector
	left_faces = dpu_face_detector.process(left_frame)
	right_faces = dpu_face_detector.process(right_frame)

	# if one face detected in each image, calculate the centroids to detect distance range
	distance_valid = False
	if (len(left_faces) == 1) & (len(right_faces) == 1):

		# loop over the left faces
		for i,(left,top,right,bottom) in enumerate(left_faces):
			cornerRect(frame2,(left,top,right,bottom),colorR=(255,255,255),colorC=(255,255,255))
 
			# centroid
			if bUseLandmarks == False:  
				x = int((left+right)/2)
				y = int((top+bottom)/2)
				cv2.circle(frame2,(x,y),4,(255,255,255),-1)
			# get left coordinate (keep float, for full precision)  
			left_cx = (left+right)/2
			left_cy = (top+bottom)/2

			# get face landmarks
			if bLandmarksAvailable == True:
				startX = int(left)
				startY = int(top)
				endX   = int(right)
				endY   = int(bottom)      
				face = left_frame[startY:endY, startX:endX]
				landmarks = dpu_face_landmark.process(face)
				if bUseLandmarks == True:  
					for i in range(5):
						x = startX + int(landmarks[i,0] * (endX-startX))
						y = startY + int(landmarks[i,1] * (endY-startY))
						cv2.circle( frame2, (x,y), 3, (255,255,255), 2)
					x = startX + int(landmarks[nLandmarkId,0] * (endX-startX))
					y = startY + int(landmarks[nLandmarkId,1] * (endY-startY))
					cv2.circle( frame2, (x,y), 4, (255,255,255), -1)
				# get left coordinate (keep float, for full precision)  
				left_lx = left   + (landmarks[nLandmarkId,0] * (right-left))
				left_ly = bottom + (landmarks[nLandmarkId,1] * (bottom-top))  

		# loop over the right faces
		for i,(left,top,right,bottom) in enumerate(right_faces): 
			cornerRect(frame2,(left,top,right,bottom),colorR=(255,255,0),colorC=(255,255,0))

			# centroid
			if bUseLandmarks == False:  
				x = int((left+right)/2)
				y = int((top+bottom)/2)
				cv2.circle(frame2,(x,y),4,(255,255,0),-1)
			# get right coordinate (keep float, for full precision)  
			right_cx = (left+right)/2
			right_cy = (top+bottom)/2

			# get face landmarks
			if bLandmarksAvailable == True:
				startX = int(left)
				startY = int(top)
				endX   = int(right)
				endY   = int(bottom)      
				face = right_frame[startY:endY, startX:endX]
				landmarks = dpu_face_landmark.process(face)
				if bUseLandmarks == True:  
					for i in range(5):
						x = startX + int(landmarks[i,0] * (endX-startX))
						y = startY + int(landmarks[i,1] * (endY-startY))
						cv2.circle( frame2, (x,y), 3, (255,255,0), 2)  
					x = startX + int(landmarks[nLandmarkId,0] * (endX-startX))
					y = startY + int(landmarks[nLandmarkId,1] * (endY-startY))
					cv2.circle( frame2, (x,y), 4, (255,255,0), -1)  
				# get right coordinate (keep float, for full precision)  
				right_lx = left   + (landmarks[nLandmarkId,0] * (right-left))
				right_ly = bottom + (landmarks[nLandmarkId,1] * (bottom-top))  

		delta_cx = abs(left_cx - right_cx)
		delta_cy = abs(right_cy - left_cy)
		message1 = "delta_cx="+str(int(delta_cx))

		if bLandmarksAvailable == True:
			delta_lx = abs(left_lx - right_lx)
			delta_ly = abs(right_ly - left_ly)
			message2 = "delta_lx="+str(int(delta_lx))

		if bUseLandmarks == False:  
			delta_x = delta_cx
			delta_y = delta_cy
			cv2.putText(frame2,message1,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)
			if bLandmarksAvailable == True:
				cv2.putText(frame2,message2,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)

		if bUseLandmarks == True:
			if bLandmarksAvailable == True:  
				delta_x = delta_lx
				delta_y = delta_ly
				cv2.putText(frame2,message1,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
				cv2.putText(frame2,message2,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)

		# distance = (baseline * focallength) / disparity
		#    ref : https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/
		#
		# baseline = 50 mm (measured)
		# focal length = 2.48mm * (1 pixel / 0.003mm) = 826.67 pixels => gives better results
		# focal length = 2.48mm * (1280 pixels / 5.565mm) = 570 pixels => 
		#    ref: http://avnet.me/ias-ar0144-datasheet
		#
		disparity = delta_x * (1280 / width) # scale back to active array
		distance = (50 * 827) / (disparity)
		#distance = (50 * 570) / (disparity)
		message1 = "disparity : "+str(int(disparity))+" pixels"
		message2 = "distance : "+str(int(distance))+" mm"
		cv2.putText(frame1,message1,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
		cv2.putText(frame1,message2,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)

		if ( (distance > 500) & (distance < 1000) ):
			distance_valid = True

	# loop over the left faces
	for i,(left,top,right,bottom) in enumerate(left_faces): 

		if distance_valid == True:
			cornerRect(frame1,(left,top,right,bottom),colorR=(0,255,0),colorC=(0,255,0))
		if distance_valid == False:
			cornerRect(frame1,(left,top,right,bottom),colorR=(0,0,255),colorC=(0,0,255))


	# Format the output frame
	if output_select ==1:
		display_frame = frame1
	else:
		display_frame = cv2.hconcat([frame1, frame2])

	# Display status messages
	status = ""
	if fps_overlay == True and rt_fps_valid == True:
		status = status + " " + rt_fps_message
	cv2.putText(display_frame, status, (rt_fps_x,rt_fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

	# Display the output frame
	if output_select == 0:
		cv2.imshow("Stereo Face Detection", display_frame)    
	if output_select > 0 and out.isOpened():
		out.write(display_frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("d"):
		if bLandmarksAvailable == True:  
			bUseLandmarks = not bUseLandmarks
			print("bUseLandmarks = ",bUseLandmarks);

	if key == ord("l"):
		nLandmarkId = nLandmarkId + 1
		if nLandmarkId >= 5:
			nLandmarkId = 0
		print("nLandmarkId = ",nLandmarkId);

	# if the ESC or 'q' key was pressed, break from the loop
	if key == 27 or key == ord("q"):
		break
  
	# Use 'b' and 'B" keys to adjust brightness
	if key == ord("B"):
		brightness = min(brightness + 256,65535)
		dualcam.set_brightness(brightness)  
	if key == ord("b"):
		brightness = max(brightness - 256,256)
		dualcam.set_brightness(brightness)  
 
 	# Update the real-time FPS counter
	rt_fps_count = rt_fps_count + 1
	if rt_fps_count >= 10:
		t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
		rt_fps_valid = True
		rt_fps = 10.0/t
		rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
		#print("[INFO] ",rt_fps_message)
		rt_fps_count = 0

# Stop the face detector
dpu_face_detector.stop()
del densebox_dpu

# Stop the landmark detector
dpu_face_landmark.stop()
del landmark_dpu

# Cleanup
cv2.destroyAllWindows()
