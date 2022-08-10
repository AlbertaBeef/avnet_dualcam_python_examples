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

# USAGE
# python stereo_face_detection.py [--width 640] [--height 480] [--detthreshold 0.55] [--nmsthreshold 0.35] [--threads 1] [--fps 1] [--output 0]

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
import queue

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from avnet_dualcam.dualcam import DualCam
from vitis_ai_vart.facedetect import FaceDetect
from vitis_ai_vart.facelandmark import FaceLandmark
from vitis_ai_vart.utils import get_child_subgraph_dpu


global bQuit

global width
global height

global bLandmarksAvailable
global bUseLandmarks
global nLandmarkId

global fps_overlay

global output_select
#global out

global dualcam
global brightness

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
    # gst-launch-1.0 tcpclientsrc host=ZUB1CG_IP port=5000 ! jpegdec ! videoconvert ! autovideosink sync=false

  print("[INFO] gstreamer output pipeline = ",pipeline_out)
  return cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, 0, 25.0, (640,480))

def taskCapture(width,height,queueLeft,queueRight):

    global dualcam
    global brightness
    global bQuit

    #print("[INFO] taskCapture : starting thread ...")

    # Initialize the capture pipeline
    print("[INFO] Initializing the capture pipeline ...")
    dualcam = DualCam('ar0144_dual',width,height)
    dualcam.set_brightness(brightness)  

    while not bQuit:
        # Capture image from camera
        left_frame,right_frame = dualcam.capture_dual()

        # Push captured image to input queue
        queueLeft.put(left_frame)
        queueRight.put(right_frame)

    #print("[INFO] taskCapture : exiting thread ...")



def taskFaceDet(side,worker,densebox_dpu,detThreshold,nmsThreshold,queueImg,queueFaces):

    global bQuit

    global width
    global height

    global bLandmarksAvailable
    global bUseLandmarks
    global nLandmarkId

    #print("[INFO] taskFaceDet[",side,worker,"] : starting thread ...")

    # Start the face detector
    dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
    dpu_face_detector.start()

    while not bQuit:
        # Pop captured image from input queue
        frame = queueImg.get()

        # Vitis-AI/DPU based face detector
        faces = dpu_face_detector.process(frame)

        # Push detected faces to output queue
        queueFaces.put(faces)

    # Stop the face detector
    dpu_face_detector.stop()

    #print("[INFO] taskFaceDet[",side,worker,"] : exiting thread ...")


def taskWorker(worker,landmark_dpu,queueCapLeft,queueCapRight,queueLeftImg,queueRightImg,queueLeftFaces,queueRightFaces,queueOut):

    global bQuit

    global width
    global height

    global bLandmarksAvailable
    global bUseLandmarks
    global nLandmarkId

    #print("[INFO] taskWorker[",worker,"] : starting thread ...")

    # Start the face landmark
    dpu_face_landmark = FaceLandmark(landmark_dpu)
    dpu_face_landmark.start()

    while not bQuit:
        # Pop captured image from input queue
        left_frame = queueCapLeft.get()
        right_frame = queueCapRight.get()

        # Make copies of left/right images for graphical annotations and display
        frame1 = left_frame.copy()
        frame2 = right_frame.copy()

        # Vitis-AI/DPU based face detector
        #left_faces = dpu_face_detector.process(left_frame)
        #right_faces = dpu_face_detector.process(right_frame)

        # Implement face detection with two parallel tasks
        queueLeftImg.put(left_frame)
        queueRightImg.put(right_frame)
        left_faces = queueLeftFaces.get()
        right_faces = queueRightFaces.get()

        # if one face detected in each image, calculate the centroids to detect distance range
        distance = 600
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

        # Push processed image to output queue
        queueOut.put(display_frame)

    # Stop the face landmark
    dpu_face_landmark.stop()

    # workaround : to ensure other worker threads stop, 
    #              make sure input queue is not empty 
    queueIn.put(frame)

    #print("[INFO] taskWorker[",worker,"] : exiting thread ...")

def taskDisplay(queueOut):

    global bQuit

    global width
    global height

    global img_kiss

    global fps_overlay
    
    global output_select
    #global out

    global dualcam
    global brightness

    #print("[INFO] taskDisplay : starting thread ...")

    # init the real-time FPS display
    rt_fps_count = 0;
    rt_fps_time = cv2.getTickCount()
    rt_fps_valid = False
    rt_fps = 0.0
    rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
    rt_fps_x = 10
    rt_fps_y = height-10

    #output_select = 1
    out = []	
    if output_select > 0:
        out = initVideoWriter(output_select)
    print("[INFO] output_select=",output_select," out=",out)

    while not bQuit:
        # Update the real-time FPS counter
        if rt_fps_count == 0:
            rt_fps_time = cv2.getTickCount()
    
        # Pop processed image from output queue
        display_frame = queueOut.get()

        # display fps
        if rt_fps_valid == True:
            cv2.putText(display_frame, rt_fps_message, (rt_fps_x,rt_fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # Display the processed image
        if output_select == 0:
            cv2.imshow("Stereo Face Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
		
        if output_select > 0 and out.isOpened():
            out.write(display_frame)

        if key == ord("d"):
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

    # Trigger all threads to stop
    bQuit = True

    # Cleanup
    cv2.destroyAllWindows()

    #print("[INFO] taskDisplay : exiting thread ...")



def main(argv):

    global bQuit
    bQuit = False

    global width
    global height

    global bLandmarksAvailable
    global bUseLandmarks
    global nLandmarkId
    bLandmarksAvailable = False
    bUseLandmarks = False
    nLandmarkId = 2

    global fps_overlay
    global output_select
    #global out

    global brightness

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-W", "--width", required=False,
        help = "input width (default = 640)")
    ap.add_argument("-H", "--height", required=False,
        help = "input height (default = 480)")
    ap.add_argument("-d", "--detthreshold", required=False,
        help = "face detector softmax threshold (default = 0.55)")
    ap.add_argument("-n", "--nmsthreshold", required=False,
        help = "face detector NMS threshold (default = 0.35)")
    ap.add_argument("-t", "--threads", required=False,
        help = "number of worker threads (default = 1)")
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

    if not args.get("threads",False):
        threads = 1
    else:
        threads = int(args["threads"])
    print('[INFO] number of worker threads = ', threads )

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
    all_densebox_left_dpu_runners = [];
    all_densebox_right_dpu_runners = [];
    for i in range(int(threads)):
      all_densebox_left_dpu_runners.append(vart.Runner.create_runner(densebox_subgraphs[0], "run"));
      all_densebox_right_dpu_runners.append(vart.Runner.create_runner(densebox_subgraphs[0], "run"));

    # Initialize Vitis-AI/DPU based face landmark
    landmark_xmodel = "/usr/share/vitis_ai_library/models/face_landmark/face_landmark.xmodel"
    landmark_graph = xir.Graph.deserialize(landmark_xmodel)
    landmark_subgraphs = get_child_subgraph_dpu(landmark_graph)
    if len(landmark_subgraphs) == 1: # only one DPU kernel
      bLandmarksAvailable = True
    all_landmark_dpu_runners = [];
    for i in range(int(threads)):
      all_landmark_dpu_runners.append(vart.Runner.create_runner(landmark_subgraphs[0], "run"));

    # Init synchronous queues for inter-thread communication
    queueCapLeft = queue.Queue(maxsize=2)
    queueCapRight = queue.Queue(maxsize=2)
    queueLeftImg = queue.Queue(maxsize=2*threads)
    queueLeftFaces = queue.Queue(maxsize=2*threads)
    queueRightImg = queue.Queue(maxsize=2*threads)
    queueRightFaces = queue.Queue(maxsize=2*threads)
    queueOut = queue.Queue(maxsize=4)

    # Launch threads
    threadAll = []
    tc = threading.Thread(target=taskCapture, args=(width,height,queueCapLeft,queueCapRight))
    threadAll.append(tc)
    for i in range(threads):
        tl = threading.Thread(target=taskFaceDet, args=("left",i,all_densebox_left_dpu_runners[i],detThreshold,nmsThreshold,queueLeftImg,queueLeftFaces))
        threadAll.append(tl)
        tr = threading.Thread(target=taskFaceDet, args=("right",i,all_densebox_right_dpu_runners[i],detThreshold,nmsThreshold,queueRightImg,queueRightFaces))
        threadAll.append(tr)
        tw = threading.Thread(target=taskWorker, args=(i,all_landmark_dpu_runners[i],queueCapLeft,queueCapRight,queueLeftImg,queueRightImg,queueLeftFaces,queueRightFaces,queueOut))
        threadAll.append(tw)
    td = threading.Thread(target=taskDisplay, args=(queueOut,))
    threadAll.append(td)
    for x in threadAll:
        x.start()

    # Wait for all threads to stop
    for x in threadAll:
        x.join()

    # Cleanup VART API
    del all_densebox_left_dpu_runners
    del all_densebox_right_dpu_runners
    del all_landmark_dpu_runners


if __name__ == "__main__":
    main(sys.argv)

