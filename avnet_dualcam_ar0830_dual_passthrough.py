'''
Copyright 2021 Avnet Inc.

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

import numpy as np
import cv2
import argparse
import sys
import os

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from avnet_dualcam.dualcam import DualCam

# USAGE
# python dual_passthrough.py [--width 640] [--height 480]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-W", "--width", required=False,
	help = "input width (default = 640)")
ap.add_argument("-H", "--height", required=False,
	help = "input height (default = 480)")
ap.add_argument("-f", "--fps", required=False, default=False, action="store_true",
	help = "fps overlay (default = off")
ap.add_argument("-b", "--brightness", required=False,
	help = "brightness in 0-65535 range (default = 256)")
ap.add_argument("-e", "--exposure", required=False,
	help = "exposure in 1-12 range (default = 12)")
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

if not args.get("fps",False):
  fps_overlay = False 
else:
  fps_overlay = True
print('[INFO] fps overlay =  ',fps_overlay)

if not args.get("brightness",False):
  brightness = 256
else:
  brightness = int(args["brightness"])
print('[INFO] brightness = ',brightness)

if not args.get("exposure",False):
  exposure = 12
else:
  exposure = int(args["exposure"])
print('[INFO] exposure = ',exposure)

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
dualcam = DualCam('ar0830_dual',width,height)
dualcam.set_brightness(brightness)  
dualcam.set_exposure(exposure) 

while(True):
	# Update the real-time FPS counter
	if rt_fps_count == 0:
		rt_fps_time = cv2.getTickCount()

	# Capture input
	left,right = dualcam.capture_dual()

	# dual passthrough
	output = cv2.hconcat([left,right])

	# Display status messages
	status = ""
	if fps_overlay == True and rt_fps_valid == True:
		status = status + " " + rt_fps_message
	cv2.putText(output, status, (rt_fps_x,rt_fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

	# Display output
	cv2.imshow('avnet_dualcam - ar0830_dual - dual passthrough',output)

	key = cv2.waitKey(1) & 0xFF

	# if the ESC or 'q' key was pressed, break from the loop
	if key == 27 or key == ord("q"):
		break
   
	# Use 'b' and 'B' keys to adjust brightness
	if key == ord("B"):
		brightness = min(brightness + 256,65535)
		dualcam.set_brightness(brightness)  
	if key == ord("b"):
		brightness = max(brightness - 256,256)
		dualcam.set_brightness(brightness)  

	# Use 'e' and 'E' keys to adjust exposure
	if key == ord("E"):
		exposure = min(exposure + 1,12)
		dualcam.set_exposure(exposure)  
	if key == ord("e"):
		exposure = max(exposure - 1,0)
		dualcam.set_exposure(exposure)  

 	# Update the real-time FPS counter
	rt_fps_count = rt_fps_count + 1
	if rt_fps_count >= 10:
		t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
		rt_fps_valid = True
		rt_fps = 10.0/t
		rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
		#print("[INFO] ",rt_fps_message)
		rt_fps_count = 0

# When everything done, release the capture
dualcam.release()
cv2.destroyAllWindows()

