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
# python anaglyph.py [--width 640] [--height 480]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-W", "--width", required=False,
	help = "input width (default = 640)")
ap.add_argument("-H", "--height", required=False,
	help = "input height (default = 480)")
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

# Initialize the capture pipeline
print("[INFO] Initializing the capture pipeline ...")
dualcam = DualCam('ar0144_dual',width,height)

while(True):
  # Capture input
  left,right = dualcam.capture_dual()

  # Calculate anaglyph
  # reference : https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
  # - right : cyan (blue+green)
  anaglyph = right
  # - left : red
  anaglyph[:,:,2] = left[:,:,2]

  # Display output
  cv2.imshow('u96v2_sbc_dualcam_ar0144 - anaglyph',anaglyph)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
dualcam.release()
cv2.destroyAllWindows()

