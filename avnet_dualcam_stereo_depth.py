'''
Copyright 2023 Avnet Inc.

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
from calibration_store import load_stereo_coefficients

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from avnet_dualcam.dualcam import DualCam

# USAGE
# python avnet_dualcam_depth.py [--sensor ar0144] [--width 640] --height 480] --calibration_file stereo_data/calib/dualcam_stereo.yaml 

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument("-S", "--sensor", required=False, help = "image sensor (ar0144|ar1335|ar0830, default = ar0144)")
    parser.add_argument("-C", '--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument("-W", '--width', type=int, required=True, help='Input resolution width')
    parser.add_argument("-H", '--height', type=int, required=True, help='Input resolution height')

    args = parser.parse_args()
    print(args)

    if args.sensor == None:
        sensor = 'ar0144'
    else:
        sensor = args.sensor
          
    width = args.width
    height = args.height   

    dualcam = DualCam(sensor,'dual',width,height)

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    while True:  # Loop until 'q' pressed or stream ends
        leftFrame,rightFrame = dualcam.capture_dual()
        
        height, width, channel = leftFrame.shape  # We will use the shape for remap
        print("size =",height,"X",width, "chan", channel) 

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disparity_image = depth_map(gray_left, gray_right)  # Get the disparity map

        disparity_heatmap = cv2.applyColorMap(disparity_image, cv2.COLORMAP_JET)
        output2 = cv2.hconcat([left_rectified,disparity_heatmap,right_rectified])
        cv2.imshow('Stereo Depth', output2)
                
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break
        elif key & 0xFF == ord('c'):
            cv2.imwrite("./img_stereo_depth.png", output)
            print("image taken")


    # Release the sources.
    dualcam.release()
    cv2.destroyAllWindows()

