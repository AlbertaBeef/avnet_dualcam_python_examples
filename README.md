# avnet_dualcam_python_examples
Python-based OpenCV examples for the AP1302-based Dual Camera modules

# Instructions for use with Vitis 2022.2

To clone this repo, run:
```
git clone -b 2022.2 https://github.com/AlbertaBeef/avnet_dualcam_python_examples.git
```

# Tools Version

The supported Xilinx tools release is 2022.2.

# Supported Hardware

The following AP1302-based dualcam modules, and their respective carriers, are supported:
   - Ultra96-V2 + DualCam Mezzanine (96Boards)
   - ZUBoard + DualCam Mezzanine (SYZYGY)

# Contents

1. avnet_dualcam

   This folder contains a python wrapper for the AP1302-based dualcam modules, including:
   - Ultra96-V2 + DualCam Mezzanine (96Boards)
   - ZUBoard + DualCam Mezzanine (SYZYGY)

   The dualcam.py script will auto-detect the following:
   - /dev/video# (detected by searching for "vcap_CAPTURE_PIPELINE_v_proc_ss" string)
   - /dev/media# (detected by searching for "vcap_CAPTURE_PIPELINE_v_proc_ss" string)
   - ap1302.#-003c
   - 

2. depthai_helpers

   This folder contains python scripts for stereo calibration.
   These files are modified versions taken from the DepthAI OAK camera project.

3. stereo_data

   This folder contains sample captured stereo images (left, right) and stereo calibration data.


# Documentation

1. To launch a dual ar0144 passthrough

   $ python3 avnet_dualcam_passthrough.py --sensor ar0144 --mode dual

2. To launch a single ar0144 passthrough

   $ python3 avnet_dualcam_dual_passthrough.py --sensor ar0144 --mode primary
   $ python3 avnet_dualcam_dual_passthrough.py --sensor ar0144 --mode secondary

3. To launch anaglyph example

   $ python3 avnet_dualcam_anaglyph.py --sensor ar0144 --mode dual

4. To perform stereo calibration, then software-based depth estimation

   $ python3 avnet_dualcam_stereo_calibrate.py ... --width 640 --height 480

   $ python3 avnet_dualcam_stereo_depth.py --calibration_file ./stereo_data/calib/dualcam_stereo.yml --width 640 --height 480


5. To perform stereo face detection (requires Vitis-AI 2.0)

   Single-thread:
   $ python3 avnet_dualcam_stereo_face_detection.py

   Multi-thread:
   $ python3 avnet_dualcam_stereo_face_detection_mt.py


