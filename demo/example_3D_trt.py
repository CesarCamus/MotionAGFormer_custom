"""
This file presents an example of how to run the triton model for 3D pose estimation. Specify the input video path, corresponding sequence of 2D keypoints path and the output directory where the results will be saved. The 2D keypoints are loaded from the specified path and the 3D pose is estimated using the triton model. The results are saved in the output directory as a JSON file.
"""

import norfair
from norfair import Detection, Paths, Tracker
from triton_backend.utils import BoundingBox, postprocess
from triton_backend.render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from triton_backend.labels import COCOLabels
from triton_backend.trt_client import Pose3DTritonInferencer
import cv2
import sys
import torch
from typing import List
import numpy as np
import onnx
import onnxruntime
import os
import json

pose3D_detector = Pose3DTritonInferencer()

# def parser_2d_pose(json_file_path):
#     with open(json_file_path) as f:
#         data = json.load(f)
    
#     two_d_data = data['twod_pose']
#     frames = []
#     for frame_id in two_d_data:
#         frame_data = data['twod_pose'][frame_id]
#         keypoints = frame_data["keypoints"]
#         scores = frame_data["score"]
        
#         # Add score to each keypoint
#         frame_array = np.array([[x, y, s] for (x, y), s in zip(keypoints, scores)])
#         frames.append(frame_array)

#     # Convert to NumPy array with shape (num_frames, 17, 3)
#     frames_np = np.stack(frames)  # Shape: (num_frames, 17, 3)

#     # Add batch dimension → shape: (1, num_frames, 17, 3)
#     output_array = np.expand_dims(frames_np, axis=0)

#     #print("Shape:", output_array.shape)
#     return output_array


json_file_path = "raw_maneuvers_data/json_example.json"
example_video_path = "/home/cesar/Desktop/code/models_trial/MotionAGFormer_custom/demo/raw_maneuvers_data/360/360_10.mp4"
keypoints_path = "processed_videos/360/360_10/input_2D/keypoints.npz"
example_keypoints = np.load(keypoints_path, allow_pickle=True)["reconstruction"]

#print("Example keypoints shape: ", example_keypoints.shape)
output_dir = "demo/results_trt/360/360_10"

pose3D_detector.process_2D_seq(example_keypoints,example_video_path, output_dir)

pose3D_detector.triton_client.close()

#kpts = parser_2d_pose(json_file_path)