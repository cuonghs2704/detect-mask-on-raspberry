import onnx
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path

import torch
import torchvision
import cv2
from collections import OrderedDict,namedtuple
import time

from tool import *


time_start = time.perf_counter()

#initial weight and model onnx
w = './weight/FastestDet (2).onnx'
ori_image = cv2.imread("maksssksksss0.png")
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
#preparing input {input.1 : array[]}

outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]

# Open the video file for reading
cap = cv2.VideoCapture('maskvideo.mp4')  # Replace 'input_video.mp4' with your input video file
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# Get video information (e.g., frame width, height, frame rate)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
# Define the codec for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Create a VideoWriter object for saving the edited video in MP4 format
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
# Loop through the video frames
num_frame = 0
aver_time = 0
while True:
    s_frame = time.perf_counter()
    ret, frame = cap.read()  # Read a frame
    
    if not ret:
        break  # Break the loop if we've reached the end of the video
    
    frame = predict(frame, 640, 640, inname, outname, session)
    # Write the edited frame to the output video
    frame = cv2.resize(frame, (frame_width, frame_height), interpolation = cv2.INTER_LINEAR)

    out.write(frame)
    e_frame = time.perf_counter()

    aver_time += (e_frame - s_frame)
    num_frame += 1

print(aver_time / num_frame)
# Release the video objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video editing and saving in MP4 format complete.")


time_end  = time.perf_counter()

print("time excuted = ", time_end - time_start)
  # 绘制预测框

