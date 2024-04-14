import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():

    ret, frame = cap.read()

    results = model(frame)
    for detection in results.xyxy[0]:
        bbox = detection[:4]  # Extract bounding box coordinates (xmin, ymin, xmax, ymax)
        conf = detection[4]  # Confidence score
        class_index = int(detection[5])  # Class index
        class_name = model.names[class_index]  # Class name

        # Print bounding box data and class information
        print("Bounding Box:", bbox)
        print("Class:", class_name, " Confidence:", conf)

    cv2.imshow("Frame", np.squeeze(results.render()))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break