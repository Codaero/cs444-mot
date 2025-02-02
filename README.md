# SORT Implementation

## Abstract
This repository is an implementation of the SORT object tracking model which utilizes a Yolo Model, Kalman Filter, and Hungarian Algorithms to perform object tracking and assignment with predicated/measured filtering. An output of our tracking model is shown in the link below, and our final report is linked in the file named CS444__SORT.pdf.

## Instructions 

⚠️ DISCLAIMER: CUDA IS NECESSARY TO RUN THIS REPOSITORY ⚠️

The first step is to clone the repository and modify test.py file_path variable with a new filepath to a video file (.mp4, .mov, etc.). After modifying the file, run:
```
python3 test.py
```

## Pipeline

![image](https://github.com/user-attachments/assets/237d039d-c52a-407e-b3ea-e503ed93c9b0)

Above shows an image of the SORT pipeline. First, we capture a frame and is sent into a Yolo model. The Yolo model produces bounding box estimates, which is sent into a hungarian algorithm. At the same time, the hungarian algorithm stores a previous stashed Object tracking output which includes bounding boxes with IDs associated with them. The algorithm will use IOU of the previous frame and current frame to place new IDs on the new frame. Note bounding boxes with higher IOUs from the previous and current frames will most likely be assigned the same IDs. The hungarian model will output the same bounding boxes, but with IDs which is sent into a kalman filter. The kalman filter will first predict where the bounding boxes were supposed to be based on the previous frame and a linear velocity model, and then compare it with the measured result from our hungarian algorithm in an update step. The resulting output is a frame with bounding boxes and IDs associated with the boxes.

## Test Files

The following files are how exactly we unit tested each component within the pipeline. Since we utilitzed a pre-trained Yolo model, we did not have to test the object detection model.

```
python3 hungarian_algorithms_tester.py
```

```
python3 kalman_filter_tester.py
```

## Results

🛥️ From the video link below, we found some important observations in terms of our pipeline. First, the Kalman filter is not really needed for the video below. We believe that the kalman filter would be best used for when frames are being captured at slightly different locations. For example, if we placed a camera on "vessel 1," and if the vessel is capturing another vessel, say "vessel 2". Vessel 1 may be moving up and down with the current, causing some frames to be taken on a slightly higher altitude than other frames. Therefore, we need to do balance our predicted bounding box location with a measured bounding box location to produce the most accurate bounding box location. 

✈️ Our second observation is that the hungarian algorithms worked perfectly fine except when the Yolo Model does not locate an object on a a previous or current frame. This means that when the object is relocated by the Yolo Model, the object will be assigned a new idea. One method of fixing this is to not just look at a previous frame for running the hungarian algorithms for object assignment, but look at maybe the past 3 to 5 frames. 

🚗 Our last observation was in terms of the Yolo Model. We actually developed our own Yolov1 model on top of a resnet model and tested it. It was able to detect certain objects, but sometimes, it would throw random bounding boxes as well. This is not good for object tracking because a random bounding box could interfere with assignments based on IOU.  

🚀 Check out our test [video](https://drive.google.com/file/d/1YMBSVw7hP-Ys9FJncAQ56dNmT81XtkoR/view?usp=sharing) here
