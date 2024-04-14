import os
import cv2
import torch
import numpy as np

from tracker import Tracker
from src.resnet_yolo import resnet50

device = torch.device("cuda")

load_network_path = None
# load_network_path = os.getcwd() + '/checkpoints/detector2.pth'
pretrained = True

# use to load a previously trained network
if load_network_path is not None:
    print('Loading saved network from {}'.format(load_network_path))
    net = resnet50().to(device)
    net.load_state_dict(torch.load(load_network_path))
else:
    print('Load pre-trained model')
    net = torch.hub.load('ultralytics/yolov5', 'yolov5s')

'''
@params: 
file_path - filepath to initial video
output_folder - folder to stash every frame
filename - combines all stashed frames into mp4
'''

file_path = "test_video.mp4"
output_folder = "video_images"
filename = "output.mp4"


def test():
    cap = cv2.VideoCapture(file_path)
    os.makedirs(output_folder, exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Instantiate Tracker
    tracker = Tracker(net, output_folder, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        upd_frame = tracker.process_frame(frame)

        cv2.imshow("Frame", upd_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    frames = tracker.get_frames()
    # Define parameters
    fps = 30

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Iterate over the list of frames and write each frame to video
    for frame in frames:

        out.write(frame)

    # Release VideoWriter object
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()