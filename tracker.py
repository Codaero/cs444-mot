import os
import cv2
import numpy as np
import torch

from src.predict import predict_image
from src.config import VOC_CLASSES, COLORS

from hungarian_algorithm import object_assign
from kalman_filter import KalmanFilter

'''
Object Attributes:
- id : int
- center : tuple(float, float)
- height : float
- width : float
- velocity: tuple(float, float)
- class : string
- prob : float
- last_frame_seen: int
- active_track: bool
- cov: array <= <dim : (6,6)>
'''

'''
Frames List Attribute:
- Frame Number
- Object List

result - [[(x1, y1) <top left>, (x2, y2) <bottom right>, VOC_CLASSES[cls_index] <class prob>, image_name <filename>, prob <confidence>], ... ]
'''

class Tracker:
    def __init__(self, net, output_folder, frame_dim):
        self.frame_dim = frame_dim
        self.frame_count = 0
        self.frame_list = []
        self.net = net
        self.object_track = []
        self.output_folder = output_folder
        self.T_lost = 5  # The original paper stated if an object reappears, it will take a new identity
        self.IOU_min = 0.1  # The paper states to reject any object assignments less than IOUmin
        self.start_assign = False
        self.kalman_filter = KalmanFilter()

    def process_frame(self, frame):
        frame_filename = os.path.join(self.output_folder, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # TASK: GENERATE RESULTS FROM YOLO MODEL
        result = self.net(frame)
        result = result.xyxy[0]

        # TASK: REMOVE BOUNDING BOXES RESULTING FROM ZERO AREA
        rm_idx = []
        for obj_idx, obj in enumerate(result):
            width = abs(obj[0] - obj[2])
            height = abs(obj[1] - obj[3])
            area = width * height
            if area == 0 or self.net.names[int(obj[5])] != 'person':
                rm_idx.append(obj_idx)

        # TASK: TRANSFORM OBJECT DATA WITH CENTER COORDINATE
        upd_result = [obj for idx, obj in enumerate(result) if idx not in rm_idx]
        objects = self.create_objects(upd_result)

        # TASK: GET THE SET OF OBJECT TRACKS WE WANT TO DISPLAY VIA HUNGARIAN ALGORITHM
        if not self.start_assign:
            # assign unique ids to every object
            self.object_track = objects
            if len(self.object_track) > 0:
                self.start_assign = True

        else:
            # check previous frame for object assignment
            print(self.object_track)
            print('\n\n\n')
            # get all predictions
            kalman_x, kalman_p, kalman_ids = [], [], []
            for obj in self.object_track:
                if obj['active_track']:
                    x = obj['center'][0]
                    y = obj['center'][1]
                    vx = obj['velocity'][0]
                    vy = obj['velocity'][1]
                    kalman_x.append([x, y, vx, vy])
                    kalman_ids.append(obj['id'])
                    kalman_p.append(obj['cov'])

            kalman_x = torch.tensor(kalman_x).cpu().numpy()
            kalman_p = torch.tensor(kalman_p).cpu().numpy()

            pred_x, pred_p = [], []
            for idx in range(len(kalman_x)):
                x, p = self.kalman_filter.predict(kalman_x[idx], kalman_p[idx])
                pred_x.append(x)
                pred_p.append(p)
            pred_x, pred_p = np.array(pred_x), np.array(pred_p)

            # get all measurements
            measured = object_assign(objects, self.object_track, self.frame_count, self.IOU_min)
            kalman_m = []
            for obj in measured:
                if obj['active_track'] and obj['id'] in kalman_ids:
                    m_x = obj['center'][0]
                    m_y = obj['center'][1]
                    m_vx = obj['velocity'][0]
                    m_vy = obj['velocity'][0]
                    kalman_m.append([m_x, m_y, m_vx, m_vy])
            kalman_m = torch.tensor(kalman_m).cpu().numpy()

            self.object_track = measured

            # get all targets and save them
            for idx in range(len(kalman_m)):
                targ_x, targ_p = self.kalman_filter.update(kalman_m[idx], pred_x[idx], pred_p[idx])
                self.object_track[kalman_ids[idx]]['cov'] = targ_p
                self.object_track[kalman_ids[idx]]['center'] = (targ_x[0], targ_x[1])
                self.object_track[kalman_ids[idx]]['velocity'] = (targ_x[2], targ_x[3])

        # TASK: BOUND CHECK FOR OBJECTS ENTERING AND LEAVING THE FRAME
        self.bound_check()

        # TASK: DISPLAY TRACKS
        frame = self.edit_frame(frame)

        cv2.imwrite(frame_filename, frame)

        self.frame_count += 1
        self.frame_list.append(frame)

        return frame

    def bound_check(self):
        for obj_idx, obj in enumerate(self.object_track):
            center = obj['center']
            width = obj['width']
            height = obj['height']
            velocity = obj['velocity']
            if self.frame_count - obj['last_frame_seen'] > self.T_lost:
                # Get corners of bounding box
                tp_lt = (int(center[0] - width / 2), int(center[1] - height / 2))
                bt_rt = (int(center[0] + width / 2), int(center[1] + height / 2))
                if (tp_lt[0] <= 5 and velocity[0] < 0) or (tp_lt[1] <= 5 and velocity[1] < 0) or (bt_rt[0] >= 635 and velocity[0] > 0) or (bt_rt[1] >= 355 and velocity[1] > 0):
                    self.object_track[obj_idx]['active_track'] = False

    def edit_frame(self, frame):
        for obj in self.object_track:
            center = obj['center']
            width = obj['width']
            height = obj['height']
            class_name = obj['class']
            id_val = obj['id']

            if obj['active_track'] and obj['last_frame_seen'] == self.frame_count:

                left_up = (int(center[0] - width / 2), int(center[1] - height / 2))
                right_bottom = (int(center[0] + width / 2), int(center[1] + height / 2))
                color = COLORS[VOC_CLASSES.index(class_name)]
                cv2.rectangle(frame, left_up, right_bottom, color, 2)
                label = str(id_val)
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (left_up[0], left_up[1] - text_size[1])
                cv2.rectangle(frame, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(frame, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        return frame
        
    def create_objects(self, result):

        object_transform = []

        if len(result) == 0:
            return []

        if not self.start_assign:
            obj_idx = 0
        else:
            obj_idx = self.object_track[-1]['id'] + 1

        for obj in result:

            center = ((obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2)
            height = abs(obj[1] - obj[3])
            width = abs(obj[0] - obj[2])

            new_object = dict()
            new_object['id'] = obj_idx
            new_object['center'] = center
            new_object['height'] = height
            new_object['width'] = width
            new_object['velocity'] = (0, 0)
            new_object['prob'] = obj[4]
            new_object['class'] = self.net.names[int(obj[5])]
            new_object['last_frame_seen'] = self.frame_count
            new_object['active_track'] = True
            new_object['cov'] = np.identity(4)
            obj_idx += 1

            object_transform.append(new_object)

        return object_transform

    def get_frames(self):
        return self.frame_list
