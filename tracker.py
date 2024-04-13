import os
import cv2
from src.predict import predict_image
from src.config import VOC_CLASSES, COLORS

from hungarian_algorithm import object_assign

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
        self.T_lost = 1  # The original paper stated if an object reappears, it will take a new identity
        self.IOU_min = 0.1  # The paper states to reject any object assignments less than IOUmin


    def process_frame(self, frame):
        frame_filename = os.path.join(self.output_folder, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # TASK: GENERATE RESULTS FROM YOLO MODEL
        result = predict_image(self.net, frame_filename, root_img_directory="")

        # TASK: REMOVE BOUNDING BOXES RESULTING FROM ZERO AREA
        rm_idx = []
        for obj_idx, obj in enumerate(result):
            width = abs(obj[0][0] - obj[1][0])
            height = abs(obj[0][1] - obj[1][1])
            area = width * height
            if area == 0:
                rm_idx.append(obj_idx)

        # TASK: TRANSFORM OBJECT DATA WITH CENTER COORDINATE
        upd_result = [obj for idx, obj in enumerate(result) if idx not in rm_idx]
        objects = self.create_objects(upd_result)

        # TASK: GET THE SET OF OBJECT TRACKS WE WANT TO DISPLAY VIA HUNGARIAN ALGORITHM
        if self.frame_count == 0:
            # assign unique ids to every object
            self.object_track = objects

        else:
            # check previous frame for object assignment
            self.object_track = object_assign(objects, self.object_track, self.frame_count, self.IOU_min)


        # TASK: BOUND CHECK FOR OBJECTS ENTERING AND LEAVING THE FRAME
        for obj_idx, obj in enumerate(self.object_track):
            center = obj['center']
            width = obj['width']
            height = obj['height']
            left_up = (int(center[0] - width / 2), int(center[1] - height / 2))
            right_bottom = (int(center[0] + width / 2), int(center[1] + height / 2))
            if self.frame_count - obj['last_frame_seen'] > self.T_lost:
                if (left_up[0] < 0 or left_up[0] > self.frame_dim[0]) and (right_bottom[0] < 0 or right_bottom[0] > self.frame_dim[0]):
                    if (left_up[1] < 0 or left_up[1] > self.frame_dim[1]) and (right_bottom[1] < 0 or right_bottom[1] > self.frame_dim[1]):
                        self.object_track[obj_idx]['active_track'] = False



        # TASK: DISPLAY TRACKS
        for obj in self.object_track:

            center = obj['center']
            width = obj['width']
            height = obj['height']
            class_name = obj['class']
            id_val = obj['id']

            if obj['active_track']:

                left_up = (int(center[0] - width / 2), int(center[1] - height / 2))
                right_bottom = (int(center[0] + width / 2), int(center[1] + height / 2))
                color = COLORS[VOC_CLASSES.index(class_name)]
                cv2.rectangle(frame, left_up, right_bottom, color, 2)
                label = str(id_val)
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (left_up[0], left_up[1] - text_size[1])
                cv2.rectangle(frame, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(frame, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        cv2.imwrite(frame_filename, frame)

        self.frame_count += 1
        self.frame_list.append([frame])

        return frame

    def create_objects(self, result):

        object_transform = []
        
        for obj_idx, obj in enumerate(result):
            
            tp_lt, btm_rt, cls_prob, filename, prob = obj

            center = ((tp_lt[0] + btm_rt[0]) / 2, (tp_lt[1] + btm_rt[1]) / 2)
            height = abs(tp_lt[1] - btm_rt[1])
            width = abs(tp_lt[0] - btm_rt[0])

            new_object = dict()
            new_object['id'] = obj_idx
            new_object['center'] = center
            new_object['height'] = height
            new_object['width'] = width
            new_object['velocity'] = (0, 0)
            new_object['class'] = obj[2]
            new_object['prob'] = obj[4]
            new_object['last_frame_seen'] = self.frame_count
            new_object['active_track'] = True

            object_transform.append(new_object)

        return object_transform


    def get_frames(self):
        return self.frame_list
         

        
    