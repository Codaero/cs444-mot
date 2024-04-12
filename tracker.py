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
- class : string
- prob : float
- last_frame_seen: int
- active_track: bool
'''

'''
Frames List Attribute:
- Frame Number
- Object List
'''

class Tracker:
    
    def __init__(self, net, output_folder):
        self.frame_count = 0
        self.frame_list = []
        self.net = net
        self.object_track = []
        self.output_folder = output_folder
        self.T_lost = 30


    def process_frame(self, frame):
        frame_filename = os.path.join(self.output_folder, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # result - [[(x1, y1) <top left>, (x2, y2) <bottom right>, VOC_CLASSES[cls_index] <class prob>, image_name <filename>, prob <confidence>], ... ]
        result = predict_image(self.net, frame_filename, root_img_directory="")

        objects = self.create_objects(result)
        if self.frame_count == 0:
            # assign unique ids to every object
            for ids, obj in enumerate(objects):
                center, height, width, cls_prob, prob = obj
                self.object_track.append([ ids, center, height, width, cls_prob, prob, 0, True ])
        else:
            # check previous frame for object assignment
            self.object_track = object_assign(objects, self.object_track, self.frame_count)


        # check out-of-commision
        for obj_idx, obj in enumerate(self.object_track):

            if self.frame_count - obj[6] > self.T_lost:

                self.object_track[obj_idx][7] = False



        # draw rectangles
        for id_val, center, height, width, class_name, prob, last_seen, active_track in self.object_track:

            if active_track:

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
        
        for obj in result: 
            
            tp_lt, btm_rt, cls_prob, filename, prob = obj

            center = ((tp_lt[0] + btm_rt[0]) / 2, (tp_lt[1] + btm_rt[1]) / 2)
            height = abs(tp_lt[1] - btm_rt[1])
            width = abs(tp_lt[0] - btm_rt[0])

            object_transform.append([center, height, width, cls_prob, prob])

        return object_transform


    def get_frames(self):
        return self.frame_list
         

        
    
