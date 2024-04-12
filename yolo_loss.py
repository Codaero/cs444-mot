import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.mse = lambda x, y: torch.sum((x - y) ** 2)
        self.N = 0

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        func = lambda v1, v2: v1 / self.S - 0.5 * v2
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([func(x, w), func(y, h), func(x, -w), func(y, -h)], dim=1)

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        bounding_box_preds = [self.xywh2xyxy(pred_box_list[0][:, :4]),
                              self.xywh2xyxy(pred_box_list[1][:, :4])]  # (N*S*S, 4)
        bounding_target = self.xywh2xyxy(box_target)  # (N*S*S, 4)

        iou_preds = [torch.diag(compute_iou(bounding_box_preds[0], bounding_target)),
                     torch.diag(compute_iou(bounding_box_preds[1], bounding_target))]

        best_ious, indices = torch.max(torch.stack([iou_preds[0], iou_preds[1]]), dim=0)

        best_boxes = torch.empty_like(pred_box_list[0])

        for i in range(len(bounding_box_preds[0])):
            best_boxes[i] = pred_box_list[indices[i]][i]

        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        return self.mse(classes_pred[has_object_map], classes_target[has_object_map]) / self.N

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        idx = has_object_map.reshape(-1) == False
        b1, b2 = pred_boxes_list[0].reshape(-1, 5)[idx][:, -1], pred_boxes_list[1].reshape(-1, 5)[idx][:, -1]
        return self.l_noobj * (self.mse(b1, torch.zeros_like(b1)) + self.mse(b2, torch.zeros_like(b2))) / self.N

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        return self.mse(box_pred_conf, box_target_conf.detach()) / self.N

    def get_regression_loss(self, box_pred, box_target):
        """
        Parameters:
        box_pred : (tensor) size (-1, 4)
        box_target : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.reshape_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        xp, yp, wp, hp = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2] ** 0.5, box_pred[:, 3] ** 0.5
        xt, yt, wt, ht = box_target[:, 0], box_target[:, 1], box_target[:, 2] ** 0.5, box_target[:, 3] ** 0.5
        return self.l_coord * (self.mse(xp, xt) + self.mse(yp, yt) + self.mse(wp, wt) + self.mse(hp, ht)) / self.N

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        self.N, total_loss = pred_tensor.size(0), 0.0

        pred_boxes_list, pred_cls = [pred_tensor[..., :5], pred_tensor[..., 5:10]], pred_tensor[..., 10:]

        classification_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        no_object_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        idx = has_object_map.reshape(-1) == True
        bbox_transform = [pred_boxes_list[0].reshape(-1, 5)[idx], pred_boxes_list[1].reshape(-1, 5)[idx]]
        target_transform = target_boxes.reshape(-1, 4)[idx]

        best_iou, best_boxes = self.find_best_iou_boxes(bbox_transform, target_transform)

        reg_loss = self.get_regression_loss(best_boxes[:, :4], target_transform)

        conf_loss = self.get_contain_conf_loss(best_boxes[:, 4], best_iou)

        total_loss = conf_loss + no_object_loss + classification_loss + reg_loss

        return dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=conf_loss,
            no_obj_loss=no_object_loss,
            cls_loss=classification_loss,
        )