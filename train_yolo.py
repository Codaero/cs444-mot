import os
import random
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import models

from src.resnet_yolo import resnet50
from yolo_loss import YoloLoss
from src.dataset import VocDetectorDataset




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_network_path = None #'checkpoints/best_detector.pth' 
pretrained = True

if load_network_path is not None:
    print('Loading saved network from {}'.format(load_network_path))
    net = resnet50().to(device)
    net.load_state_dict(torch.load(load_network_path))
else:
    print('Load pre-trained model')
    net = resnet50(pretrained=pretrained).to(device)

'''
Parameters: 
B - number of bounding box predictions per cell
S - width/height of network output grid (larger than 7x7 from paper since we use a different network) 
learning_rate - SGD learning rate
num_epochs - amount of iterations
batch_size - train size per epoch
lambda_coord & lambda_noobj - hyper params for yolo loss
'''

B = 2
S = 14
learning_rate = 0.001
num_epochs = 50
batch_size = 24
lambda_coord = 5
lambda_noobj = 0.5
criterion = YoloLoss(S, B, lambda_coord, lambda_noobj)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)



def load_train():
    file_root_train = 'data/VOCdevkit_2007/VOC2007/JPEGImages/'
    annotation_file_train = 'data/voc2007.txt'

    train_dataset = VocDetectorDataset(root_img_dir=file_root_train,dataset_file=annotation_file_train,train=True, S=S)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
    print('Loaded %d train images' % len(train_dataset))

    return train_loader


def train_yolo():
    train_loader = load_train()

    learning_rate = 1e-3
    for epoch in range(num_epochs):
        net.train()
        
        # Update learning rate late in training
        if epoch == 30 or epoch == 40:
            learning_rate /= 10.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        
        total_loss = collections.defaultdict(int)
        
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            data = (item.to(device) for item in data)
            images, target_boxes, target_cls, has_object_map = data
            pred = net(images)
            
            loss_dict = criterion(pred, target_boxes, target_cls, has_object_map)
            
            for key in loss_dict:
                total_loss[key] += loss_dict[key].item()
            
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            
        
        torch.save(net.state_dict(),'checkpoints/detector.pth')        

        

if __name__ == '__main__':
    train_yolo()