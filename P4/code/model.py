from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

import torch
import torch.nn.functional as F
import torch.nn as nn

from segment_anything import SamPredictor, sam_model_registry

import cv2

import warnings
warnings.filterwarnings("ignore")

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.checkpoint = "../Data_full/SegmentAnything/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        self.model = self.model.image_encoder


        self.conv1 = nn.Conv2d(256, 32, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(32, 24, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn2 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc = nn.Linear(in_features=512, out_features=1000, bias=True)

        self.conv1_2 = nn.Conv2d(256, 32, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn1_2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_2 = nn.ReLU(inplace=True)

        # self.conv2_2 = nn.Conv2d(32, 24, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn2_2 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.avgpool_2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(in_features=64, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=24, bias=True)


    def forward(self, data):
        x1 = self.model(data[0])
        x2 = self.model(data[1])
        x3 = self.model(data[2])
        x4 = self.model(data[3])
        x5 = self.model(data[4])
        
        # x1 = self.model2(x1)
        # x2 = self.model2(x2)
        # x3 = self.model2(x3)
        # x4 = self.model2(x4)
        # x5 = self.model2(x5)

        # x = (x1 + x2 + x3 + x4 + x5) / 5
        x1 = self.conv1(x1)
        # x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # x1 = self.conv2(x1)
        # x1 = self.bn2(x1)
        x1 = self.avgpool(x1)

        x2 = self.conv1(x2)
        # x2 = self.bn1(x2)
        x2 = self.relu(x2)
        # x2 = self.conv2(x2)
        # x2 = self.bn2(x2)
        x2 = self.avgpool(x2)

        x3 = self.conv1(x3)
        # x3 = self.bn1(x3)
        x3 = self.relu(x3)
        # x3 = self.conv2(x3)
        # x3 = self.bn2(x3)
        x3 = self.avgpool(x3)

        x4 = self.conv1(x4)
        # x4 = self.bn1(x4)
        x4 = self.relu(x4)
        # x4 = self.conv2(x4)
        # x4 = self.bn2(x4)
        x4 = self.avgpool(x4)

        x5 = self.conv1_2(x5)
        # x5 = self.bn1_2(x5)
        x5 = self.relu_2(x5)
        # x5 = self.conv2_2(x5)
        # x5 = self.bn2_2(x5)
        x5 = self.avgpool_2(x5)
        
        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)
        x4 = torch.flatten(x4)
        x5 = torch.flatten(x5)

        x_svi = (x1 + x2 + x3 + x4)/4

        print(x_svi.shape)
        print(x5.shape)
        x = torch.cat((x_svi, x5), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

# model = Encoder().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
# print(model)




