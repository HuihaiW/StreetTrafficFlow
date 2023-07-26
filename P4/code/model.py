from torch import flatten

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50

from segment_anything import SamPredictor, sam_model_registry

import cv2

import warnings
warnings.filterwarnings("ignore")

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model1 = resnet50(pretrained=False)
        self.model1.conv1 = nn.Conv2d(256, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model2 = resnet50(pretrained=False)
        self.model2.conv1 = nn.Conv2d(256, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.fc_x1 = nn.Linear(in_features=55, out_features=32, bias=True)
        self.fc_x2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.fc_x3 = nn.Linear(in_features=16, out_features=16, bias=True)

        self.fc1 = nn.Linear(in_features=1000, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=16, bias=True)

        self.fc1_r = nn.Linear(in_features=1000, out_features=64, bias=True)
        self.fc2_r = nn.Linear(in_features=64, out_features=16, bias=True)

        self.fc3 = nn.Linear(in_features=48, out_features=32, bias=True)
        self.fc4 = nn.Linear(in_features=32, out_features=24, bias=True)


    def forward(self, data):
        x1 = self.model1(data[0])
        x2 = self.model1(data[1])
        x3 = self.model1(data[2])
        x4 = self.model1(data[3])
        x5 = self.model2(data[4])

        x6 = data[5]
        x6 = self.fc_x1(x6)
        x6 = self.relu(x6)
        x6 = self.fc_x2(x6)
        x6 = self.relu(x6)
        x6 = self.fc_x3(x6)
        x6 = self.relu(x6)

        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)

        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)

        x3 = self.fc1(x3)
        x3 = self.relu(x3)
        x3 = self.fc2(x3)
        x3 = self.relu(x3)

        x4 = self.fc1(x4)
        x4 = self.relu(x4)
        x4 = self.fc2(x4)
        x4 = self.relu(x4)

        x5 = self.fc1_r(x5)
        x5 = self.relu(x5)
        x5 = self.fc2_r(x5)
        x5 = self.relu(x5)

        x_svi = (x1 + x2 + x3 + x4)/4

        x = torch.cat([x_svi, x5, x6], 1)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

class Embedding(Encoder):
    def __init__(self):
        super(Embedding, self).__init__()
    def forward(self, data):
        x1 = self.model1(data[0])
        
        x2 = self.model1(data[1])
        x3 = self.model1(data[2])
        x4 = self.model1(data[3])
        x5 = self.model2(data[4])

        x6 = data[5]
        x6 = self.fc_x1(x6)
        x6 = self.relu(x6)
        x6 = self.fc_x2(x6)
        x6 = self.relu(x6)
        x6 = self.fc_x3(x6)
        x6 = self.relu(x6)

        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)

        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)

        x3 = self.fc1(x3)
        x3 = self.relu(x3)
        x3 = self.fc2(x3)
        x3 = self.relu(x3)

        x4 = self.fc1(x4)
        x4 = self.relu(x4)
        x4 = self.fc2(x4)
        x4 = self.relu(x4)

        x5 = self.fc1_r(x5)
        x5 = self.relu(x5)
        x5 = self.fc2_r(x5)
        x5 = self.relu(x5)

        x_svi = (x1 + x2 + x3 + x4)/4

        x = torch.cat([x_svi, x5, x6], 1)

        x = self.fc3(x)
        x = self.relu(x)

        return x

# model = Embedding()
# print(model)



