from torch import flatten
from torch_geometric.nn import GATConv
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

        # self.relu = nn.ReLU(inplace=True)
        # self.act = nn.Tanh()
        self.act = nn.ReLU(inplace=True)
        self.act1 = nn.Sigmoid()

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
        x6 = self.act1(x6)
        x6 = self.fc_x2(x6)
        x6 = self.act1(x6)
        x6 = self.fc_x3(x6)
        x6 = self.act1(x6)

        x1 = self.fc1(x1)
        x1 = self.act(x1)
        x1 = self.fc2(x1)
        x1 = self.act(x1)

        x2 = self.fc1(x2)
        x2 = self.act(x2)
        x2 = self.fc2(x2)
        x2 = self.act(x2)

        x3 = self.fc1(x3)
        x3 = self.act(x3)
        x3 = self.fc2(x3)
        x3 = self.act(x3)

        x4 = self.fc1(x4)
        x4 = self.act(x4)
        x4 = self.fc2(x4)
        x4 = self.act(x4)

        x5 = self.fc1_r(x5)
        x5 = self.act(x5)
        x5 = self.fc2_r(x5)
        x5 = self.act(x5)

        x_svi = (x1 + x2 + x3 + x4)/4

        x = torch.cat([x_svi, x5, x6], 1)

        x = self.fc3(x)
        x = self.act(x)
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
        x6 = self.act(x6)
        x6 = self.fc_x2(x6)
        x6 = self.act(x6)
        x6 = self.fc_x3(x6)
        x6 = self.act(x6)

        x1 = self.fc1(x1)
        x1 = self.act(x1)
        x1 = self.fc2(x1)
        x1 = self.act(x1)

        x2 = self.fc1(x2)
        x2 = self.act(x2)
        x2 = self.fc2(x2)
        x2 = self.act(x2)

        x3 = self.fc1(x3)
        x3 = self.act(x3)
        x3 = self.fc2(x3)
        x3 = self.act(x3)

        x4 = self.fc1(x4)
        x4 = self.act(x4)
        x4 = self.fc2(x4)
        x4 = self.act(x4)

        x5 = self.fc1_r(x5)
        x5 = self.act(x5)
        x5 = self.fc2_r(x5)
        x5 = self.act(x5)

        x_svi = (x1 + x2 + x3 + x4)/4

        x = torch.cat([x_svi, x5, x6], 1)

        x = self.fc3(x)
        x = self.act(x)

        return x

class GraphNet(torch.nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        
        self.phy1 = nn.Linear(2, 32)
        self.phy2 = nn.Linear(32, 64)

        self.poi1 = nn.Linear(13, 64)
        self.poi2 = nn.Linear(64, 64)
        
        self.se1 = nn.Linear(40, 128)
        self.se2 = nn.Linear(128, 128)

        # self.scene1 = nn.Linear(1280, 64)
        self.scene1 = nn.Linear(365, 64)
        self.scene2 = nn.Linear(64, 64)

        self.conv1 = GATConv(128 + 64 + 64 + 64, 128)
        self.conv2 = GATConv(128, 64)
        self.conv3 = GATConv(64, 64)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 24)

        self.act1 = nn.Sigmoid()
        self.act2 = nn.ReLU()
        self.act3 = nn.Tanh()

    def forward(self, graph):
        x_all, edge_index = graph.x, graph.edge_index

        x_phy = x_all[:, 0:2]
        x_poi = x_all[:, 2:15]
        x_se = x_all[:, 15:55]
        x_scene = x_all[:, 55:]

        x_phy = self.phy1(x_phy)
        x_phy = self.act2(x_phy)
        x_phy = self.phy2(x_phy)
        x_phy = self.act2(x_phy)

        x_poi = self.poi1(x_poi)
        x_poi = self.act2(x_poi)
        x_poi = self.poi2(x_poi)
        x_poi = self.act2(x_poi)

        x_se = self.se1(x_se)
        x_se = self.act2(x_se)
        x_se = self.se2(x_se)
        x_se = self.act2(x_se)

        x_scene = self.scene1(x_scene)
        x_scene = self.act1(x_scene)
        x_scene = self.scene2(x_scene)
        x_scene = self.act2(x_scene)

        x = torch.cat((x_phy, x_poi, x_se, x_scene), 1)


        # x = torch.cat((x_phy, x_poi, x_se), 1)

        x = self.conv1(x, edge_index)
        x = self.act2(x)
        # x = F.dropout(x, p=0.5)

        x = self.conv2(x, edge_index)
        x = self.act2(x)
        # x = F.dropout(x, p=0.5)

        # x = self.conv3(x, edge_index)
        # x = self.act2(x)
        # x = F.dropout(x, p=0.5)

        x = self.fc1(x)
        x = self.act2(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = self.act2(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        
        return x


        

# model = Embedding()
# print(model)



