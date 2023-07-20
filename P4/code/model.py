import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
import pandas as pd

import statistics
from tqdm import tqdm
import numpy as np
import random
import os
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Dataset
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import sklearn
from segment_anything import SamPredictor, sam_model_registry
from math import sqrt
import random
from tqdm import tqdm
import cv2

import warnings
warnings.filterwarnings("ignore")

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.checkpoint = "../Data_full/SegmentAnything/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)

    def forward(self, data):
        self.predictor.set_image(data)
        image_embedding = self.predictor.get_image_embedding()
        return image_embedding

model = Encoder()
img = cv2.imread(r'test.png')
result = model(img)
print(result)

