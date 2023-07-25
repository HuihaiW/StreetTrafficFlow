from utils import ImageEncodingDataset
from model import Encoder
import torch
import numpy as np
import os

def train(dataloader):
    for data in dataloader:
        continue
        


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images/embedding')

test_data = data[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder().to(device)
model = model.float()
im1 = test_data[0].to(device).float()
im2 = test_data[1].to(device).float()
im3 = test_data[2].to(device).float()
im4 = test_data[3].to(device).float()
im5 = test_data[4].to(device).float()
x = test_data[5].to(device).float()
input = [im1, im2, im3, im4, im5, x]

print(x.shape)
result = model(input)

print(result.shape)