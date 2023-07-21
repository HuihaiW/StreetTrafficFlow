from utils import ImageEncodingDataset
from model import Encoder
import torch
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images')

test_data = data[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder().to(device)
im1 = test_data[0].to(device)
im2 = test_data[1].to(device)
im3 = test_data[2].to(device)
im4 = test_data[3].to(device)
im5 = test_data[4].to(device)
input = [im1, im2, im3, im4, im5]
result = model(input)

print(result.shape)