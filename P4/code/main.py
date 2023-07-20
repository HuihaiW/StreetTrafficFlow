from utils import ImageEncodingDataset
import torch
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images')

print(data[0][0].shape)
# a = torch.tensor([1, 2, 3]).to(device)
# b = torch.tensor([1, 2, 3]).to(device)
# c = [a, b]

# print(c)
# print(os.getcwd())
# path = r'D'