from utils import ImageEncodingDataset
from utils import MAPE
from utils import train
from model import Encoder
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder().to(device)
# model.load_state_dict(torch.load(r'Weights/Epoch/295.pt'))
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
trainData = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images/embedding')
valData = ImageEncodingDataset(r'Data/Encoding/val.csv', r'../Data_full/Images/embedding')

training_dataloader = DataLoader(trainData, 
                                 batch_size=20, 
                                 shuffle=False, 
                                 num_workers=10)
validation_dataloader = DataLoader(valData, 
                                 batch_size=20, 
                                 shuffle=False, 
                                 num_workers=10)
epoch = 3000
weight_path = r'Weight'
best = 100
train(training_dataloader, validation_dataloader, model, epoch, optimizer, MAPE, device, weight_path, best)

