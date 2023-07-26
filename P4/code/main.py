from utils import ImageEncodingDataset
from utils import MAPE
from utils import train
from utils import getEmbedding
from model import Encoder
from model import Embedding
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder().to(device)
# model.load_state_dict(torch.load(r'Weights/Epoch/295.pt'))
learning_rate = 0.01
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
weight_path = r'Weights'
best = 100
train(training_dataloader, validation_dataloader, model, epoch, optimizer, MAPE, device, weight_path, best)

# Creating new embeddings

# DataPth = r'Data/X_Y_whole.csv'
# rootFld = r'../Data_full/Images/embedding'
# model = Embedding().to(device)
# model.load_state_dict(torch.load(r'Weights/best_2.pt'))
# result = getEmbedding(DataPth, rootFld, model, device)
# # print(len(result[0]))
# df = pd.DataFrame(result)
# df.to_csv(r'Data/embeddings.csv')

# # normalize the target
# df = pd.read_csv(r'Data/Hour_Y.csv').drop(columns=['Unnamed: 0'])
# df_values = df.values
# mask = df_values != -1
# df_values = df_values[mask]
# print(df_values.mean())
# print(df_values.std())
# print(df_values.shape)
