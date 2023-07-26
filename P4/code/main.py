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
# model = Encoder().to(device)
# # model.load_state_dict(torch.load(r'Weights/Epoch/295.pt'))
# learning_rate = 0.0001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# trainData = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images/embedding')
# valData = ImageEncodingDataset(r'Data/Encoding/val.csv', r'../Data_full/Images/embedding')

# training_dataloader = DataLoader(trainData, 
#                                  batch_size=20, 
#                                  shuffle=False, 
#                                  num_workers=10)
# validation_dataloader = DataLoader(valData, 
#                                  batch_size=20, 
#                                  shuffle=False, 
#                                  num_workers=10)
# epoch = 3000
# weight_path = r'Weight'
# best = 100
# train(training_dataloader, validation_dataloader, model, epoch, optimizer, MAPE, device, weight_path, best)

# Creating new embeddings
# df = pd.read_csv(r'Data/X_ave.csv')
# SVIID_L = df['SVIID'].values.tolist()
# print(df.columns)

DataPth = r'Data/X_Y_whole.csv'
rootFld = r'../Data_full/Images/embedding'
model = Embedding().to(device)
model.load_state_dict(torch.load(r'Weights/best_2.pt'))
result = getEmbedding(DataPth, rootFld, model, device)

