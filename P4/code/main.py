from utils import ImageEncodingDataset
from utils import GraphDataset
from utils import MAPE
from utils import train
from utils import getEmbedding
from utils import train_graph
from model import Encoder
from model import Embedding
from model import GraphNet
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
import os

# # Train embedding model
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = Encoder().to(device)
# # model.load_state_dict(torch.load(r'Weights/Epoch/295.pt'))
# learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# trainData = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images/embedding')
# valData = ImageEncodingDataset(r'Data/Encoding/val.csv', r'../Data_full/Images/embedding')

# training_dataloader = DataLoader(trainData, 
#                                  batch_size=1, 
#                                  shuffle=False, 
#                                  num_workers=10)
# validation_dataloader = DataLoader(valData, 
#                                  batch_size=1, 
#                                  shuffle=False, 
#                                  num_workers=10)
# epoch = 3000
# weight_path = r'Weights'
# best = 100
# train(training_dataloader, validation_dataloader, model, epoch, optimizer, MAPE, device, weight_path, best)

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path_se = r'Data/X_ave.csv'
# data_path_image = r'Data/Encoding.csv'
data_path_image = r'../Data_full/X/scene.csv'
data_path_y = r'Data/Hour_Y_masked.csv'
adjacentMtxPath = r'Data/adjacentMatrix.csv'

data = GraphDataset(data_path_se, data_path_image, data_path_y, adjacentMtxPath)
graphtrain, train, val, test = data.get_data()
graphtrain.to(device)

graphtest, train, val, test = data.get_data()
graphtest.to(device)

model = GraphNet()
model.to(device)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

weight_path = r'Weights/Graph'
best = 100000
epoch = 3000

train_graph(graphtrain, graphtest, train, val,test, model, optimizer, device, weight_path, MAPE, best, epoch)




