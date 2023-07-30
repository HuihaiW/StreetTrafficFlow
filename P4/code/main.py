from utils import ImageEncodingDataset
from utils import GraphDataset
from utils import MAPE
from utils import train
from utils import getEmbedding
from utils import train_graph
from utils import draw_result
from model import Encoder
from model import Embedding
from model import GraphNet
from torch.utils.data import DataLoader
from torch import nn
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
data_path_image = r'Data/Encoding.csv'
# data_path_image = r'../Data_full/X/scene.csv'
data_path_y = r'Data/Hour_Y_masked.csv'
adjacentMtxPath = r'Data/adjacentMatrix.csv'

data = GraphDataset(data_path_se, data_path_image, data_path_y, adjacentMtxPath)
graphtrain, all_mask, repeat_list1= data.get_data()
graphtrain.to(device)

graphtest, all_mask, repeat_list= data.get_data()
print(graphtest.y[graphtest.y != -1].numpy().mean())
print(graphtest.y[graphtest.y != -1].numpy().std())
graphtest.to(device)

repeat_list = repeat_list.to(device)





weight_path = r'Weights/Graph'
best = 10000000000000
loss_best = 10
epoch = 150
k_fold = 10
loss_all = []
for i in range(30):


    model = GraphNet()
    model.to(device)
    loss_function = MAPE
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_function = nn.MSELoss()
    model.train()
    all_mask = torch.tensor(all_mask, dtype=bool)
    test_mask = train_graph(all_mask, graphtrain, graphtest, k_fold, model, optimizer, repeat_list, weight_path, loss_function, best, epoch)
    test_mask_df =  pd.DataFrame(test_mask, columns=['Mask'])
    test_mask_df.to_csv(r'Data/testMask.csv')

    test_mask = pd.read_csv(r'Data/testMask.csv')['Mask'].values
    model.load_state_dict(torch.load(r'Weights/Graph/best.pt'))
    model.eval()
    test_result = model(graphtrain)
    # test_result = torch.repeat_interleave(test_result, repeat_list, dim=0)
    test_result1 = test_result[test_mask]
    loss = MAPE(test_result1, graphtrain.y[test_mask])
    loss = loss.detach().cpu().numpy()
    loss_all.append(loss)
    print("Final loss is: ", loss)

    # draw_result(test_result.detach().cpu().numpy(), graphtest.y.detach().cpu().numpy(), 500)
    if loss < loss_best:
        torch.save(model.state_dict(), r'Weights/Graph/best_all.pt')
        loss_best = loss

        for idx in range(150):
            path = r'Data/Result'
            draw_result(test_result.detach().cpu().numpy()[test_mask], 
                    graphtest.y.detach().cpu().numpy()[test_mask], idx, path)
            
df_loss = pd.DataFrame(loss_all)
df_loss.to_csv(r'Weights/Graph/loss_all.csv')
print(np.array(loss_all).mean())
print(np.array(loss_all).std())





