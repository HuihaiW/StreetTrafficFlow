import torch
from model import GraphNet
from utils import GraphDataset
import pandas as pd
import os

net = GraphNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path_se = r'Data/X_ave.csv'
data_path_image = r'Data/Encoding.csv'

data_path_y = r'Data/Hour_Y_masked.csv'
adjacentMtxPath = r'Data/adjacentMatrix.csv'

data = GraphDataset(data_path_se, data_path_image, data_path_y, adjacentMtxPath)
graphtrain, all_mask, repeat_list1= data.get_data()
graphtrain.to(device)

net = net.to(device)
net.load_state_dict(torch.load(r'Weights/Graph/best_all.pt'))
net.eval()
result = net(graphtrain)

result = result.detach().cpu().numpy()
df = pd.DataFrame(result)
df.to_csv(r'Data/Result_Y.csv')