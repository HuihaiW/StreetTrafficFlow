from utils import ImageEncodingDataset
from model import Encoder
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import os

def MAPE(pred, real):
    pred = pred * 150 + 139
    real = real * 150 + 139
    mask_2 = (real!= 0)
    real = real[mask_2]
    pred = pred[mask_2]
    return torch.mean(torch.abs((pred - real)/(real)))

def train(trainloader, valloader, model, epoch, optimizer, loss_function,
          device, weights_path, best):
    train_loss = []
    val_loss = []
    for i in range(epoch):
        epoch_loss = []
        # Do training
        for data in tqdm(trainloader):
            optimizer.zero_grad()
            x1 = data[0].to(device).float()
            x2 = data[1].to(device).float()
            x3 = data[2].to(device).float()
            x4 = data[3].to(device).float()
            x5 = data[4].to(device).float()
            x6 = data[5].to(device).float()
            y = data[6].to(device).float()
            input = [x1, x2, x3, x4, x5, x6]
            
            output = model(input)

            l = loss_function(output, y)

            l.backward()
            optimizer.step()

            epoch_loss.append(l.tolist())
        aveTLoss = sum(epoch_loss)/len(epoch_loss)
        train_loss.append(aveTLoss)

        # Do validation
        with torch.no_grad():
            epoch_loss = []
            for data in valloader:  
                x1 = data[0].to(device).float()
                x2 = data[1].to(device).float()
                x3 = data[2].to(device).float()
                x4 = data[3].to(device).float()
                x5 = data[4].to(device).float()
                x6 = data[5].to(device).float()
                y = data[6].to(device).float()
                input = [x1, x2, x3, x4, x5, x6]

                out_val = model(input)
                l = loss_function(out_val, y)
                epoch_loss.append(l.tolist())
        aveVLoss = sum(epoch_loss)/len(epoch_loss)
        val_loss.append(aveVLoss)

        print('Training loss is: ' + str(aveTLoss) + ',  validation loss is: ' + str(aveVLoss))

        if aveVLoss < best:
            best = aveVLoss
            best_path = os.path.join(weights_path, 'best.pt')
            torch.save(model.state_dict(), best_path)
        if i%5 == 0:
            epoch_path = os.path.join(weights_path, 'Epoch/' + str(i) + '.pt')
            torch.save(model.state_dict(), epoch_path)
            


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder().to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainData = ImageEncodingDataset(r'Data/Encoding/train.csv', r'../Data_full/Images/embedding')
valData = ImageEncodingDataset(r'Data/Encoding/val.csv', r'../Data_full/Images/embedding')

training_dataloader = DataLoader(trainData, 
                                 batch_size=10, 
                                 shuffle=False, 
                                 num_workers=10)
validation_dataloader = DataLoader(valData, 
                                 batch_size=10, 
                                 shuffle=False, 
                                 num_workers=5)
epoch = 300
weight_path = r'Weights'
best = 1000000
train(training_dataloader, validation_dataloader, model, epoch, optimizer, MAPE, device, weight_path, best)

