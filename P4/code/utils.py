from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os
import cv2


class ImageEncodingDataset(Dataset):
    
    def __init__(self, data_path, root_folder):
        self.svi_folder = os.path.join(root_folder, 'SVI')
        self.rm_folder = os.path.join(root_folder, 'RM')
        self.data = pd.read_csv(data_path)
        self.x_columns = ['SVIID','StreetWidt', 'Length', 'Commercial', 'CulturalFa', 'EducationF',
                           'Government', 'HealthServ', 'Miscellane', 'PublicSafe', 'Recreation',
                           'ReligiousI', 'Residentia', 'SocialServ', 'Transporta', 'Water',
                           'Avg_B01001', 'Avg_B010_1', 'Avg_B010_2', 'Avg_B010_3', 'Avg_B02001',
                           'Avg_B020_1', 'Avg_B020_2', 'Avg_B08006', 'Avg_B080_1', 'Avg_B080_2',
                           'Avg_B08013', 'Avg_B08124', 'Avg_B15003', 'Avg_B19001', 'Avg_B19013',
                           'Avg_B23013', 'Avg_B24011', 'Avg_B240_1', 'Avg_B240_2', 'Avg_B240_3',
                           'Avg_B240_4', 'Avg_B240_5', 'Avg_B240_6', 'Avg_B240_7', 'Avg_B240_8',
                           'Avg_B240_9', 'Avg_B24_10', 'Avg_B24_11', 'Avg_B24_12', 'Avg_B24_13',
                           'Avg_B24_14', 'Avg_B24_15', 'Avg_B24_16', 'Avg_B24_17', 'Avg_B24_18',
                           'Avg_B24_20', 'Avg_B24_21', 'Avg_B24_22', 'Avg_B24_23', 'Avg_B24_24']
        self.y_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                          '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
        self.x = self.data[self.x_columns].values
        self.y = self.data[self.y_columns].values
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Resize((1024, 1024)),
        #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        SVIID = str(int(self.x[idx, 0]))
        x = torch.tensor(self.x[idx, 1:])
        y = torch.tensor(self.y[idx, :])

        try:
            svi1 = np.load(os.path.join(self.svi_folder, SVIID + '_0.npy'))
            svi1 = np.squeeze(svi1)
        except:
            svi1 = np.zeros((256, 64, 64))
        try:
            svi2 = np.load(os.path.join(self.svi_folder, SVIID + '_90.npy'))
            svi2 = np.squeeze(svi2)
        except:
            svi2 = np.zeros((256, 64, 64))
        try:
            svi3 = np.load(os.path.join(self.svi_folder, SVIID + '_180.npy'))
            svi3 = np.squeeze(svi3)
        except:
            svi3 = np.zeros((256, 64, 64))
        try:
            svi4 = np.load(os.path.join(self.svi_folder, SVIID + '_270.npy'))
            svi4 = np.squeeze(svi4)
        except:
            svi4 = np.zeros((256, 64, 64))

        svi1 = torch.from_numpy(svi1)
        svi2 = torch.from_numpy(svi2)
        svi3 = torch.from_numpy(svi3)
        svi4 = torch.from_numpy(svi4)

        rm = np.load(os.path.join(self.rm_folder, SVIID + '.npy'))
        rm = np.squeeze(rm)
        rm = torch.from_numpy(rm)

        return [svi1, svi2, svi3, svi4, rm, x, y]


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

        print('Epoch: ' + str(i) + ',    Training loss is: ' + str(aveTLoss) + ',  validation loss is: ' + str(aveVLoss))

        if aveVLoss < best:
            best = aveVLoss
            best_path = os.path.join(weights_path, 'best_2.pt')
            torch.save(model.state_dict(), best_path)
        if i%50 == 0:
            epoch_path = os.path.join(weights_path, 'Epoch/' + str(i) + '_2.pt')
            torch.save(model.state_dict(), epoch_path)

def getEmbedding(DataPth, rootFld, model, device):
    result = []
    dataset = ImageEncodingDataset(DataPth, rootFld)
    dataLoader = DataLoader(dataset,
                            batch_size=20,
                            shuffle=False,
                            num_workers=10)
    model = model.eval()
    for data in dataLoader:
        with torch.no_grad():
            x1 = data[0].to(device).float()
            x2 = data[1].to(device).float()
            x3 = data[2].to(device).float()
            x4 = data[3].to(device).float()
            x5 = data[4].to(device).float()
            x6 = data[5].to(device).float()
            input = [x1, x2, x3, x4, x5, x6]
            embedding = model(input)
            result.append(embedding.detach().cpu().tolist())
            return result




