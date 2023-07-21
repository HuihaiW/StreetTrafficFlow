from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((1024, 1024)),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        SVIID = str(int(self.x[idx, 0]))
        x = torch.tensor(self.x[idx, 1:])
        y = torch.tensor(self.y[idx, :])

        svi1 = cv2.imread(os.path.join(self.svi_folder, SVIID + '_0.jpg'))
        svi2 = cv2.imread(os.path.join(self.svi_folder, SVIID + '_90.jpg'))
        svi3 = cv2.imread(os.path.join(self.svi_folder, SVIID + '_180.jpg'))
        svi4 = cv2.imread(os.path.join(self.svi_folder, SVIID + '_270.jpg'))

        svi1 = self.transform(cv2.cvtColor(svi1, cv2.COLOR_BGR2RGB))
        svi2 = self.transform(cv2.cvtColor(svi2, cv2.COLOR_BGR2RGB))
        svi3 = self.transform(cv2.cvtColor(svi3, cv2.COLOR_BGR2RGB))
        svi4 = self.transform(cv2.cvtColor(svi4, cv2.COLOR_BGR2RGB))

        svi1 = svi1[None, :]
        svi2 = svi2[None, :]
        svi3 = svi3[None, :]
        svi4 = svi4[None, :]

        rm = cv2.imread(os.path.join(self.rm_folder, SVIID + '.png'))
        rm = self.transform(cv2.cvtColor(rm, cv2.COLOR_BGR2RGB))
        rm = rm[None, :]

        return [svi1, svi2, svi3, svi4, rm, x, y]