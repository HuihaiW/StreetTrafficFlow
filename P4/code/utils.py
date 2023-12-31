from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        y = torch.tensor((self.y[idx, :] - 498)/543.4)

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

        mark = SVIID

        return [svi1, svi2, svi3, svi4, rm, x, y, mark]

class GraphDataset():
    
    def __init__(self, data_path_se, data_path_image, data_path_y, adjacentMtxPath):

        self.adjacent_matrix = pd.read_csv(adjacentMtxPath)
        self.From= self.adjacent_matrix['From'].values
        self.To = self.adjacent_matrix['To'].values
        self.edge = torch.tensor([self.From, self.To], dtype=torch.long)

        self.data_se = pd.read_csv(data_path_se)
        self.x_columns = ['StreetWidt', 'Length', 'Commercial', 'CulturalFa', 'EducationF',
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

        self.x_se = self.data_se[self.x_columns]
        # self.x_se = (self.x_se - self.x_se.mean())/self.x_se.std()
        self.x_se = self.x_se.values

        self.x_phy = self.x_se[:, 0:2]
        self.x_poi = self.x_se[:, 2:15]
        self.x_social = self.x_se[:, 15:]

        self.x_phy = (self.x_phy - self.x_phy.mean())/self.x_phy.std()
        self.x_poi = (self.x_poi - self.x_poi.mean())/self.x_poi.std()
        self.x_social = (self.x_social - self.x_social.mean())/self.x_social.std()

        self.x_se = np.concatenate([self.x_phy, self.x_poi, self.x_social], 1)
        self.x_se =torch.tensor(self.x_se, dtype=torch.float)
        # self.x_se = (self.x_se - self.x_se.mean())/self.x_se.std
        # self.x_se = torch.tensor(self.x_se, dtype=torch.float)

        self.x_image = pd.read_csv(data_path_image).drop(columns=['Unnamed: 0']).values
        self.x_image = torch.tensor(self.x_image, dtype=torch.float)

        self.x = torch.concat([self.x_se, self.x_image], 1)

        # self.y_columns = ['7', '8', '9', '16', '17', '18', '19', '20']
        self.y_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                          '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
        self.y_df = pd.read_csv(data_path_y)
        self.y = self.y_df[self.y_columns].values


        # repeat the dataset to make sure the distribution of the target is even
        maxes = self.y.max(axis=1)

        max_tensor = torch.tensor(maxes)
        self.repeat_list = [1] * max_tensor.shape[0]

        m_1000_2000_1 = max_tensor > 1000
        m_1000_2000_2 = max_tensor < 2000
        mask_1000_2000 = torch.logical_and(m_1000_2000_1, m_1000_2000_2)
        idx_1000_2000 = mask_1000_2000.nonzero(as_tuple=True)[0]

        m_2000 = max_tensor >= 2000
        idx_2000 = m_2000.nonzero(as_tuple=True)[0]

        # for i in idx_1000_2000:
        #     self.repeat_list[i] = 4
        # for i in idx_2000:
        #     self.repeat_list[i] = 20

        self.repeat_list = torch.tensor(self.repeat_list)
        self.y = torch.tensor(self.y, dtype=torch.float)

        self.all_mask = self.y[:, 0] != -1

        # self.y = torch.repeat_interleave(self.y, self.repeat_list, dim=0)

        
        # self.y = (self.y - 497)/543.4
        
        # self.y = torch.sum(self.y, 1)

        # self.train_mask = self.y_df['Train_mask'].values
        # self.val_mask = self.y_df['Val_mask'].values
        # self.test_mask = self.y_df['Test_mask'].values

    # def __len__(self):
    #     return self.x.shape[0]

    def get_data(self):
        graph = Data(x=self.x, edge_index=self.edge, y=self.y)
        return (graph, self.all_mask, self.repeat_list)




def MAPE(pred, real):

    # pred = pred * 543.4 + 497.0
    # real = real * 543.4 + 497.0
    mask2 = (real > 60)
    # mask3 = real < 4000

    # mask = torch.logical_and(mask2, mask3)


    real = real[mask2]
    pred = pred[mask2]
    error = torch.mean(torch.abs((pred - real)/(real)))
    # print(error)
    return error

def train(trainloader, valloader, model, epoch, optimizer, loss_function,
          device, weights_path, best):
    train_loss = []
    val_loss = []
    for i in range(epoch):
        epoch_loss = []
        idx = 0
        # Do training
        for data in tqdm(trainloader):
            idx += 1
            optimizer.zero_grad()
            x1 = data[0].to(device).float()
            x2 = data[1].to(device).float()
            x3 = data[2].to(device).float()
            x4 = data[3].to(device).float()
            x5 = data[4].to(device).float()
            x6 = data[5].to(device).float()
            y = data[6].to(device).float()
            mark = data[7]
            input = [x1, x2, x3, x4, x5, x6]
            
            output = model(input)

            l = loss_function(output, y)

            l.backward()
            optimizer.step()
            # if l > 50:
            #     print('a')
            #     print(idx, mark)

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
            best_path = os.path.join(weights_path, 'best_3.pt')
            torch.save(model.state_dict(), best_path)
        if i%50 == 0:
            epoch_path = os.path.join(weights_path, 'Epoch/' + str(i) + '_3.pt')
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
            result = result + embedding.detach().cpu().tolist()
            return result
        
def train_graph(mask, graph, graph_test, k_fold, model, optimizer, 
                repeat_list, weight_path, loss_function, best, epoch):
    train_loss = []
    val_loss = []

    # mask = y_origin[:, 0] != -1
    indexes = mask.nonzero(as_tuple=True)[0]
    indexes = indexes.cpu().numpy()
    np.random.shuffle(indexes)
    train_mask_index = indexes[0:1400]
    test_mask_index = indexes[1400:]
    # train_mask_index = indexes[0:3000]
    # test_mask_index = indexes[3000:]

    train_mask = [False] * mask.shape[0]
    val_mask = [False] * mask.shape[0]
    test_mask = [False] * mask.shape[0]

    for idx in test_mask_index:
        test_mask[idx] = True
    

    for k in range(k_fold):

        print('*******Training fold: ' + str(k) + '*******')
        print('***********************************')

        length = int(train_mask_index.shape[0] / k_fold)
        val_index = train_mask_index[k*length:(k+1)*length]
        train_index = np.concatenate([train_mask_index[0:k*length], train_mask_index[(k+1)*length:]])
        for v in val_index:
            val_mask[v] = True
        for t in train_index:
            train_mask[t] = True
        
        train_mask1 = torch.tensor(train_mask)
        # train_mask1 = torch.repeat_interleave(train_mask, repeat_list.cpu(), dim=0)

        val_mask1 = torch.tensor(val_mask)
        # val_mask1 = torch.repeat_interleave(val_mask, repeat_list.cpu(), dim=0)

        for i in range(epoch):
            optimizer.zero_grad()
            output = model(graph)
            # print(output.shape)
            # print(output)
            # print(repeat_list)
            # output = torch.repeat_interleave(output, repeat_list, dim=0)
            # real_mask = graph.y[train_mask] > 0
            # y = torch.repeat_interleave(graph.y, repeat_list, dim=0)
            l = loss_function(output[train_mask1], graph.y[train_mask1])
            l.backward()
            optimizer.step()

            with torch.no_grad():
                out = model(graph_test)
                # out = torch.repeat_interleave(out, repeat_list, dim=0)
                lV = loss_function(out[val_mask1], graph.y[val_mask1])

            print('Fold: ' + str(k) + ',    epoch: ' + str(i) + ',   Train error: ' + str(l.detach().cpu().tolist()) + 
                ', Val error: ' + str(lV.detach().cpu().tolist()))


            if lV < best:
                best = lV
                best_path = os.path.join(weight_path, 'best.pt')
                torch.save(model.state_dict(), best_path)
            if i%100 == 0:
                epoch_path = os.path.join(weight_path, 'Epoch/' + str(i) + '.pt')
                torch.save(model.state_dict(), epoch_path)

        # model.load_state_dict(torch.load(best_path))
    final_path = os.path.join(weight_path, 'Fianl.pt')    
    torch.save(model.state_dict(), final_path)
    return test_mask

def draw_result(predict, real, i, path):
    mask = real[:, 0] != -1

    real = real[mask]
    predict = predict[mask]

    real_v = real[i, :]
    pred_v = predict[i, :]

    out_path = os.path.join(path, str(i) + '.jpg')

    plt.plot(real_v, label='real')
    plt.plot(pred_v, label='pred')
    plt.legend()

    plt.savefig(out_path)
    plt.clf()









