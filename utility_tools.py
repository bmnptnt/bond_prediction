import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import dataloader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler

def data_encoding(data):

    le=LabelEncoder()
    le.fit(data)
    encoded_data=le.transform(data)
    print(encoded_data)
    return encoded_data

def data_regularization(data):
    sc=StandardScaler()
    mm=MinMaxScaler()
    transform_data=sc.fit_transform(data)
    transform_data=mm.fit_transform(transform_data)
    return transform_data

def load_data(data_path,data_selec,data_size,label_name):
    print('load the total data...')
    origin_data = pd.read_excel(data_path, engine='openpyxl')
    selected_data = origin_data[data_selec]
    selected_data=selected_data[:data_size+100]
    selected_data = selected_data[::-1].reset_index(drop=True)
    selected_data = selected_data[:data_size]

    selected_data[label_name] = data_encoding(selected_data[label_name])

    selected_data.iloc[:,1:]=data_regularization(selected_data.iloc[:,1:])


    return selected_data

def transform_np_dataloader(data, period,batch_size,label_name):
    xy_data = []

    for i in tqdm(range(len(data) - period)):
        x = data[i:i + period].drop(label_name, axis=1)
        y = data[label_name][i + period]
        x_np = np.array(x)
        x_np = np.transpose(x_np, (1, 0))
        y_np = np.array(y)
        xy_data.append([x_np, y_np])

    xy_loader=dataloader.DataLoader(dataset=xy_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return xy_loader


def generate_dataset(data,valid_scale,period,batch_size,label_name):

    train_split, valid_split = train_test_split(data, test_size=valid_scale, shuffle=False)
    print(len(train_split),len(valid_split))
    print('generate training data...')
    train_loader=transform_np_dataloader(train_split,period,batch_size,label_name)
    print('generate validation data...')
    valid_loader=transform_np_dataloader(valid_split.reset_index(drop=True),period,batch_size,label_name)

    return train_loader, valid_loader

