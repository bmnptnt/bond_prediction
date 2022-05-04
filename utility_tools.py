import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import dataloader


def data_encoding(data):

    le=LabelEncoder()
    le.fit(data)
    encoded_data=le.transform(data)

    return encoded_data

def generate_dataset(data, period,batch_size):
    xy_data = []
    for i in range(len(data) - period):
        x = data[i:i + period].drop('label', axis=1)
        y = data['label'][i + period]
        x_np = np.array(x)
        x_np = np.transpose(x_np, (1, 0))
        y_np = np.array(y)
        xy_data.append([x_np, y_np])

    xy_loader=dataloader.DataLoader(dataset=xy_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return xy_loader
