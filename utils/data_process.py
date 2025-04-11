import xarray as xr
import numpy as np
from utils.util import grid_arithmetics
import torch, math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# 定义数据集类
class ClimateDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def getdata():
    ############################## read data ##############################
    # Load grid data
    mypath = r'D:\climate_data\fh_data/'
    # 读取预测因子 地势高度

    z500 = xr.open_dataset(mypath + "2001-2020_g500.nc")["z"].values
    new_z500 = grid_arithmetics(z500, 9.80665, "/")
    z700 = xr.open_dataset(mypath + "2001-2020_g700.nc")["z"].values
    new_z700 = grid_arithmetics(z700, 9.80665, "/")
    z850 = xr.open_dataset(mypath + "2001-2020_g850.nc")["z"].values
    new_z850 = grid_arithmetics(z850, 9.80665, "/")

    # 比湿
    q500 = xr.open_dataset(mypath + "2001-2020_hus500.nc")["q"].values
    q700 = xr.open_dataset(mypath + "2001-2020_hus700.nc")["q"].values
    q850 = xr.open_dataset(mypath + "2001-2020_hus850.nc")["q"].values

    # 温度
    t500 = xr.open_dataset(mypath + "2001-2020_ta500.nc")["t"].values
    t700 = xr.open_dataset(mypath + "2001-2020_ta700.nc")["t"].values
    t850 = xr.open_dataset(mypath + "2001-2020_ta850.nc")["t"].values

    # 纬向风
    u500 = xr.open_dataset(mypath + "2001-2020_u500.nc")["u"].values
    u700 = xr.open_dataset(mypath + "2001-2020_u700.nc")["u"].values
    u850 = xr.open_dataset(mypath + "2001-2020_u850.nc")["u"].values

    # 经向风
    v500 = xr.open_dataset(mypath + "2001-2020_v500.nc")["v"].values
    v700 = xr.open_dataset(mypath + "2001-2020_v700.nc")["v"].values
    v850 = xr.open_dataset(mypath + "2001-2020_v850.nc")["v"].values

    # 加载降水变量
    y = xr.open_dataset(mypath + "2001-2020_prec.nc")["prec"].values

    arrays = [new_z500, new_z700, new_z850, q500, q700, q850, t500, t700, t850, u500, u700, u850, v500, v700, v850]
    stack_array = np.stack(arrays, axis=1)

    # 将原始 NumPy 数组转换为 PyTorch 张量
    tensor_x = torch.tensor(stack_array, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).transpose(1, 2)
    tensor_y = torch.nan_to_num(tensor_y, nan=0.0)
    return tensor_x, tensor_y


if __name__ == '__main__':
    getdata()


def split_data(start_time, cut_time, tensor_x, tensor_y):
    # 分段时间序列
    train_period_dates = pd.date_range(start=start_time, end=cut_time)

    train_days = len(train_period_dates)

    train_x = tensor_x[:train_days]
    test_x = tensor_x[train_days:]
    train_y = tensor_y[:train_days]
    test_y = tensor_y[train_days:]

    return train_x, test_x, train_y, test_y


"""
按列标准化：fit_transform 是在整个数据集的列上进行标准化的。这意味着每一列（特征）都被独立处理，其标准化是基于该列的数据计算得出的。
"""


def standardize_dataset(train_x, test_x):
    # 标准化处理
    scaler_x = StandardScaler()

    train_x_reshaped = train_x.reshape(train_x.shape[0], -1)
    train_x_scaled = scaler_x.fit_transform(train_x_reshaped).reshape(train_x.shape)

    test_x_reshaped = test_x.reshape(test_x.shape[0], -1)
    test_x_scaled = scaler_x.transform(test_x_reshaped).reshape(test_x.shape)

    tensor_train_x_scaled = torch.tensor(train_x_scaled, dtype=torch.float32)
    tensor_test_x_scaled = torch.tensor(test_x_scaled, dtype=torch.float32)
    return tensor_train_x_scaled, tensor_test_x_scaled


def train_val_split(train_x, train_y, ratio, k_fold=False, index=0):
    # 划分训练集和验证集
    train_y = train_y - 0.99
    train_y[train_y < 0] = 0
    if not k_fold:
        val_len = math.floor(len(train_x) * (1 - ratio))
        val_x = train_x[val_len:]
        train_x = train_x[:val_len]
        val_y = train_y[val_len:]
        train_y = train_y[:val_len]
    else:
        length = math.floor(ratio * len(train_x))
        val_x = train_x[index * length:(index + 1) * length]
        train_x = torch.cat((train_x[:index * length], train_x[(index + 1) * length:]), dim=0)
        val_y = train_y[index * length:(index + 1) * length]
        train_y = torch.cat((train_y[:index * length], train_y[(index + 1) * length:]), dim=0)

    return train_x, val_x, train_y, val_y


def train_val_split_lstm(train_x, train_y, ratio, window=3, k_fold=False, index=0):
    # 划分训练集和验证集
    train_y = train_y - 0.99
    train_y[train_y < 0] = 0
    if not k_fold:
        val_len = math.floor(len(train_x) * (1 - ratio))
        val_x = train_x[val_len:]
        train_x = train_x[:val_len]
        val_y = train_y[val_len:]
        train_y = train_y[:val_len]
    else:
        length = math.floor(ratio * len(train_x))
        val_x = train_x[index * length:(index + 1) * length]
        train_x = torch.cat((train_x[:index * length], train_x[(index + 1) * length:]), dim=0)
        val_y = train_y[index * length:(index + 1) * length]
        train_y = torch.cat((train_y[:index * length], train_y[(index + 1) * length:]), dim=0)
    return train_x, val_x, train_y, val_y


def split_by_window(x, window=3):
    data = []
    for i in range(x.shape[0] - window + 1):
        if len(x.shape) == 4:
            data.append(x[i:i + window, :, :, :].reshape(-1, 6, 9))
        else:
            data.append(x[i + window - 1, :, :].reshape(-1, 159))
    return torch.stack(data, dim=0)


# 准备数据
def prepare_data(train_x, train_y, val_x, val_y, test_x, test_y):
    train_dataset = ClimateDataset(train_x, train_y)
    val_dataset = ClimateDataset(val_x, val_y)
    test_dataset = ClimateDataset(test_x, test_y)

    return train_dataset, val_dataset, test_dataset
