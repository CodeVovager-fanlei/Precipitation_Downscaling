import torch

# from CNN import CNNModel
from MAN import MANModel
from utils.computerainfall import compute_rainfall
from utils.util import binary_grid
from utils.data_process import getdata, split_data, standardize_dataset

tensor_x, tensor_y = getdata()

start_time = '2001-01-01'
cut_time = '2010-12-31'
train_x, test_x, train_y, test_y = split_data(start_time, cut_time, tensor_x, tensor_y)
# 标准化处理
train_x_scaled, test_x_scaled = standardize_dataset(train_x, test_x)

input_shape = tensor_x.shape[1:]
output_shape = tensor_y.shape[1:]
cnnModel = MANModel(input_shape, output_shape)
cnnModel.load_state_dict(torch.load('model_parameter/MAN_Model_best.pth'))

cnnModel.eval()
with torch.no_grad():
    ref_pred = cnnModel(train_x_scaled)[0]
    ref_obs = (train_y >= 1).int()
    predictParameter = cnnModel(test_x_scaled)
    Rainfall = compute_rainfall(predictParameter[1], predictParameter[2], True, 1)
    a = Rainfall.numpy()
    # 调用函数
    finalRainFall = binary_grid(Rainfall, predictParameter[0], ref_obs, ref_pred)
    b = finalRainFall.numpy()
    torch.save(finalRainFall, './data/Precipitation_Prediction(MAN).pt')
    torch.save(test_y, './data/Precipitation_True.pt')

