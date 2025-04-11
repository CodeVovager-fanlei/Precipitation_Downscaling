import torch
from utils.util import compute_evaluate_metrics

true_precipitation = torch.load('./data/Precipitation True.pt')
predict_precipitation = torch.load('./data/Precipitation Prediction(ResLapv3_bicubic).pt')

compute_evaluate_metrics(true_precipitation, predict_precipitation, label='ResLap', name='../ResLap/ResLap_Frequency')