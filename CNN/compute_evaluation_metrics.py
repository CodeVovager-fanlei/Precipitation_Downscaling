import torch
from utils.util import compute_evaluate_metrics

true_precipitation = torch.load('./data/Precipitation_True.pt')
predict_precipitation = torch.load('./data/Precipitation_Prediction.pt')

compute_evaluate_metrics(true_precipitation, predict_precipitation, label='CNN', name='../CNN/CNN_Frequency')