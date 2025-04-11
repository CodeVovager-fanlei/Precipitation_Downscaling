import torch
from utils.util import compute_evaluate_metrics

true_precipitation = torch.load('./data/Precipitation_True.pt')
predict_precipitation = torch.load('./data/Precipitation_Prediction(MAN).pt')

compute_evaluate_metrics(true_precipitation, predict_precipitation, label='MAN', name='../MAN/MAN_Frequency')