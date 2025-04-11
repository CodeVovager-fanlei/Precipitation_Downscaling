import torch
from utils.util import compute_evaluate_metrics

true_precipitation = torch.load('./data/Precipitation_True.pt')
predict_precipitation = torch.load('./data/Precipitation Prediction(ACMix32).pt')

compute_evaluate_metrics(true_precipitation, predict_precipitation, label='ACMix', name='../ACMix/ACMix_Frequency')