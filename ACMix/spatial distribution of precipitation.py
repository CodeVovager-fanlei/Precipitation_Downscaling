import torch
import numpy as np

from utils.plot_tools import prior_obs_field_map


class Field:
    def __init__(self, name, lat, lon, value):
        self.name = name
        self.value = value
        self.lat = lat
        self.lon = np.mod(lon, 360)
        self.nlat = np.size(lat)
        self.nlon = np.size(lon)


"""
    参数
"""
load_path1 = './data/Precipitation Prediction(ACMix32).pt'
load_path2 = './data/Precipitation_True.pt'
model_name = 'ACMix'
save_path = '../ACmix/pic/ACMix_spatial_distribution.png'
########################################################################################################################

origin_data = [None, None]
predict_precipitation = torch.load(load_path1).mean(dim=0)
lat = np.linspace(32.05, 41.95, num=100)
lon = np.linspace(100.1, 115.9, num=159)
origin_data[0] = Field("predict precipitation", lat, lon, value=predict_precipitation)

# Percentage Change=(MODEL-GPM)/GPM × 100%
true_precipitation = torch.load(load_path2).mean(dim=0)
percentage_change = ((predict_precipitation - true_precipitation) / true_precipitation) * 100
lat = np.linspace(32.05, 41.95, num=100)
lon = np.linspace(100.1, 115.9, num=159)
origin_data[1] = Field("percentage change", lat, lon, value=percentage_change)

prior_obs_field_map(origin_data,
                    mode=['Precipitation', "Percentage Change"],
                    model=[model_name],
                    savefig_settings={
                        'path': save_path,
                        'format': 'png',
                        'dpi': 600,
                        'bbox_inches': 'tight',
                        'pad_inches': 0.05
                    })

# true_precipitation = torch.load(load_path2).mean(dim=0)
# lat = np.linspace(32.05, 41.95, num=100)
# lon = np.linspace(100.1, 115.9, num=159)
origin_data = Field("true", lat, lon, value=predict_precipitation)
prior_obs_field_map(origin_data,
                    mode=['Precipitation'],
                    model=['ACMix'],
                    savefig_settings={
                        'path': '../ACMix/pic/ACMix.png',
                        'format': 'png',
                        'dpi': 600,
                        'bbox_inches': 'tight',
                        'pad_inches': 0.05
                    })
