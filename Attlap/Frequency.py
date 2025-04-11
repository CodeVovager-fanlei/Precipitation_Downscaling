import torch
import numpy as np
import matplotlib.pyplot as plt

gpm = torch.load('./data/Precipitation True.pt')
reslap = torch.load('./data/Precipitation Prediction(reslap_CBAM_1e-4).pt')
reslap_fpn = torch.load('./data/Precipitation Prediction(ResLap_FPN).pt')
reslap_fpn_aff = torch.load('./data/Precipitation Prediction(ResLap_FPN_AFF).pt')
reslapv3 = torch.load('./data/Precipitation Prediction(ResLapv3_bicubic).pt')

"""
    Frequency
"""
# 使用 torch.unique 计算各个数值的频次
precipitation_range = torch.arange(1, 101, 1)

true_1 = gpm.flatten()
true_1 = true_1[(true_1 >= 1) & (true_1 <= 100)]
counts, bin_edges = np.histogram(true_1, bins=precipitation_range, density=True)

pred_1 = reslap.flatten()
pred_1 = pred_1[(pred_1 >= 1) & (pred_1 <= 100)]
counts_reslap, _ = np.histogram(pred_1, bins=precipitation_range, density=True)

pred_2 = reslap_fpn.flatten()
pred_2 = pred_2[(pred_2 >= 1) & (pred_2 <= 100)]
counts_reslap_fpn, _ = np.histogram(pred_2, bins=precipitation_range, density=True)
#
pred_3 = reslap_fpn_aff.flatten()
pred_3 = pred_3[(pred_3 >= 1) & (pred_3 <= 100)]
counts_reslap_fpn_aff, _ = np.histogram(pred_3, bins=precipitation_range, density=True)
#
pred_4 = reslapv3.flatten()
pred_4 = pred_4[(pred_4 >= 1) & (pred_4 <= 100)]
counts_reslapv3, _ = np.histogram(pred_4, bins=precipitation_range, density=True)

# 计算区间中心
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# data = [(counts, 'GPM'), (counts_cnn, 'CNN'), (counts_rrdb, 'RRDB'), (counts_rrdb_unsample, 'RRDB_Unsample'),
#         (counts_man, 'MAN')]
data = [(counts, 'GPM'), (counts_reslap, 'reslap'), (counts_reslap_fpn, 'fpn'),(counts_reslap_fpn_aff, 'fpn_aff'),
        (counts_reslapv3, 'reslapv3')]
for (frequency, label) in data:
    # 绘制直方图
    plt.plot(bin_centers, frequency, label=label)

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.yscale('log')
plt.legend()
plt.show()
