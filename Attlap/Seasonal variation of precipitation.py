import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_by_seasonal(data):
    # 生成日期范围，五年从某一年的1月1日开始
    dates = pd.date_range(start='2011-01-01', periods=3653)

    # 创建掩码，按季节划分
    spring_mask = (dates.month >= 3) & (dates.month <= 5)  # 春：3月到5月
    summer_mask = (dates.month >= 6) & (dates.month <= 8)  # 夏：6月到8月
    autumn_mask = (dates.month >= 9) & (dates.month <= 11)  # 秋：9月到11月
    winter_mask = (dates.month == 12) | (dates.month <= 2)  # 冬：12月到2月

    # 分别提取每个季节的数据
    spring_data = data[spring_mask, :, :]
    summer_data = data[summer_mask, :, :]
    autumn_data = data[autumn_mask, :, :]
    winter_data = data[winter_mask, :, :]

    # 分别计算每个季节的均值
    spring_mean = spring_data.mean()
    summer_mean = summer_data.mean()
    autumn_mean = autumn_data.mean()
    winter_mean = winter_data.mean()
    return winter_mean, spring_mean, summer_mean, autumn_mean


gpm = torch.load('./data/Precipitation True.pt')
reslap = torch.load('./data/Precipitation Prediction(reslap_CBAM_1e-4).pt')
reslap_fpn = torch.load('./data/Precipitation Prediction(ResLap_FPN).pt')
reslap_fpn_aff = torch.load('./data/Precipitation Prediction(ResLap_FPN_AFF).pt')
reslapv3 = torch.load('./data/Precipitation Prediction(ResLapv3_bicubic).pt')

# 假设降水数据是 (1827, lat, lon) 的三维数组
# 这里用随机数生成一个示例的降水数据
GPM = mean_by_seasonal(gpm)
ResLap = mean_by_seasonal(reslap)
ResLap_FPN = mean_by_seasonal(reslap_fpn)
ResLap_FPN_AFF = mean_by_seasonal(reslap_fpn_aff)
ResLapv3 = mean_by_seasonal(reslapv3)

seasons = ['DJF', 'MAM', 'JJA', 'SON']

# 计算模型与GPM的差异
diff1 = np.array(ResLap) - np.array(GPM)
diff2 = np.array(ResLap_FPN) - np.array(GPM)
diff3 = np.array(ResLap_FPN_AFF) - np.array(GPM)
diff4 = np.array(ResLapv3) - np.array(GPM)

# 设置柱状图的宽度和x轴位置
bar_width = 0.15
index = np.arange(len(seasons))

# 创建一个 1x2 的图形布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制左边的柱状图 (各模型的降水量)
ax1.bar(index, GPM, bar_width, label='GPM', color='skyblue')
ax1.bar(index + 1 * bar_width, ResLap, bar_width, label='ResLap', color='lightgrey')
ax1.bar(index + 2 * bar_width, ResLap_FPN, bar_width, label='ResLap_FPN', color='orange')
ax1.bar(index + 3 * bar_width, ResLap_FPN_AFF, bar_width, label='ResLap_FPN_AFF', color='black')
ax1.bar(index + 4 * bar_width, ResLapv3, bar_width, label='ResLapv3', color='yellow')

# 设置左图的标签和标题
ax1.set_xlabel('Season')
ax1.set_ylabel('Precipitation (mm/day)')
ax1.set_title('Seasonal Precipitation by Model')
ax1.set_xticks(index + 2 * bar_width)
ax1.set_xticklabels(seasons)
ax1.legend()
# 设置左图的纵坐标范围为 0 到 4
ax1.set_ylim(0, 4)

# 绘制右边的柱状图 (各模型与GPM的差异)
ax2.bar(index + bar_width, diff1, bar_width, label='ResLap-GPM', color='lightgrey')
ax2.bar(index + 2 * bar_width, diff2, bar_width, label='ResLap_FPN-GPM', color='orange')
ax2.bar(index + 3 * bar_width, diff3, bar_width, label='ResLap_FPN_AFF-GPM', color='black')
ax2.bar(index + 4 * bar_width, diff4, bar_width, label='ResLapv3-GPM', color='yellow')

# 设置右图的标签和标题
ax2.set_xlabel('Season')
ax2.set_ylabel('Difference (mm/day)')
ax2.set_title('Model Differences from GPM')
ax2.set_xticks(index + 1.5 * bar_width)
ax2.set_xticklabels(seasons)
ax2.legend()
# 设置右图的纵坐标范围为 -0.6 到 0.6
ax2.set_ylim(-0.6, 0.6)

# 显示图形
plt.tight_layout()
plt.show()
