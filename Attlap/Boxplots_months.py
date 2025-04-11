import matplotlib.pyplot as plt
from utils.util import pearson_corrcoef, get_95th_percentile_of_precipitation
import torch
import matplotlib.ticker as mticker
import pandas as pd


def mean_by_month(data):
    # 创建日期范围，假设数据从 '2011-01-01' 开始
    date_range = pd.date_range(start='2011-01-01', periods=3653)
    # 将原始数据转为 pandas DataFrame，每行代表某一天
    # reshape 将 100 和 159 展平成 15900
    df = pd.DataFrame(data.reshape(3653, -1).numpy(), index=date_range)
    # 按年月进行分组，然后对每个月的数据求平均
    monthly_sum_df = df.groupby(df.index.to_period('M')).mean()
    return torch.tensor(monthly_sum_df.values)


GPM = torch.load('./data/Precipitation True.pt')
reslap = torch.load('./data/Precipitation Prediction(reslap_CBAM_1e-4).pt')
reslap_fpn = torch.load('./data/Precipitation Prediction(ResLap_FPN).pt')
reslap_fpn_aff = torch.load('./data/Precipitation Prediction(ResLap_FPN_AFF).pt')
reslapv3 = torch.load('./data/Precipitation Prediction(ResLapv3_bicubic).pt')

GPM_month = mean_by_month(GPM)
reslap_month = mean_by_month(reslap)
reslap_fpn_month = mean_by_month(reslap_fpn)
reslap_fpn_aff_month = mean_by_month(reslap_fpn_aff)
reslapv3_month = mean_by_month(reslapv3)

# 计算均方根误差
rmse1, rmse2, rmse3, rmse4 = [], [], [], []
for i in range(GPM_month.shape[1]):
    rmse = torch.sqrt(((reslap_month[:, i] - GPM_month[:, i]) ** 2).mean())
    rmse1.append(rmse)
    rmse = torch.sqrt(((reslap_fpn_month[:, i] - GPM_month[:, i]) ** 2).mean())
    rmse2.append(rmse)
    rmse = torch.sqrt(((reslap_fpn_aff_month[:, i] - GPM_month[:, i]) ** 2).mean())
    rmse3.append(rmse)
    rmse = torch.sqrt(((reslapv3_month[:, i] - GPM_month[:, i]) ** 2).mean())
    rmse4.append(rmse)

RMSE = [rmse1, rmse2, rmse3, rmse4]

# 计算相关系数
cc1, cc2, cc3, cc4 = [], [], [], []
for i in range(GPM_month.shape[1]):
    cc1.append(pearson_corrcoef(reslap_month[:, i], GPM_month[:, i]))
    cc2.append(pearson_corrcoef(reslap_fpn_month[:, i], GPM_month[:, i]))
    cc3.append(pearson_corrcoef(reslap_fpn_aff_month[:, i], GPM_month[:, i]))
    cc4.append(pearson_corrcoef(reslapv3_month[:, i], GPM_month[:, i]))

CC = [cc1, cc2, cc3, cc4]

# 计算降水95分位数的百分比偏差
GPM_95th_percentile = get_95th_percentile_of_precipitation(GPM)
reslap1_95th_percentile = get_95th_percentile_of_precipitation(reslap)
reslap2_95th_percentile = get_95th_percentile_of_precipitation(reslap_fpn)
reslap3_95th_percentile = get_95th_percentile_of_precipitation(reslap_fpn_aff)
reslap4_95th_percentile = get_95th_percentile_of_precipitation(reslapv3)

reslap1_95th_percentage_bias = ((reslap1_95th_percentile - GPM_95th_percentile) / GPM_95th_percentile).flatten()
reslap2_95th_percentage_bias = ((reslap2_95th_percentile - GPM_95th_percentile) / GPM_95th_percentile).flatten()
reslap3_95th_percentage_bias = (
        (reslap3_95th_percentile - GPM_95th_percentile) / GPM_95th_percentile).flatten()
reslap4_95th_percentage_bias = ((reslap4_95th_percentile - GPM_95th_percentile) / GPM_95th_percentile).flatten()

Percentage_Bias = [reslap1_95th_percentage_bias, reslap2_95th_percentage_bias, reslap3_95th_percentage_bias,
                   reslap4_95th_percentage_bias]

##########################################  绘图     ####################################################################
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))  # 增大图表宽度
model_labels = ['Reslap', 'Reslap_FPN', 'Reslap_FPN_AFF', 'Reslapv3']

# 定义箱线图样式
boxprops = dict(linestyle='-', linewidth=1.5, color='black')  # 箱体边框样式
whiskerprops = dict(linestyle='--', linewidth=1.5, color='gray')  # 须的样式（虚线）
# medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线样式
medianprops = {'visible': False}  # 中位线样式
capprops = dict(linestyle='-', linewidth=1.5, color='gray')  # 顶帽样式
widths = 0.7  # 设置较小的箱体宽度以增加间距

# 设置箱线图的位置，并手动控制位置
positions = [1, 2, 3, 4]  # 手动指定箱线图的位置

# 绘制RMSE箱线图
ax1.boxplot(RMSE, vert=True, showfliers=False, boxprops=boxprops,
            whiskerprops=whiskerprops, medianprops=medianprops,
            capprops=capprops, widths=widths, positions=positions, showmeans=True,
            meanline=True)
ax1.set_title('(a)', loc='left', pad=10)
ax1.set_ylabel('RMSE(mm/day)')
ax1.set_xlabel('Models')
ax1.set_xticks(positions)
ax1.set_xticklabels(model_labels)

# 绘制CC箱线图
ax2.boxplot(CC, vert=True, showfliers=False, boxprops=boxprops,
            whiskerprops=whiskerprops, medianprops=medianprops,
            capprops=capprops, widths=widths, positions=positions, showmeans=True,
            meanline=True)
ax2.set_title('(b)', loc='left', pad=10)
ax2.set_ylabel('CC')
ax2.set_xlabel('Models')
ax2.set_xticks(positions)
ax2.set_xticklabels(model_labels)

# 绘制Percentage Bias箱线图
ax3.boxplot(Percentage_Bias, vert=True, showfliers=False, boxprops=boxprops,
            whiskerprops=whiskerprops, medianprops=medianprops,
            capprops=capprops, widths=widths, positions=positions, showmeans=True,
            meanline=True)
ax3.set_title('(c)', loc='left', pad=10)
ax3.set_ylabel('Bias in the 95$^{th}$ Percentile')
ax3.set_xlabel('Models')
ax3.set_xticks(positions)
ax3.set_xticklabels(model_labels)

# 将纵坐标格式化为百分比
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# 调整子图之间的间距，并增加图像边距，避免箱线图紧挨边框
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

# 保存图像，确保保存的图像与显示一致
plt.savefig('./boxplots_months.png', format='png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
