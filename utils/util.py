import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gamma


def calculate_yearly_precipitation_mean(data, start_year, end_year):
    # 假设每年有365天，处理闰年的情况
    days_per_year = [366 if year % 4 == 0 else 365 for year in range(start_year, end_year + 1)]
    mean_yearly = []

    start_idx = 0
    for days in days_per_year:
        end_idx = start_idx + days
        mean_yearly.append(data[start_idx:end_idx, :, :].mean(dim=0))
        start_idx = end_idx

    return torch.stack(mean_yearly, dim=0)


def pearson_corrcoef(x, y):
    # Ensure the inputs are 1D tensors
    x = x.flatten()
    y = y.flatten()

    # Compute means
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # Compute the covariance between x and y
    covariance = torch.mean((x - mean_x) * (y - mean_y))

    # Compute the standard deviations of x and y
    std_x = torch.std(x)
    std_y = torch.std(y)

    # Compute Pearson correlation coefficient
    pearson_corr = covariance / (std_x * std_y + 1e-6)

    return pearson_corr


"""
参数说明
pred2 由computerainfall()计算得出的预测结果
p 测试集数据经过模型后得到的降水概率
ref_obs 测试集的真实降水量的二值化结果
ref_pred 训练集数据经过模型后得到的降水概率
"""


def binary_grid(pred2, p, ref_obs, ref_pred):
    # 获取预测降水的形状
    n_days, n_lat, n_lon = pred2.shape

    # 初始化二值化的结果数组
    binary_pred = torch.zeros_like(pred2)

    # 将训练集的实际降水数据和降水概率重整为2D矩阵
    ref_obs = ref_obs.reshape(ref_obs.shape[0], -1)
    ref_pred = ref_pred.reshape(ref_pred.shape[0], -1)

    # 计算每个位置的降水概率阈值
    freq = torch.mean((ref_obs == 0).float(), dim=0)

    # 遍历每一个网格点，计算阈值并二值化
    for i in range(n_lat * n_lon):
        # 计算阈值
        threshold = torch.quantile(ref_pred[:, i], freq[i])

        # 对测试集的预测结果进行二值化（注意广播机制）
        binary_pred[:, i // n_lon, i % n_lon] = (p[:, i // n_lon, i % n_lon] >= threshold).int()

    # 计算最终的降水量
    return pred2 * binary_pred


# 运算函数，执行乘除法
def grid_arithmetics(data, value, operator):
    if operator == "/":
        return data / value
    elif operator == "*":
        return data * value
    else:
        raise ValueError("Unsupported operator")


# Bernoulli-Gamma Loss
class BernoulliGammaLoss(nn.Module):
    def __init__(self, last_connection=None, epsilon=1e-6):
        super(BernoulliGammaLoss, self).__init__()
        self.last_connection = last_connection
        self.epsilon = epsilon

    def forward(self, true, pred):
        true = true.cpu()
        pred = pred.cpu()
        if self.last_connection == "dense":
            D = pred.size(1) // 3
            occurrence = pred[:, :D]
            shape_parameter = torch.exp(pred[:, D:(2 * D)])
            scale_parameter = torch.exp(pred[:, (2 * D):])
        elif self.last_connection == "conv":
            occurrence = pred[0, ...]
            shape_parameter = torch.exp(pred[1, ...])
            scale_parameter = torch.exp(pred[2, ...])
        else:
            raise ValueError("Invalid value for last_connection. Use 'dense' or 'conv'.")

        bool_rain = (true > 0).float()

        loss = -torch.mean(
            (1 - bool_rain) * torch.log(1 - occurrence + self.epsilon) +
            bool_rain * (
                    torch.log(occurrence + self.epsilon) +
                    (shape_parameter - 1) * torch.log(true + self.epsilon) -
                    shape_parameter * torch.log(scale_parameter + self.epsilon) -
                    torch.lgamma(shape_parameter + self.epsilon) -
                    true / (scale_parameter + self.epsilon)
            )
        )
        return loss


def custom_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def compute_evaluate_metrics(true_precipitation, predict_precipitation, label, name):
    """
        Evaluation metrics of annual mean precipitation
    """
    # 先计算平均态，即时间维度求平均
    true_mean = true_precipitation.mean(dim=0)
    predict_mean = predict_precipitation.mean(dim=0)
    print("Evaluation metrics of annual mean precipitation")
    # 计算difference
    Difference = ((predict_mean - true_mean)).abs().mean()
    MBE =  ((predict_mean - true_mean)).mean()
    print(f"Difference: {Difference},MBE{MBE}")
    # 计算均方根误差
    RMSE = torch.sqrt(((predict_mean - true_mean) ** 2).mean())
    print(f"RMSE: {RMSE}")
    CC = pearson_corrcoef(predict_mean, true_mean)
    print(f"CC:{CC}")
    true_mean_mean = true_mean.mean()
    r_square = 1 - (((predict_mean - true_mean) ** 2).sum() / ((true_mean_mean - true_mean) ** 2).sum())
    print(f"可解释性方差{r_square}")

    """
        Evaluation metrics of daily precipitation
    """
    true = true_precipitation.mean(dim=1).mean(dim=1)
    pred = predict_precipitation.mean(dim=1).mean(dim=1)
    print("Evaluation metrics of daily precipitation")
    Difference1 = (pred - true).mean()
    print(f"Difference: {Difference1}")
    RMSE1 = torch.sqrt(((pred - true) ** 2).mean())
    print(f"RMSE: {RMSE1}")
    CC = pearson_corrcoef(pred, true)
    print(f"CC:{CC}")

    """
        Frequency
    """
    # 使用 torch.unique 计算各个数值的频次
    precipitation_range = torch.arange(1, 101, 1)

    true_1 = true_precipitation.flatten()
    true_1 = true_1[(true_1 >= 1) & (true_1 <= 100)]
    counts, bin_edges = np.histogram(true_1, bins=precipitation_range, density=True)

    pred_1 = predict_precipitation.flatten()
    pred_1 = pred_1[(pred_1 >= 1) & (pred_1 <= 100)]
    counts_pre, _ = np.histogram(pred_1, bins=precipitation_range, density=True)
    # 计算区间中心
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 绘制直方图
    plt.plot(bin_centers, counts, label="GPM")
    plt.plot(bin_centers, counts_pre, label=label)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.yscale('log')
    plt.legend()
    plt.savefig(name)
    plt.show()


def compute_r95_evaluate_metrics(true_r95, predict_r95):
    """
        Evaluation metrics of annual mean precipitation
    """
    print("Evaluation metrics of annual mean precipitation")
    # 计算difference
    Difference = ((predict_r95 - true_r95)).abs().mean()
    MBE =  ((predict_r95 - true_r95)).mean()
    print(f"Difference: {Difference},{MBE}")
    # 计算均方根误差
    RMSE = torch.sqrt(((predict_r95 - true_r95) ** 2).mean())
    print(f"RMSE: {RMSE}")
    CC = pearson_corrcoef(predict_r95, true_r95)
    print(f"CC:{CC}")


def get_95th_percentile_of_precipitation(data):
    # 复制数据，避免修改原始数据
    data_copy = data.clone()
    # 将小于 1 的降水量设为 0
    data_copy[data_copy < 1] = 0

    # 转换为 numpy 数组，便于高效处理
    data_np = data_copy.numpy()

    t, lat, lon = data_np.shape
    Threshold_Matrix = np.zeros((lat, lon))

    # 计算 95% 阈值矩阵
    for i in range(lat):
        for j in range(lon):
            # 获取该点非零降水数据
            no_contain_zero = data_np[:, i, j][data_np[:, i, j] > 0]
            if len(no_contain_zero) == 0:
                continue

            # 计算该点的 95% 分位数阈值
            Threshold_Matrix[i, j] = np.percentile(no_contain_zero, 95)

    return torch.tensor(Threshold_Matrix)


def get_99th_percentile_of_precipitation(data):
    # 复制数据，避免修改原始数据
    data_copy = data.clone()
    # 将小于 1 的降水量设为 0
    data_copy[data_copy < 1] = 0

    # 转换为 numpy 数组，便于高效处理
    data_np = data_copy.numpy()

    t, lat, lon = data_np.shape
    Threshold_Matrix = np.zeros((lat, lon))

    # 计算 95% 阈值矩阵
    for i in range(lat):
        for j in range(lon):
            # 获取该点非零降水数据
            no_contain_zero = data_np[:, i, j][data_np[:, i, j] > 0]
            if len(no_contain_zero) == 0:
                continue

            # 计算该点的 95% 分位数阈值
            Threshold_Matrix[i, j] = np.percentile(no_contain_zero, 99)

    return torch.tensor(Threshold_Matrix)


def get_percentile_of_precipitation(data, coordinate):
    # 复制数据，避免修改原始数据
    data_copy = data.clone()
    # 将小于 1 的降水量设为 0
    data_copy[data_copy < 1] = 0

    # 转换为 numpy 数组，便于高效处理
    data_np = data_copy.numpy()

    lat, lon = coordinate
    Threshold_Matrix = np.zeros(21)

    # 计算 95% 阈值矩阵

    percentile = np.append(np.array(np.arange(0, 100, 5)), 99)
    # 获取该点非零降水数据
    no_contain_zero = data_np[:, lat, lon][data_np[:, lat, lon] > 0]
    for pi, p in enumerate(percentile):
        # 计算该点的 95% 分位数阈值
        Threshold_Matrix[pi] = np.percentile(no_contain_zero, p)

    return torch.tensor(Threshold_Matrix)


from sklearn.neighbors import NearestNeighbors


def generate_graph(climate_data):
    climate_data = climate_data.detach()
    batch_size, _, _ = climate_data.shape
    edge_index_list = []
    edge_weights_list = []
    label_list = []
    for ii in range(batch_size):
        data = climate_data[ii]
        # 使用KNN建立图
        # knn = NearestNeighbors(n_neighbors=5, metric='euclidean')  # k=5, 使用欧氏距离
        knn = NearestNeighbors(n_neighbors=8, metric='cosine')
        knn.fit(data.cpu().numpy())

        # 获取每个节点的K个最近邻
        distances, indices = knn.kneighbors(data)

        # 创建边连接关系和边的权重（边权重可以使用距离或相似度）
        edge_index = []
        edge_weights = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # 不包括自己
                edge_index.append([i, indices[i][j]])
                # 将余弦距离转换为相似度 (0, 1范围)
                similarity = 1 - distances[i][j] / 2  # 余弦距离 [0, 2] 映射到 [0, 1]
                edge_weights.append(similarity)

        edge_index = torch.tensor(edge_index).T  # 转置为 (2, num_edges)
        edge_weights = torch.tensor(edge_weights)
        edge_index_list.append(edge_index)
        edge_weights_list.append(edge_weights)
        label = torch.full((edge_index.size(1),), ii, dtype=torch.long)
        label_list.append(label)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_weights = torch.cat(edge_weights_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    # # 归一化边权重
    # edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    return edge_index, edge_weights, labels


if __name__ == '__main__':
    # a = torch.randn((128, 54, 32))
    # generate_graph(a)
    pass
