import torch
from scipy.stats import gamma


def compute_rainfall(log_alpha, log_beta, simulate=False, bias=None):
    # 初始化输出数据列表
    outputs = []
    log_alpha = torch.Tensor(log_alpha)
    log_beta = torch.Tensor(log_beta)

    simulated_rainfall = []
    # 转换数据为矩阵形式
    alpha = torch.exp(log_alpha).reshape(log_alpha.shape[0], -1)
    beta = torch.exp(log_beta).reshape(log_beta.shape[0], -1)
    if simulate:
        # 使用伽马分布生成模拟降水量
        rainfall = torch.empty_like(alpha)
        for i in range(alpha.shape[0]):
            rainfall[i, :] = torch.tensor(gamma.rvs(a=alpha[i, :], scale=beta[i, :]))
    else:
        rainfall = alpha * beta

    # 如果有偏置，则加上偏置
    if bias is not None:
        rainfall += bias
    return rainfall.reshape(log_alpha.shape[0], log_alpha.shape[1], log_alpha.shape[2])
