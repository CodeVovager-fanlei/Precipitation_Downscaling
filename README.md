# Precipitation_Downscaling
<<<<<<< HEAD

基于注意力的黄土高原降水降尺度系统 (V1.0)
该仓库包含了 基于注意力的黄土高原降水降尺度系统V1.0 的代码和模型。该系统使用深度学习模型进行高分辨率区域降水预测，特别为中国黄土高原地区设计。

概述
该系统利用基于注意力的CNN模型进行降水降尺度，包括多种模型（如AttLap、ACMix、MAN和CNN），基于PyTorch框架构建。系统使用ERA5和GPM等气象数据，预测区域降水并提高精度和分辨率。

系统要求
硬件：
处理器：Intel i5 或更高（推荐：Intel i7 或 AMD Ryzen 7）
内存：16GB 或更高（推荐：32GB）
显卡：支持 CUDA 的 NVIDIA GPU（如 GTX 1080、RTX 2080）且显存至少为 8GB（推荐：12GB+）
存储：至少 10GB 可用空间（推荐使用 SSD）
软件：
操作系统：Windows 10/11，Linux（Ubuntu 18.04及以上）
Python版本：3.7 或更高（推荐：Python 3.8 或 3.9）
深度学习框架：PyTorch 1.x（推荐：1.9 及以上）
依赖库：
NumPy, Pandas, SciPy, xarray, netCDF4
Matplotlib, Seaborn, Cartopy, SciencePlots
训练：
# 进入要训练的模型目录中
python train.py
预测：
# 训练完成后，运行该目录下的predict.py
python predict.py
生成的高分辨率降水位于对应目录系下的data文件夹下
评估
python compute_evaluation_metrics.py 
该命令将计算模型的评估指标，如RMSE、CC等。
python spatial distribution of precipitation.py
会生成多年平均降水的空间分布
...
