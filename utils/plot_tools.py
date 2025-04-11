#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from utils.visual import savefig, showfig, setlabel
import geopandas as gpd


def prior_obs_field_map(data, region=[100, 116, 32, 42], figsize=[6, 1.7],
                        wspace=0.3, hspace=0, left=0, right=1, bottom=0, top=1,
                        coord_font_size=7.7, coord_font_weight='normal',
                        extendfrac='auto', colorbar_size=7.5, aspect=35, shrink=1, color_bar_pad=0.15,
                        savefig_settings=None, mode=None,
                        coord_pad=0.8, title_pad=3, sub_title_size=8, title_loc='center',
                        edge_line_width=0.49, inline_width=0.45,
                        level=np.linspace(0, 480, 10), ticks=np.linspace(0, 480, 9), model=None):
    if type(data) is not list:
        # 从数据中获取纬度和经度信息
        lat = data.lat
        lon = data.lon
        field_X, field_Y = np.meshgrid(lon, lat)  # 创建经纬度网格
        # 计算观测场的均值
        # precipitation = np.nanmean(data.value, axis=0)
        precipitation = data.value
    else:
        lat = data[0].lat
        lon = data[0].lon
        field_X, field_Y = np.meshgrid(lon, lat)  # 创建经纬度网格
        precipitation = data[0].value
        lat1 = data[1].lat
        lon1 = data[1].lon
        field_X1, field_Y1 = np.meshgrid(lon1, lat1)  # 创建经纬度网格
        percentage_change = data[1].value

    plt.ioff()

    # 创建图形对象，并设置图形大小和分辨率
    fig = plt.figure(figsize=figsize, dpi=600)

    # 调整子图之间的间距和位置
    fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    # 创建一个1x3的网格布局
    gs = plt.GridSpec(1, 2)

    # 创建地图投影对象
    proj = ccrs.PlateCarree()

    # 如果没有提供保存设置，则初始化为空字典
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

    # 设置字体和线宽
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.linewidth'] = edge_line_width
    mpl.rcParams['mathtext.fontset'] = 'stix'  # 设置数学文本的字体

    # 读取边界线数据
    loess_border_file = gpd.read_file('../river_data/river.shp')
    target_crs = {'init': 'epsg:4326'}
    loess_border_file.to_crs(target_crs, inplace=True)  # 转换坐标系

    # 创建第一个子图，并设置投影
    ax1 = plt.subplot(gs[0, 0], projection=proj)
    ax1.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax1.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    cmap1 = mcolors.LinearSegmentedColormap.from_list('brown_yellow_green', ['#654321', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=4)
    con1 = ax1.contourf(field_X, field_Y, precipitation, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax1.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax1.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax1.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax1.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    if model is not None:
        ax1.set_ylabel(model[0], fontsize=coord_font_size, fontweight=coord_font_weight)

    # 添加颜色条
    cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
                       extendfrac=extendfrac, aspect=aspect)
    cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax1.grid(False)

    # 设置标题
    ax1.set_title(mode[0], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    if type(data) is list:
        # 创建第一个子图，并设置投影
        ax1 = plt.subplot(gs[0, 1], projection=proj)
        ax1.set_extent(region, crs=proj)  # 设置子图范围

        # 添加边界线
        ax1.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                           linewidth=inline_width, zorder=4)

        # 绘制填充等值线图
        # 设置颜色映射和归一化
        cmap2 = plt.get_cmap('RdBu')  # 红色到蓝色的配色
        norm2 = mcolors.TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
        level = np.linspace(-60, 60, 10)
        con1 = ax1.contourf(field_X1, field_Y1, percentage_change, cmap=cmap2, extend='both', norm=norm2,
                            levels=level)

        # 设置坐标轴刻度
        ax1.set_xticks(np.arange(region[0], region[1] + 1, 4))
        ax1.set_yticks(np.arange(region[2], region[3] + 1, 2))
        ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
        ax1.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

        # 设置刻度参数
        ax1.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                        labelsize=coord_font_size, pad=coord_pad)
        ax1.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                        labelsize=coord_font_size, pad=coord_pad)
        ax1.tick_params(axis='both', which='both', top=False, right=False)

        # 添加颜色条
        ticks = np.linspace(-60, 60, 9)
        cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
                           extendfrac=extendfrac, aspect=aspect)
        cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

        # 关闭网格
        ax1.grid(False)

        # 设置标题
        ax1.set_title(mode[1], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 根据保存设置保存图形或显示图形
    if 'path' in savefig_settings:
        savefig(fig, settings=savefig_settings)
    else:
        showfig(fig)


def plot_rmse(data, region=[100, 116, 32, 42], figsize=[8, 1.7],
              wspace=0.3, hspace=0, left=0, right=1, bottom=0, top=1,
              coord_font_size=7.7, coord_font_weight='normal',
              extendfrac='auto', colorbar_size=7.5, aspect=35, shrink=1, color_bar_pad=0.15,
              savefig_settings=None, mode=None,
              coord_pad=0.8, title_pad=3, sub_title_size=8, title_loc='center',
              edge_line_width=0.49, inline_width=0.45,
              level=np.linspace(0, 13, 15), ticks=np.linspace(0, 13, 14), model=None):
    lat = data[0].lat
    lon = data[0].lon
    field_X, field_Y = np.meshgrid(lon, lat)  # 创建经纬度网格

    rmse1 = data[0].value
    rmse2 = data[1].value
    rmse3 = data[2].value

    # 关闭交互模式
    plt.ioff()

    # 创建图形对象，并设置图形大小和分辨率
    fig = plt.figure(figsize=figsize, dpi=600)

    # 调整子图之间的间距和位置
    fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    # 创建一个1x3的网格布局
    gs = plt.GridSpec(1, 4)

    # 创建地图投影对象
    proj = ccrs.PlateCarree()

    # 如果没有提供保存设置，则初始化为空字典
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

    # 设置字体和线宽
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.linewidth'] = edge_line_width
    mpl.rcParams['mathtext.fontset'] = 'stix'  # 设置数学文本的字体

    # 读取边界线数据
    loess_border_file = gpd.read_file('./river_data/river.shp')
    target_crs = {'init': 'epsg:4326'}
    loess_border_file.to_crs(target_crs, inplace=True)  # 转换坐标系

    # 创建第一个子图，并设置投影
    ax1 = plt.subplot(gs[0, 0], projection=proj)
    ax1.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax1.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('light_yellow_orange_red', ['#F2E5A1', '#D96C4A', '#9B2D20'])

    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=13)
    con1 = ax1.contourf(field_X, field_Y, rmse1, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax1.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax1.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax1.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax1.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    if model is not None:
        ax1.set_ylabel(model[0], fontsize=coord_font_size, fontweight=coord_font_weight)

    # 关闭网格
    ax1.grid(False)

    # 设置标题
    ax1.set_title(mode[0], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 创建第二个子图，并设置投影
    ax2 = plt.subplot(gs[0, 1], projection=proj)
    ax2.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax2.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('light_yellow_orange_red', ['#F2E5A1', '#D96C4A', '#9B2D20'])
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=13)
    con1 = ax2.contourf(field_X, field_Y, rmse2, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax2.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax2.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax2.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax2.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax2.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax2.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax2.tick_params(axis='both', which='both', top=False, right=False)

    # # 添加颜色条
    # cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
    #                    extendfrac=extendfrac, aspect=aspect)
    # cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax2.grid(False)

    # 设置标题
    ax2.set_title(mode[1], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 创建第三个子图，并设置投影
    ax3 = plt.subplot(gs[0, 2], projection=proj)
    ax3.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax3.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('light_yellow_orange_red', ['#F2E5A1', '#D96C4A', '#9B2D20'])
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=13)
    con1 = ax3.contourf(field_X, field_Y, rmse3, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax3.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax3.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax3.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax3.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax3.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax3.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax3.tick_params(axis='both', which='both', top=False, right=False)

    # 添加颜色条
    cbar_ax = fig.add_axes([0.78, 0.1, 0.005, 0.75])  # 添加颜色条轴（右侧）
    cb1 = fig.colorbar(con1, cax=cbar_ax, ticks=ticks, orientation='vertical', extendfrac=extendfrac, aspect=aspect)
    cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax3.grid(False)

    # 设置标题
    ax3.set_title(mode[2], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 根据保存设置保存图形或显示图形
    if 'path' in savefig_settings:
        savefig(fig, settings=savefig_settings)
    else:
        showfig(fig)


def plot_r2(data, region=[100, 116, 32, 42], figsize=[8, 1.7],
            wspace=0.3, hspace=0, left=0, right=1, bottom=0, top=1,
            coord_font_size=7.7, coord_font_weight='normal',
            extendfrac='auto', colorbar_size=7.5, aspect=35, shrink=1, color_bar_pad=0.15,
            savefig_settings=None, mode=None,
            coord_pad=0.8, title_pad=3, sub_title_size=8, title_loc='center',
            edge_line_width=0.49, inline_width=0.45,
            level=np.linspace(0, 0.6, 8), ticks=np.linspace(0, 0.6, 7), model=None):
    lat = data[0].lat
    lon = data[0].lon
    field_X, field_Y = np.meshgrid(lon, lat)  # 创建经纬度网格

    rmse1 = data[0].value
    rmse2 = data[1].value
    rmse3 = data[2].value

    # 关闭交互模式
    plt.ioff()

    # 创建图形对象，并设置图形大小和分辨率
    fig = plt.figure(figsize=figsize, dpi=600)

    # 调整子图之间的间距和位置
    fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    # 创建一个1x3的网格布局
    gs = plt.GridSpec(1, 4)

    # 创建地图投影对象
    proj = ccrs.PlateCarree()

    # 如果没有提供保存设置，则初始化为空字典
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

    # 设置字体和线宽
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.linewidth'] = edge_line_width
    mpl.rcParams['mathtext.fontset'] = 'stix'  # 设置数学文本的字体

    # 读取边界线数据
    loess_border_file = gpd.read_file('./river_data/river.shp')
    target_crs = {'init': 'epsg:4326'}
    loess_border_file.to_crs(target_crs, inplace=True)  # 转换坐标系

    # 创建第一个子图，并设置投影
    ax1 = plt.subplot(gs[0, 0], projection=proj)
    ax1.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax1.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('green_gold_red', ['#A8D08D', '#E0A800', '#9C3D3D'])

    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=0.6)
    con1 = ax1.contourf(field_X, field_Y, rmse1, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax1.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax1.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax1.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax1.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    if model is not None:
        ax1.set_ylabel(model[0], fontsize=coord_font_size, fontweight=coord_font_weight)

    # 添加颜色条
    # cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
    #                    extendfrac=extendfrac, aspect=aspect)
    # cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax1.grid(False)

    # 设置标题
    ax1.set_title(mode[0], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 创建第二个子图，并设置投影
    ax2 = plt.subplot(gs[0, 1], projection=proj)
    ax2.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax2.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('green_gold_red', ['#A8D08D', '#E0A800', '#9C3D3D'])
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=0.6)
    con1 = ax2.contourf(field_X, field_Y, rmse2, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax2.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax2.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax2.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax2.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax2.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax2.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax2.tick_params(axis='both', which='both', top=False, right=False)

    # # 添加颜色条
    # cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
    #                    extendfrac=extendfrac, aspect=aspect)
    # cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax2.grid(False)

    # 设置标题
    ax2.set_title(mode[1], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 创建第三个子图，并设置投影
    ax3 = plt.subplot(gs[0, 2], projection=proj)
    ax3.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax3.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('green_gold_red', ['#A8D08D', '#E0A800', '#9C3D3D'])
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=0, vmax=0.6)
    con1 = ax3.contourf(field_X, field_Y, rmse3, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax3.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax3.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax3.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax3.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax3.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax3.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax3.tick_params(axis='both', which='both', top=False, right=False)

    # 添加颜色条
    # 添加颜色条
    cbar_ax = fig.add_axes([0.78, 0.1, 0.005, 0.75])  # 添加颜色条轴（右侧）
    cb1 = fig.colorbar(con1, cax=cbar_ax, ticks=ticks, orientation='vertical', extendfrac=extendfrac, aspect=aspect)
    cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax3.grid(False)

    # 设置标题
    ax3.set_title(mode[1], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 根据保存设置保存图形或显示图形
    if 'path' in savefig_settings:
        savefig(fig, settings=savefig_settings)
    else:
        showfig(fig)


def plot_percentile_bias(data, region=[100, 116, 32, 42], figsize=[8, 1.7],
                         wspace=0.3, hspace=0, left=0, right=1, bottom=0, top=1,
                         coord_font_size=7.7, coord_font_weight='normal',
                         extendfrac='auto', colorbar_size=7.5, aspect=35, shrink=1, color_bar_pad=0.15,
                         savefig_settings=None, mode=None,
                         coord_pad=0.8, title_pad=3, sub_title_size=8, title_loc='center',
                         edge_line_width=0.49, inline_width=0.45,
                         level=np.linspace(-0.7, 0.8, 8), ticks=np.linspace(-0.7, 0.8, 7), model=None):
    lat = data[0].lat
    lon = data[0].lon
    field_X, field_Y = np.meshgrid(lon, lat)  # 创建经纬度网格

    rmse1 = data[0].value
    rmse2 = data[1].value
    rmse3 = data[2].value

    # 关闭交互模式
    plt.ioff()

    # 创建图形对象，并设置图形大小和分辨率
    fig = plt.figure(figsize=figsize, dpi=600)

    # 调整子图之间的间距和位置
    fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    # 创建一个1x3的网格布局
    gs = plt.GridSpec(1, 4)

    # 创建地图投影对象
    proj = ccrs.PlateCarree()

    # 如果没有提供保存设置，则初始化为空字典
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

    # 设置字体和线宽
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.linewidth'] = edge_line_width
    mpl.rcParams['mathtext.fontset'] = 'stix'  # 设置数学文本的字体

    # 读取边界线数据
    loess_border_file = gpd.read_file('./river_data/river.shp')
    target_crs = {'init': 'epsg:4326'}
    loess_border_file.to_crs(target_crs, inplace=True)  # 转换坐标系

    # 创建第一个子图，并设置投影
    ax1 = plt.subplot(gs[0, 0], projection=proj)
    ax1.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax1.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('brown_white_blue', ['#8C564B', '#FFFFFF', '#1F77B4'])

    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=-0.7, vmax=0.8)
    con1 = ax1.contourf(field_X, field_Y, rmse1, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax1.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax1.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax1.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax1.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    if model is not None:
        ax1.set_ylabel(model[0], fontsize=coord_font_size, fontweight=coord_font_weight)

    # # 添加颜色条
    # cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
    #                    extendfrac=extendfrac, aspect=aspect)
    # cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax1.grid(False)

    # 设置标题
    ax1.set_title(mode[0], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 创建第二个子图，并设置投影
    ax2 = plt.subplot(gs[0, 1], projection=proj)
    ax2.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax2.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('brown_white_blue', ['#8C564B', '#FFFFFF', '#1F77B4'])
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=-0.7, vmax=0.8)
    con1 = ax2.contourf(field_X, field_Y, rmse2, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax2.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax2.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax2.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax2.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax2.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax2.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax2.tick_params(axis='both', which='both', top=False, right=False)

    # # 添加颜色条
    # cb1 = fig.colorbar(con1, ticks=ticks, orientation='horizontal', shrink=shrink, pad=color_bar_pad,
    #                    extendfrac=extendfrac, aspect=aspect)
    # cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax2.grid(False)

    # 设置标题
    ax2.set_title(mode[1], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 创建第三个子图，并设置投影
    ax3 = plt.subplot(gs[0, 2], projection=proj)
    ax3.set_extent(region, crs=proj)  # 设置子图范围

    # 添加边界线
    ax3.add_geometries(loess_border_file['geometry'], crs=proj, edgecolor='white', facecolor='none',
                       linewidth=inline_width, zorder=4)

    # 绘制填充等值线图
    # 定义自定义颜色映射（从棕色到浅黄再到绿色）
    cmap1 = mcolors.LinearSegmentedColormap.from_list('brown_white_blue', ['#8C564B', '#FFFFFF', '#1F77B4'])
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('custom_brown_yellow_green', ['#8B4513', '#F0E68C', '#006400'])
    # 设置归一化
    norm1 = mcolors.Normalize(vmin=-0.7, vmax=0.8)
    con1 = ax3.contourf(field_X, field_Y, rmse3, cmap=cmap1, extend='both', norm=norm1,
                        levels=level)

    # 设置坐标轴刻度
    ax3.set_xticks(np.arange(region[0], region[1] + 1, 4))
    ax3.set_yticks(np.arange(region[2], region[3] + 1, 2))
    ax3.xaxis.set_major_formatter(LongitudeFormatter())  # 格式化经度刻度
    ax3.yaxis.set_major_formatter(LatitudeFormatter())  # 格式化纬度刻度

    # 设置刻度参数
    ax3.tick_params(axis='x', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax3.tick_params(axis='y', top=True, which='major', direction='out', length=5, width=edge_line_width,
                    labelsize=coord_font_size, pad=coord_pad)
    ax3.tick_params(axis='both', which='both', top=False, right=False)

    # 添加颜色条
    cbar_ax = fig.add_axes([0.78, 0.1, 0.005, 0.75])  # 添加颜色条轴（右侧）
    cb1 = fig.colorbar(con1, cax=cbar_ax, ticks=ticks, orientation='vertical', extendfrac=extendfrac, aspect=aspect)
    cb1.ax.tick_params(direction='in', labelsize=colorbar_size, width=edge_line_width, length=0)

    # 关闭网格
    ax3.grid(False)

    # 设置标题
    ax3.set_title(mode[1], fontsize=sub_title_size, fontweight=coord_font_weight, pad=title_pad, loc=title_loc)

    # 根据保存设置保存图形或显示图形
    if 'path' in savefig_settings:
        savefig(fig, settings=savefig_settings)
    else:
        showfig(fig)
