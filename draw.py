from obspy.imaging.beachball import beachball
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import cm  # 用于颜色映射


def flatten_and_histogram(ndarray, bins=10, range=None, density=False):
    """
    将任意维度的 NumPy 数组展平，并统计其直方图。

    参数:
        ndarray (np.ndarray): 输入的任意维度的 NumPy 数组。
        bins (int or sequence of scalars, optional): 直方图的柱数或柱的边界。默认为 10。
        range (tuple, optional): 直方图的数据范围，格式为 (min, max)。默认为 None。
        density (bool, optional): 如果为 True，则直方图的面积总和为 1。默认为 False。

    返回:
        tuple: (hist, bin_edges)，其中 hist 是直方图的柱高，bin_edges 是柱的边界。
    """
    # 展平数组
    flattened_array = ndarray.flatten()

    # 计算直方图
    hist, bin_edges = np.histogram(
        flattened_array, bins=bins, range=range, density=density)

    # 绘制直方图
    plt.hist(flattened_array, bins=bins, range=range, density=density)
    plt.title(f"min: {flattened_array.min()}, max: {flattened_array.max()}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return hist, bin_edges


def draw_voxel(data, pmin, title):
    # 设置阈值，只显示大于阈值的体素
    dmin = data.min()
    dmax = data.max()
    threshold = dmin + pmin * (dmax - dmin)
    indices = np.where(data < threshold)
    data_t = data
    data_t[indices] = False

    # 创建颜色映射，根据数据值生成颜色
    norm = plt.Normalize(data.min(), data.max())  # 归一化数据
    colors = cm.inferno(norm(data))  # 使用 viridis 颜色映射

    # 设置透明度 (alpha)
    alpha_value = 0.3  # 设置透明度为 0.5
    colors[..., 3] = alpha_value  # 将 alpha 通道设置为透明度

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 使用 voxels 绘制三维体素
    ax.voxels(data, facecolors=colors)

    # 设置标题
    ax.set_title(title)

    # 设置坐标轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 添加颜色条（colorbar），表示数据值的颜色映射
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
    mappable.set_array(data)  # 将原始数据传给颜色条
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Data Value')  # 设置颜色条的标签

    # 显示图像
    plt.show()


def draw_maxbrightness(data, title, save=False):
    """画三维矩阵最大值处三个剖面
    """
    # 找到最大亮度值的索引
    max_idx = np.unravel_index(np.argmax(data), data.shape)
    # print("最大值位置：", max_idx, "最大值：", data[max_idx])

    # 提取三个剖面
    # 左视图（YZ平面） - 沿着 X 轴
    left_view = data[max_idx[0], :, :]

    # 顶视图（XY平面） - 沿着 Z 轴
    top_view = data[:, :, max_idx[2]]

    # 正视图（XZ平面） - 沿着 Y 轴
    front_view = data[:, max_idx[1], :]

    # 创建2行2列的子图，留一个空位
    fig, axs = plt.subplots(2, 2, figsize=(4, 4))
    fig.suptitle(title, fontsize=16)

    # 左视图
    ax1 = axs[0, 0]
    im1 = ax1.imshow(left_view, cmap='gnuplot',
                     aspect='equal', origin='lower', vmin=0)
    ax1.set_title('YZ Plane')
    plt.colorbar(im1, ax=ax1)

    # 顶视图
    ax2 = axs[0, 1]
    im2 = ax2.imshow(top_view.T, cmap='gnuplot',
                     aspect='equal', origin='lower', vmin=0)
    ax2.set_title('XY Plane')
    plt.colorbar(im2, ax=ax2)

    # 正视图
    ax3 = axs[1, 1]
    im3 = ax3.imshow(front_view.T, cmap='gnuplot',
                     aspect='equal', origin='lower', vmin=0)
    ax3.set_title('XZ Plane')
    plt.colorbar(im3, ax=ax3)

    # 留空的子图（可以删除这个图，但通常最好保留它）
    axs[1, 0].axis('off')

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 显示图像
    if save:
        plt.savefig(f"maxbrightness_{title}.png")
    else:
        plt.show()
    return max_idx


def draw_2stage_maxbrightness(s1, s2, s2_gr, title):
    """绘制两个阶段的最大亮度值处三视图剖面

    Args:
        s1 (ndarray): 第一阶段亮度场(x,y,z)
        s2 (ndarray): 第二阶段亮度场(x,y,z)
        s2_gr (dict): s2亮度场网格点范围
    """
    max_idx_s2 = np.array(np.unravel_index(np.argmax(s2), s2.shape))
    max_idx_s1 = max_idx_s2 // 2
    max_s2 = s2[max_idx_s2[0], max_idx_s2[1], max_idx_s2[2]]
    # 提取三个剖面
    # 左视图（YZ平面） - 沿着 X 轴
    left_view_s1 = s1[max_idx_s1[0], :, :]
    left_view_s2 = s2[max_idx_s2[0], :, :]
    # 正视图（XZ平面） - 沿着 Y 轴
    front_view_s1 = s1[:, max_idx_s1[1], :]
    front_view_s2 = s2[:, max_idx_s2[1], :]
    # 顶视图（XY平面） - 沿着 Z 轴
    top_view_s1 = s1[:, :, max_idx_s1[2]]
    top_view_s2 = s2[:, :, max_idx_s2[2]]

    # 创建2行2列的子图，留一个空位
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    fig.suptitle(title, fontsize=16)
    # 左视图
    ax1 = axs[0, 0]
    ax1.imshow(left_view_s1, cmap='gnuplot', alpha=0.5,
               aspect='equal', origin='lower', vmin=0, vmax=max_s2)
    ax1.imshow(left_view_s2, cmap='gnuplot', aspect='equal', origin='lower',
               extent=[s2_gr['z_min']/2, s2_gr['z_max']/2, s2_gr['y_min']/2, s2_gr['y_max']/2], vmin=0, vmax=max_s2)
    ax1.set_xlim([-0.5, left_view_s1.shape[1]-0.5])
    ax1.set_ylim([-0.5, left_view_s1.shape[0]-0.5])
    ax1.set_title('Left View (YZ Plane)')
    # plt.colorbar(im1s2, ax=ax1)

    # 顶视图
    ax2 = axs[0, 1]
    ax2.imshow(top_view_s1.T, cmap='gnuplot', alpha=0.5,
               aspect='equal', origin='lower', vmin=0, vmax=max_s2)
    ax2.imshow(top_view_s2.T, cmap='gnuplot', aspect='equal', origin='lower',
               extent=[s2_gr['x_min']/2, s2_gr['x_max']/2, s2_gr['y_min']/2, s2_gr['y_max']/2], vmin=0, vmax=max_s2)
    ax2.set_xlim([-0.5, top_view_s1.shape[1]-0.5])
    ax2.set_ylim([-0.5, top_view_s1.shape[0]-0.5])
    ax2.set_title('Top View (XY Plane)')
    # plt.colorbar(im2s2, ax=ax2)

    # 正视图
    ax3 = axs[1, 1]
    ax3.imshow(front_view_s1.T, cmap='gnuplot', alpha=0.5,
               aspect='equal', origin='lower', vmin=0, vmax=max_s2)
    ax3.imshow(front_view_s2.T, cmap='gnuplot', aspect='equal', origin='lower',
               extent=[s2_gr['x_min']/2, s2_gr['x_max']/2, s2_gr['z_min']/2, s2_gr['z_max']/2], vmin=0, vmax=max_s2)
    ax3.set_xlim([-0.5, front_view_s1.shape[0]-0.5])
    ax3.set_ylim([-0.5, front_view_s1.shape[1]-0.5])
    ax3.set_title('Front View (XZ Plane)')
    # plt.colorbar(im3s2, ax=ax3)

    # 留空的子图（可以删除这个图，但通常最好保留它）
    axs[1, 0].axis('off')

    plt.show()


def draw_waveform_time(data: np.ndarray, sr: int, tt: np.ndarray, event_time, data_starttime, title: str):
    """
    绘制多个 3 分量检波器的 Z 轴波形数据，并用细竖线标注震相到达各个检波器的时间，用粗竖线标注事件时间。

    Args:
        data (np.ndarray): 二维np数组，数据在第一维按 nez 顺序排列。
        sr (int): 采样率。
        tt (np.ndarray): 所有检波器理论走时(sec)，用于偏移发生时间，得到到达时间。
        event_time : 事件发生时间。
        data_starttime : 数据开始时间。
        title (str): 绘图标题。
    """
    num_stations = len(tt)
    z_data_length = data.shape[1]  # 每个分量的数据长度
    time = np.arange(z_data_length) / sr  # 时间轴 (秒)
    time_absolute = np.array(
        [data_starttime + timedelta(seconds=t) for t in time])  # 转换为绝对时间

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.set_ylim(-0.5, num_stations-0.5)

    for i_sta in range(num_stations):
        z_component = data[3 * i_sta + 2] / \
            np.max(np.abs(data[3 * i_sta + 2]))  # 获取 Z 分量

        # 计算震相到达的绝对时间
        arrival_time = event_time + timedelta(seconds=float(tt[i_sta]))

        # 绘制 Z 轴波形并偏移，避免波形重叠
        ax.plot(time_absolute, z_component + i_sta,
                color='black', linewidth=0.8)

        # 标注震相到达时间 (红色细虚线)
        ymin = (i_sta+1) / num_stations
        ymax = (i_sta) / num_stations
        # print(arrival_time, ymin, ymax)
        ax.axvline(arrival_time, ymin, ymax, color='r',
                   linestyle='--')

    # 事件时间的标注 (蓝色粗实线)
    ax.axvline(event_time, color='b', linestyle='-',
               linewidth=1.5)

    # 设置标签、图例和网格
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def draw_intensity(intensity, station, save=False, size=50):
    for i in range(len(intensity)):
        s = abs(intensity[i])*size
        x = station['x'][i]
        y = station['y'][i]
        if intensity[i] > 0:
            plt.plot(x, y, 'ro', markersize=s)
        else:
            plt.plot(x, y, 'bo', markersize=s)
        plt.text(x, y, station['sta_ids'][i],
                 fontsize=8, ha='right', va='bottom')
    plt.axis('equal')  # 确保数据单位比例相同
    plt.grid()
    plt.title('Intensity')
    if save:
        plt.savefig(f"intensity.png")
    else:
        plt.show()


if __name__ == '__main__':
    # test
    data = np.random.rand(10, 10, 10)
    # draw_voxel(data, "Test")
    from data import readsegy, whightening, stalta, readsac, generate_tt
    from stack import load_traveltime
    from config import read_config_file, read_station_file
    # 测试segy
    fname = "20230220_135230.sgy"
    # data, sta_ids, sample_rate, datetime_start = readsegy(
    #     fname)
    # tt_path = "G:\\fasterStack\\3draytracing\\LOC\\time"
    # tt, conf = load_traveltime(tt_path, sta_ids)
    # for i in range(len(data)):
    #     # 去均值
    #     data[i] = data[i] - np.mean(data[i])
    #     # 谱白化
    #     data[i] = whightening(data[i], 15, 60, sample_rate)
    #     # STALTA
    #     # data[i] = stalta(data[i], 5, 125)
    #     # 数据归一化
    #     data[i] = data[i] / np.max(np.abs(data[i]))
    # data = data[:, 3750:4250]
    # datetime_start = datetime_start + timedelta(seconds=3750/sample_rate)
    # draw_waveform_time(data, sample_rate, tt[:, 24, 27, 30], datetime(
    #     2023, 2, 20, 13, 52, 45, 860000), datetime_start, "Test")
    # 测试sac
    data, sta_ids, sample_rate, datetime_start = readsac(
        "F:/SimSeisData/out_chengyu/wav_4")
    data = data[:, 0:750]
    for i in range(len(data)):
        # # 去均值
        # data[i] = data[i] - np.mean(data[i])
        # # 谱白化
        # data[i] = whightening(data[i], 15, 60, sample_rate)
        # STALTA
        # data[i] = stalta(data[i], 5, 125)
        # 数据归一化
        data[i] = data[i] / np.max(np.abs(data[i]))
    folder_conf = "./confsim"
    # 读取conf
    conf = read_config_file(f'{folder_conf}/conf.txt')
    station = read_station_file(f'{folder_conf}/station.txt', conf)
    tt, inc, az = generate_tt(
        f'{folder_conf}/conf.txt', f'{folder_conf}/vel.txt', f'{folder_conf}/station.txt')
    datetime_event = datetime_start + timedelta(seconds=0.11)
    draw_waveform_time(data, sample_rate,
                       tt[:, 24620], datetime_event, datetime_start, "Test")


def draw_beachball(strike, dip, rake, size=200, linewidth=1):
    """
    Plot a beachball diagram using given focal mechanism parameters.

    :param source: Dictionary containing focal mechanism parameters.
    :param size: Size of the beachball diagram.
    :param linewidth: Line width of the beachball diagram.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot the beachball
    beachball([strike, dip, rake], size=size,
              linewidth=linewidth, facecolor='black', bgcolor='white', fig=fig)
