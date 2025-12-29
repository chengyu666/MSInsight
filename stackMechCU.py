# 加载NonLinLoc格式走时信息，叠加地震道数据
from datetime import timedelta
import time
import pickle
import os
from matplotlib import pyplot as plt
from data import generate_unique_FM_grid, readsegy, readsac, generate_tt, gen_intensity, gen_fm_grid
import numpy as np
from tqdm import tqdm
# from memory_profiler import profile
from multiprocessing import Pool, shared_memory
from config import read_config_file, read_station_file
from draw import draw_maxbrightness, draw_intensity, flatten_and_histogram
import ctypes


def stack_mech_CUDA(data, sample_rate, traveltime, intensity):
    """
    使用CUDA加速的联合叠加算法
    Args:
        data (ndarray): (n_sta*3, n_samples)
        sample_rate (float): 采样率
        traveltime (ndarray): (n_sta, n_grid)
        intensity (ndarray): (n_fm, n_grid, n_sta)
    Returns:
        ndarray: (n_grid, n_fm, n_samples)
    """
    # 计算参数
    n_fm, n_grid, n_sta = intensity.shape
    n_samples = data.shape[1]
    # 将 traveltime 转换为采样点数
    tt_samples = (traveltime * sample_rate).round().astype(np.int32)
    # print("tt_samples range:", tt_samples.min(), tt_samples.max())
    # 取数据z分量
    data = data[2::3, :]
    # 将数据转换为 C 语言风格的数组
    data = np.ascontiguousarray(data, dtype=np.float32)
    tt_samples = np.ascontiguousarray(tt_samples, dtype=np.int32)
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    # 初始化结果数组
    result = np.zeros((n_grid, n_fm, n_samples), dtype=np.float32)
    # result = np.ascontiguousarray(result, dtype=np.float32)
    if (n_samples-np.max(tt_samples) > 1024):
        print("[Error] n_samples-max(tt_samples)  > 1024")
        return -1
    clib = ctypes.CDLL('./jSSA.dll')
    stackCUDA = getattr(clib, "?stackCUDA@@YAHPEBMPEBH0HHHHPEAM@Z")
    # 参数配置
    stackCUDA.argtypes = [ctypes.POINTER(ctypes.c_float),  # data
                          ctypes.POINTER(ctypes.c_int32),  # tt_samples
                          ctypes.POINTER(ctypes.c_float),  # intensity
                          ctypes.c_int,                    # n_sta
                          ctypes.c_int,                    # n_samples
                          ctypes.c_int,                    # n_grid
                          ctypes.c_int,                    # n_fm
                          ctypes.POINTER(ctypes.c_float)]  # result
    # 返回值配置
    stackCUDA.restype = ctypes.c_int

    # 数据指针
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    tt_samples_ptr = tt_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    intensity_ptr = intensity.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    ret = stackCUDA(data_ptr, tt_samples_ptr, intensity_ptr,
                    n_sta, n_samples, n_grid, n_fm, result_ptr)
    if ret != 0:
        print("CUDA Error! ret=", ret)
        return -1
    # else:
    #     print("CUDA result shape:", result.shape)
    return result


class gridConf(ctypes.Structure):
    _fields_ = [
        ("GridSpacingX", ctypes.c_float),
        ("GridSpacingZ", ctypes.c_float),
        ("SearchSizeX", ctypes.c_int),
        ("SearchSizeY", ctypes.c_int),
        ("SearchSizeZ", ctypes.c_int),
        ("SearchOriginX", ctypes.c_float),
        ("SearchOriginY", ctypes.c_float),
        ("SearchOriginZ", ctypes.c_float)
    ]


def gen_intensity_CUDA(fm_grid, conf, station, a=1.0):
    """
    计算辐射强度
    Args:
        fm_grid (ndarray): 震源机制网格
        conf (dict): 配置信息
        station (dict): 检波器信息
        a (float): 距离衰减因子

    Returns:
        ndarray: 辐射强度,shape=(n_fm, n_grid, n_sta)
    """
    # 计算参数
    n_sta = len(station['x'])
    n_fm = fm_grid.shape[0]
    n_grid = conf['SearchSizeX'] * conf['SearchSizeY'] * conf['SearchSizeZ']
    # 将数据转换为 C 语言风格
    fm_grid_cformat = np.ascontiguousarray(fm_grid, dtype=np.float32)
    conf_cformat = gridConf()
    conf_cformat.GridSpacingX = conf['GridSpacingX']
    conf_cformat.GridSpacingZ = conf['GridSpacingZ']
    conf_cformat.SearchSizeX = conf['SearchSizeX']
    conf_cformat.SearchSizeY = conf['SearchSizeY']
    conf_cformat.SearchSizeZ = conf['SearchSizeZ']
    conf_cformat.SearchOriginX = conf['SearchOriginX']
    conf_cformat.SearchOriginY = conf['SearchOriginY']
    conf_cformat.SearchOriginZ = conf['SearchOriginZ']
    station_cformat = np.zeros((n_sta, 3), dtype=np.float32)
    for i in range(n_sta):
        station_cformat[i, 0] = station['x'][i]/1000
        station_cformat[i, 1] = station['y'][i]/1000
        station_cformat[i, 2] = station['z'][i]/1000
    station_cformat = np.ascontiguousarray(station_cformat, dtype=np.float32)
    # 初始化结果数组
    intensity = np.zeros((n_fm*n_grid*n_sta), dtype=np.float32)
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    # 数据指针
    fm_grid_ptr = fm_grid_cformat.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    station_ptr = station_cformat.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    intensity_ptr = intensity.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))

    clib = ctypes.CDLL('./jSSA.dll')
    intensityCUDA = getattr(
        clib, "?intensityCUDA@@YAHPEBMPEAUGridConfig@@PEAMHHM2@Z")
    # 参数配置
    intensityCUDA.argtypes = [ctypes.POINTER(ctypes.c_float),    # fm_grid
                              ctypes.POINTER(gridConf),          # conf
                              ctypes.POINTER(ctypes.c_float),    # stations
                              ctypes.c_int,                      # n_sta
                              ctypes.c_int,                      # n_fm
                              ctypes.c_float,                    # a
                              ctypes.POINTER(ctypes.c_float)     # intensity
                              ]
    # 返回值配置
    intensityCUDA.restype = ctypes.c_int
    # 调用C函数
    ret = intensityCUDA(fm_grid_ptr, conf_cformat, station_ptr,
                        n_sta, n_fm, a, intensity_ptr)
    if ret != 0:
        print("CUDA Error!")
        return -1

    # 将结果转换为三维数组
    intensity = intensity.reshape(
        n_fm, n_grid, n_sta)
    # print("CUDA intensity shape:", intensity.shape)
    return intensity


def drawintensity(intensity, station, i_fm, i_grid, size=300):
    """
    绘制辐射强度
    Args:
        intensity (ndarray): 辐射强度
        i_fm (int): 震源机制索引
        i_grid (int): 网格索引
        save (bool): 是否保存图片
        size (int): 点大小
    """
    n_sta = intensity.shape[2]
    s = abs(intensity[i_fm, i_grid]) * size
    x = station['x']
    y = station['y']
    c = ['r' if intensity[i_fm, i_grid, i] > 0 else 'b' for i in range(n_sta)]
    print("intensity:", intensity[i_fm, i_grid])
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=s, c=c, alpha=0.7)
    for i in range(n_sta):
        plt.text(x[i], y[i], station['sta_ids'][i], fontsize=8)
    plt.title(f'fm: {i_fm}, grid: {i_grid}')
    plt.axis('equal')
    plt.show()


def preprocess_chunk(index, shared_name, shape, dtype, sample_rate, sta):
    """
    处理单个数据块，用于并行处理
    Args:
        index (int): 数据块的索引
        shared_name (str): 共享内存的名称
        shape (tuple): 数据的形状
        dtype (np.dtype): 数据的类型
        sample_rate (float): 采样率
        sta (int): STALTA 短窗口长度
    """
    # 连接到共享内存
    shm = shared_memory.SharedMemory(name=shared_name)
    data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # 去均值
    # data[index] -= np.mean(data[index])
    # 谱白化
    # data[index] = whightening(data[index], 15, 60, sample_rate)
    # STALTA
    # data[index] = stalta(data[index], sta, sta * 10)
    # 数据归一化
    # data[index] /= np.max(np.abs(data[index]))
    shm.close()  # 关闭共享内存连接


def preprocess_parallel(data_raw, sample_rate, sta=5, num_workers=8):
    """
    并行预处理数据
    Args:
        data_raw (np.ndarray): 原始数据
        sample_rate (float): 采样率
        sta (int): STALTA 短窗口长度
        num_workers (int): 并行进程数
    Returns:
        np.ndarray: 处理后的数据
    """
    start = time.time()
    data = np.copy(data_raw)
    shape = data.shape
    dtype = data.dtype

    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shared_data, data)

    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 将任务分配给每个进程
        pool.starmap(preprocess_chunk, [
            (i, shm.name, shape, dtype, sample_rate, sta) for i in range(len(data))
        ])

    # 从共享内存中提取处理后的数据
    data = np.copy(shared_data)

    # 清理共享内存
    shm.close()
    shm.unlink()
    end = time.time()
    print(f"preprocess time:{end - start:.3f}s")

    return data


def calc_position(conf, max_index, sample_rate):
    x = max_index[0] * conf['GridSpacingX'] + conf['SearchOriginX']
    y = max_index[1] * conf['GridSpacingX'] + conf['SearchOriginY']
    z = max_index[2] * conf['GridSpacingZ'] + conf['SearchOriginZ']
    fm = max_index[3]
    t = max_index[4] / sample_rate
    # round to 3 decimal places
    x = round(x, 3)
    y = round(y, 3)
    z = round(z, 3)
    t = round(t, 3)
    return x, y, z, fm, t


def show_result(result, conf, sample_rate, fm_grid, show=True):
    max_index = np.array(np.unravel_index(np.argmax(result), result.shape))
    max_value = np.max(result)
    # 将maxindex换算为真实坐标
    x, y, z, fm, t = calc_position(conf, max_index, sample_rate)

    if show:
        print("max index:", max_index)
        print("max value:", max_value)
        print(f"max position: ({x:.3f}, {y:.3f}, {z:.3f}, {t:.3f})")
        print(f"max fm: {fm_grid[fm]}")
    return max_index


def checkData(data):
    """检查数据是否正常
    """
    for i in range(len(data)):
        plt.plot(data[i])
        plt.show()

