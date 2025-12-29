from obspy import read
from datetime import datetime
from matplotlib import pyplot as plt
import segyio
import numpy as np
from obspy.signal.trigger import classic_sta_lta
import os
import ctypes
from config import read_config_file, read_station_file
from multiprocessing import Pool, shared_memory


def datetime_fromtraceheader(header):
    y = header[157]
    d = header[159]
    h = header[161]
    m = header[163]
    s = header[165]
    time_str = "%d,%03d,%02d,%02d,%02d" % (y, d, h, m, s)
    format_str = "%Y,%j,%H,%M,%S"
    return datetime.strptime(time_str, format_str)


def get_name_from_int(num):
    nameint = num & 0xFFFFFFFF
    namebytes = nameint.to_bytes(4, byteorder='little', signed=False)
    return "".join(["%02X" % (i) for i in namebytes])  # [2:]


def readsegy(fname):
    data = []
    sta_ids = []
    with segyio.open(fname, ignore_geometry=True) as f:
        sample_rate = int(1000000/f.header[0][117])
        tnum = f.tracecount
        datetime_start = datetime_fromtraceheader(f.header[0])
        for i in range(tnum):
            # data.append(abs(f.trace[i]))
            data.append(f.trace[i])
        for i in range(int(tnum/3)):
            sta_ids.append(get_name_from_int(f.header[i*3][61]))
    return np.array(data), sta_ids, sample_rate, datetime_start


def readsac(fpath, z_only=False):
    """read all sac file in given folder
    each sac file contains 1 component of 1 station
    filename format: xxx.sta_id.component.sac
    return data, sta_ids, sample_rate, datetime_start

    Args:
        fpath (string): path to folder
    """
    data = []
    sta_ids = []
    sample_rate = None
    datetime_start = None
    for file_name in os.listdir(fpath):
        if not file_name.endswith('.sac'):
            continue
        if len(file_name.split('.')) == 4:
            _, sta, comp, _ = file_name.split('.')
        elif len(file_name.split('.')) == 5:
            _, _, sta, comp, _ = file_name.split('.')
        if z_only and comp != 'Vz':
            continue
        file_path = os.path.join(fpath, file_name)
        tr = read(file_path, format='SAC')[0]
        data.append(tr.data)
        sta_ids.append(sta)
        sample_rate = round(tr.stats.sampling_rate)
        datetime_start = tr.stats.starttime.datetime
    return np.array(data), sta_ids, sample_rate, datetime_start


def readSeveralSegy(path_list):
    """读取文件列表内所有文件的波形
    注意，需要保证文件是连续的！

    Args:
        path_list (list): 文件列表
    """
    data_all = None
    datetime_start = None
    for path in path_list:
        data, sta_ids, sample_rate, datetime_file = readsegy(path)
        if data_all is None:
            data_all = data
            datetime_start = datetime_file
        else:
            data_all = np.concatenate((data_all, data), axis=1)
    return data_all, sta_ids, sample_rate, datetime_start


def whightening(data, f1, f2, f3, f4, sample_rate, epsilon=1e-10):
    """
    对单道地震波形数据进行带通频率范围内的谱白化处理，采用梯形带通（f1~f2~f3~f4）。

    参数:
    data: 一维numpy数组，表示单道地震波形数据
    f1: 截止带下限频率 (Hz)
    f2: 通带下限频率 (Hz)
    f3: 通带上限频率 (Hz)
    f4: 截止带上限频率 (Hz)
    sample_rate: 数据的采样率 (Hz)
    epsilon: 小的常数，用于避免除零错误

    返回:
    whitened_data: 白化后的时间域信号
    """
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)
    magnitude = np.abs(fft_data)
    whitened_fft_data = np.zeros_like(fft_data, dtype=complex)

    # 梯形带通窗函数
    def trapezoid_window(f):
        af = np.abs(f)
        if af < f1 or af > f4:
            return 0.0
        elif f1 <= af < f2:
            return (af - f1) / (f2 - f1)
        elif f2 <= af <= f3:
            return 1.0
        elif f3 < af <= f4:
            return (f4 - af) / (f4 - f3)
        else:
            return 0.0

    for i, f in enumerate(freqs):
        w = trapezoid_window(f)
        if w > 0 and magnitude[i] > epsilon:
            whitened_fft_data[i] = fft_data[i] / magnitude[i] * w

    whitened_data = np.fft.ifft(whitened_fft_data)
    return np.real(whitened_data)


def stalta(data, nsta, nlta):
    """
    计算地震数据的STA/LTA比率.

    参数:
        data: 输入的地震数据 (numpy array).
        nsta: 短时窗的样本数.
        nlta: 长时窗的样本数.

    返回:
        stalta: 计算得到的STA/LTA比率 (numpy array).
    """
    # 确保窗口长度不会超过数据长度
    if len(data) < nlta:
        raise ValueError("数据长度必须大于长时窗长度。")

    result = classic_sta_lta(data, nsta, nlta)
    # result = carl_sta_trig(result, nsta, nlta, ratio=5.0, quiet=1.0)

    return result


def generate_tt(path_conf, path_vel, path_station):
    """调用dll生成走时表

    Args:
        path_conf (str): 配置文件路径
        path_vel (str): 速度模型文件路径
        path_station (str): 台站文件路径

    Returns:
        3*np.ndarray: 走时表：下标顺序为i_sta,i_grid
    """
    # 加载 DLL
    mylib = ctypes.CDLL('./RTtraveltime.dll')
    mylib.cuda_raytrace.argtypes = [
        ctypes.c_uint,                # threads_per_block
        ctypes.c_char_p,              # path_conf
        ctypes.c_char_p,              # path_vel
        ctypes.c_char_p,              # path_station
        ctypes.POINTER(ctypes.c_float),  # output_tt
        ctypes.POINTER(ctypes.c_float),  # output_incident
        ctypes.POINTER(ctypes.c_float)   # output_azimuth
    ]
    mylib.cuda_raytrace.restype = ctypes.c_int

    # 设置调用参数
    threads_per_block = 64
    conf = read_config_file(path_conf)
    stations = read_station_file(path_station, conf)
    nsta = len(stations['sta_ids'])
    nx = conf['SearchSizeX']
    ny = conf['SearchSizeY']
    nz = conf['SearchSizeZ']
    ng = nx * ny * nz
    output_size = nsta * nx * ny * nz
    # print(nsta, nx, ny, nz)
    # print("output_size:", output_size)
    # 初始化输出数组
    output_tt = np.zeros(output_size, dtype=np.float32)
    output_incident = np.zeros(output_size, dtype=np.float32)
    output_azimuth = np.zeros(output_size, dtype=np.float32)

    # 转换 numpy 数组为 ctypes 指针
    output_tt_ptr = output_tt.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_incident_ptr = output_incident.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    output_azimuth_ptr = output_azimuth.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))

    # 调用 DLL 函数
    result = mylib.cuda_raytrace(
        threads_per_block,
        path_conf.encode(),
        path_vel.encode(),
        path_station.encode(),
        output_tt_ptr,
        output_incident_ptr,
        output_azimuth_ptr
    )

    # 检查结果
    if result == 0:
        # reshape data
        output_tt = output_tt.reshape((nsta, ng))
        output_incident = output_incident.reshape((nsta, ng))
        output_azimuth = output_azimuth.reshape((nsta, ng))
    else:
        print("Raytrace calculation failed with error code:", result)

    return output_tt, output_incident, output_azimuth


def fault_parameters_to_moment_tensor(strike, dip, rake):
    """
    将断层参数（走向、倾角、滑动角）转换为矩张量。
    参数以度为单位。
    x轴指向北，y轴指向东，z轴指向下。
    """
    # 将角度转换为弧度
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)

    # 计算矩张量分量
    m_xx = -np.sin(dip) * np.cos(rake) * np.sin(2 * strike) - \
        np.sin(2 * dip) * np.sin(rake) * np.sin(strike)**2
    m_yy = np.sin(dip) * np.cos(rake) * np.sin(2 * strike) - \
        np.sin(2 * dip) * np.sin(rake) * np.cos(strike)**2
    m_xy = np.sin(dip) * np.cos(rake) * np.cos(2 * strike) + 0.5 * \
        np.sin(2 * dip) * np.sin(rake) * np.sin(2 * strike)
    m_xz = -np.cos(dip) * np.cos(rake) * np.cos(strike) - \
        np.cos(2 * dip) * np.sin(rake) * np.sin(strike)
    m_yz = -np.cos(dip) * np.cos(rake) * np.sin(strike) + \
        np.cos(2 * dip) * np.sin(rake) * np.cos(strike)
    m_zz = np.sin(2 * dip) * np.sin(rake)

    # 构建矩张量矩阵
    moment_tensor = np.array([[m_xx, m_xy, m_xz],
                              [m_xy, m_yy, m_yz],
                              [m_xz, m_yz, m_zz]])
    # print(moment_tensor)
    return moment_tensor


def calculate_radiation_intensity(moment_tensor, vector):
    """
    计算给定向量在指定矩张量下的辐射强度。
    注意vector的方向应该是从震源到接收点，三分量：北、东、下。
    """
    # 将向量归一化
    vector = vector / np.linalg.norm(vector)

    # 计算辐射强度
    intensity = np.dot(vector, np.dot(moment_tensor, vector))

    return intensity


def gen_intensity(fm_grid, conf, station, a=0.5):
    """生成所有网格点辐射强度

    Args:
        fm_grid : 震源机制网格
        conf : 配置信息
        station : 台站信息
        a : 距离衰减参数，越大衰减越多

    Returns:
        ndarray: 辐射强度，形状为 (n_fm, n_grid, n_sta)
    """
    # 生成每个震源机制、每个网格点到所有站的辐射强度
    n_fm = fm_grid.shape[0]
    n_grid = conf['SearchSizeX'] * conf['SearchSizeY'] * conf['SearchSizeZ']
    n_sta = len(station['x'])
    intensity = np.zeros((n_fm, n_grid, n_sta))
    print('computing intensity...', n_fm*n_grid*n_sta)

    for i_fm in range(n_fm):
        print(f"fm:{i_fm}/{n_fm}", fm_grid[i_fm])
        mt = fault_parameters_to_moment_tensor(
            fm_grid[i_fm, 0], fm_grid[i_fm, 1], fm_grid[i_fm, 2])
        for i_grid in range(n_grid):
            # print(f"grid:{i_grid}/{n_grid}")
            i_z = i_grid % conf['SearchSizeZ']
            i_y = (i_grid // conf['SearchSizeZ']) % conf['SearchSizeY']
            i_x = i_grid // (conf['SearchSizeZ'] * conf['SearchSizeY'])
            p_grid = np.array([conf['SearchOriginX'] + i_x * conf['GridSpacingX'],
                               conf['SearchOriginY'] +
                               i_y * conf['GridSpacingX'],
                               conf['SearchOriginZ'] + i_z * conf['GridSpacingZ']])
            # 统一单位：米
            p_grid = p_grid * 1000
            # print(f"p_grid:{p_grid}")
            for i_sta in range(n_sta):
                p_sta = np.array(
                    [station['x'][i_sta], station['y'][i_sta], station['z'][i_sta]])
                # print("p_sta", p_sta)
                # 计算向量, 从震源到接收点, 三分量：北(y)、东(x)、下
                vector = [p_sta[1]-p_grid[1], p_sta[0] -
                          p_grid[0], p_sta[2]-p_grid[2]]
                # 单位转换：从m到km
                vlen = np.linalg.norm(vector)/1000
                # print(vector)
                intensity[i_fm, i_grid, i_sta] = calculate_radiation_intensity(
                    mt, vector)*np.exp(-a*vlen)
                # print(intensity[i_fm, i_grid, i_sta])
                # 绘图
            #     plt.scatter(station['x'][i_sta], station['y'][i_sta],
            #                 c='r' if intensity[i_fm,
            #                                    i_grid, i_sta] > 0 else 'b',
            #                 s=abs(intensity[i_fm, i_grid, i_sta])*200)
            # plt.axis('equal')
            # plt.show()
        # print(f"fm:{i_fm}/{n_fm}")
    return intensity


def gen_intensity_task(i_fm, shm_name, fm_grid, conf, station):
    n_grid = conf['SearchSizeX'] * conf['SearchSizeY'] * conf['SearchSizeZ']
    n_sta = len(station['x'])
    n_fm = fm_grid.shape[0]
    mt = fault_parameters_to_moment_tensor(
        fm_grid[i_fm, 0], fm_grid[i_fm, 1], fm_grid[i_fm, 2])
    # 连接到共享内存
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_intensity = np.ndarray(shape=(n_fm, n_grid, n_sta), dtype=np.float32,
                                  buffer=shm.buf)
    for i_grid in range(n_grid):
        i_z = i_grid % conf['SearchSizeZ']
        i_y = (i_grid // conf['SearchSizeZ']) % conf['SearchSizeY']
        i_x = i_grid // (conf['SearchSizeZ'] * conf['SearchSizeY'])
        p_grid = np.array([conf['SearchOriginX'] + i_x * conf['GridSpacingX'],
                           conf['SearchOriginY'] +
                           i_y * conf['GridSpacingX'],
                           conf['SearchOriginZ'] + i_z * conf['GridSpacingZ']])
        # 统一单位：米
        p_grid = p_grid * 1000
        for i_sta in range(n_sta):
            p_sta = np.array(
                [station['x'][i_sta], station['y'][i_sta], station['z'][i_sta]])
            vector = p_sta-p_grid
            shared_intensity[i_fm, i_grid, i_sta] = calculate_radiation_intensity(
                mt, vector)
    # print(f"fm:{i_fm}/{n_fm}")
    shm.close()


def gen_intensity_parallel(fm_grid, conf, station, num_workers=10):
    # 生成每个震源机制、每个网格点到所有站的辐射强度
    n_fm = fm_grid.shape[0]
    n_grid = conf['SearchSizeX'] * conf['SearchSizeY'] * conf['SearchSizeZ']
    n_sta = len(station['x'])
    print("n_fm/n_grid/n_sta:", n_fm, n_grid, n_sta)
    # 创建共享内存
    shm = shared_memory.SharedMemory(
        create=True, size=(n_fm*n_grid*n_sta)*4)  # float32
    shared_intensity = np.ndarray(shape=(n_fm, n_grid, n_sta),
                                  dtype=np.float32, buffer=shm.buf)
    print('computing intensity...', n_fm*n_grid*n_sta)
    # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 每个任务计算一个fm，将任务分配给每个进程
        pool.starmap(gen_intensity_task, [
                     (i_fm, shm.name, fm_grid, conf, station) for i_fm in range(n_fm)])
    intensity = np.copy(shared_intensity)
    # 清理共享内存
    shm.close()
    shm.unlink()

    return intensity


def gen_fm_grid(conf):
    """generate fault mechanism grid

    Args:
        conf (dict): configuration

    Returns:
        np.ndarray: fm grid
    """
    n_strike = conf['SearchSizeStrike']
    n_dip = conf['SearchSizeDip']
    n_rake = conf['SearchSizeRake']
    strike = np.linspace(0, 360, n_strike, endpoint=False)
    dip = np.linspace(0, 90, n_dip)
    step = 360 / n_rake
    rake = np.arange(-180 + step, 180 + step, step)
    # print("strike:", strike)
    # print("dip:", dip)
    # print("rake:", rake)

    fm_grid = np.zeros((n_strike*n_dip*n_rake, 3))
    for i_strike in range(n_strike):
        for i_dip in range(n_dip):
            for i_rake in range(n_rake):
                i = i_strike * n_dip * n_rake + i_dip * n_rake + i_rake
                fm_grid[i, 0] = strike[i_strike]
                fm_grid[i, 1] = dip[i_dip]
                fm_grid[i, 2] = rake[i_rake]
    return fm_grid


def generate_unique_FM_grid(conf):
    strike_step = conf['FMSearchStep']
    dip_step = conf['FMSearchStep']
    rake_step = conf['FMSearchStep']
    unique_tensors = dict()
    cnt = 0
    for strike in range(0, 360, strike_step):
        for dip in range(0, 91, dip_step):
            for rake in range(-180, 181, rake_step):
                m = fault_parameters_to_moment_tensor(strike, dip, rake)
                vec = [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]
                vec = np.array(vec)
                vec /= np.linalg.norm(vec)  # 单位化，确保一致性
                mt_key = tuple(np.round(vec, decimals=4))
                if mt_key not in unique_tensors:
                    unique_tensors[mt_key] = (strike, dip, rake)  # 记录代表参数
                else:
                    cnt += 1
    print("exclude same FM:", cnt)
    print("unique FM:", len(unique_tensors))
    return np.array(list(unique_tensors.values()))
