# 加载配置文件夹内文件的代码
def read_config_file(filepath):
    config = {}
    with open(filepath, 'r') as file:
        for line in file:
            # 移除多余的空白字符
            line = line.strip()
            # 跳过空行或注释行
            if not line or not line[0].isalpha():
                continue
            # 拆分键和值
            key, value = line.split('=', 1)
            # 去掉键和值的多余空白并转换数值
            key = key.strip()
            value = value.strip()
            try:
                # 尝试将值转换为浮点数或整数
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # 如果不能转换为数字，保留为字符串
            config[key] = value
    return config

def read_station_file(filepath, conf):
    obs_sys={}
    obs_sys['sta_ids']=[]
    obs_sys['x']=[]
    obs_sys['y']=[]
    obs_sys['z']=[]
    xRef, yRef = conf['RefCoord'].split()[:2]
    xRef = float(xRef) * 1000
    yRef = float(yRef) * 1000
    with open(filepath, 'r') as file:
        for line in file:
            # 移除多余的空白字符
            line = line.strip()
            # 跳过空行或注释行
            if not line or not line[0].isalpha():
                continue
            #拆分站号、坐标
            sta_id, x, y, z = line.split()
            # 加入list
            obs_sys['sta_ids'].append(sta_id)
            obs_sys['x'].append(round(float(x)-xRef, 3))
            obs_sys['y'].append(round(float(y)-yRef, 3))
            obs_sys['z'].append(round(float(z), 3))
    return obs_sys



if __name__ == '__main__':
    # 示例用法
    conf_s1 = read_config_file('./config/conf_s1.txt')
    print(conf_s1)
    obs_sys = read_station_file('./config/station.txt', conf_s1)
    print(obs_sys)
