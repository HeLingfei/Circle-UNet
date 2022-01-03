import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt


# 以txt的格式存储网络所有评估结果，追加在对应网络名的文件末尾
# all_metrics 格式示例如下：
# {
#     'training time:,
#     'average inference time: ,
#     'average dice': ,
#     'average sensitivity': ,
#     'average specificity': ,
#     'average precision': ,
#     'average assd':
# }
def save_log(net, all_metrics, losses, name=None, note=None):
    m = dict()
    if name is None:
        name = net.save_name

    for key, value in all_metrics.items():
        m[key] = value
    if note is not None:
        m['note'] = note

    all_path = f'./logs/{name}'
    if not os.path.exists(all_path):
        os.makedirs(all_path)
    dir_path = f'{all_path}/{net.date}'
    os.makedirs(dir_path)
    metrics_path = f'{dir_path}/metrics_data.txt'
    f = open(metrics_path, 'a+', encoding='utf-8')
    f.write(json.dumps(m, indent=4))
    f.close()

    # 存储训练损失相关数据
    loss_path = f'{dir_path}/loss.npy'
    np.save(loss_path, np.array(losses))

    # 存储训练损失折线图
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plot_path = f'{dir_path}/loss.png'
    x = list(range(len(losses)))
    plt.plot(x, losses)
    plt.title(f'{name} loss')
    plt.xlabel('迭代次数')
    plt.ylabel('loss')

    plt.savefig(plot_path)

    print('日志保存完毕')
