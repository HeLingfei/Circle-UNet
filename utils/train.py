import json
import math
import time

from glob2 import glob
from torch import optim, nn
from torch.utils.data import DataLoader
import torch

from models.CircleUNet import CircleUNet
from models.UNet import UNet
from models.SegNet import SegNet
from models.utils.inference import show_inference_examples
from models.utils.log import save_log
from models.utils.performance import get_all_metrics
from utils.CTDataset import CTDataset
import matplotlib.pyplot as plt
import os
import numpy as np


# 取最后num次迭代的平均loss来对比，保留最优模型
def save_best_model(net, losses, num=50):
    name = net.save_name
    path = f'./pretrained_models/{name}.pkl'
    average_loss = np.mean(np.array(losses[-num:]))
    date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    net.average_loss = average_loss
    net.date = date
    if os.path.exists(path):
        old_net = torch.load(path)
        old_average_loss = getattr(old_net, "average_loss", "None")
        if hasattr(old_net, 'average_loss') and old_net.average_loss <= average_loss:
            print(f'未更新模型,last average_loss:{old_average_loss} <= current average_loss:{average_loss}')
            return False
        else:
            torch.save(net, path)
            print(f'已更新模型{name},{old_average_loss}->{average_loss}')
            return True
    else:
        torch.save(net, path)
        print(f'已保存新模型{name}')
        return True


# 更新整体log记录
def update_log_overall(net):
    name = net.save_name
    date = net.date
    path = './logs/overall.json'
    data = dict()
    if os.path.exists(path):
        with open(path, mode='r') as fr:
            data = json.load(fr)
    data['best_'+name] = date
    with open(path, mode='w') as fw:
        json.dump(data, fw, indent=4)
    print('overall更新完毕')

# save_name是保存文件夹的名字，以此为标准来保存最优模型
def train(model_name, plus_opts, train_dataset=None, epoch_num=70, batch_size=16, save_name=None):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda:0')

    if train_dataset is None:
        train_dataset = CTDataset(kind='train')

    train_data_num = len(train_dataset)
    max_iterate_num = math.ceil(train_data_num / batch_size)*epoch_num
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 根据名称选取
    models = {
        'UNet': UNet,
        'SegNet': SegNet,
        'CircleUNet': CircleUNet,
    }

    if plus_opts is not None:
        if model_name == 'CircleUNet':
            net = models[model_name](circle_nums=plus_opts).to(device)
        else:
            net = models[model_name]().to(device)
    else:
        net = models[model_name]().to(device)

    if save_name is None:
        save_name = model_name
    net.save_name = save_name

    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    criterion = nn.BCELoss().to(device)

    print(f'开始训练{model_name}模型,epoch_num={epoch_num},batch_size={batch_size}')
    start_time = time.time()
    losses = []
    for epoch_index in range(epoch_num):
        for batch_index, (input_cts, target_masks) in enumerate(train_loader):
            input_cts = input_cts.to(device)
            target_masks = target_masks.to(device)
            optimizer.zero_grad()
            output_masks = net(input_cts)
            loss = criterion(output_masks, target_masks)
            loss.backward()
            optimizer.step()

            left_iterate_num = max_iterate_num - (epoch_index * int(train_data_num / batch_size) + batch_index + 1)
            print(f'完成第{batch_index + 1}次训练，还差{left_iterate_num}次,本轮loss为：{loss.item()}')
            losses.append(loss.item())
        print(f'完成第{epoch_index + 1}次epoch')
    training_time = round(time.time() - start_time)

    print(f'训练完成,共耗时{int(training_time // 3600)}时{int(training_time % 3600 // 60)}分{int(training_time % 60)}秒')
    updated = save_best_model(net, losses)
    if updated:
        update_log_overall(net)
        plt.plot(range(len(losses)), losses)
        plt.show()
    return net, losses, training_time, updated


# 训练网络、评估记录
def train_evaluate_and_save_log(net_name, plus_opts=None, save_name=None, note=None, show_examples=True, cycle_num=1):
    train_dataset = CTDataset('train', positive=True)
    test_dataset = CTDataset('test', positive=True)
    for i in range(cycle_num):
        net, losses, training_time, updated = train(net_name, plus_opts, train_dataset=train_dataset,
                                                    save_name=save_name)
        if show_examples:
            show_inference_examples(net, train_dataset, num=3)
        # 计算度量
        metrics = get_all_metrics(net, test_dataset)
        metrics_arr = list(metrics.items())
        metrics_arr.insert(0, ('training_time', training_time))
        if updated:
            save_log(net, all_metrics=dict(metrics_arr), losses=losses, note=note)


def train_CircleUNet(limit, leftNum, nums):
    if leftNum > 0:
        leftNum -= 1
        for i in range(limit[0], limit[1] + 1):
            new_nums = nums[:]
            new_nums.append(i)
            train_CircleUNet(limit, leftNum, new_nums)
    else:
        # print(nums)
        net_name = 'CircleUNet'
        plus_opts = nums
        save_name = f'{net_name}_{plus_opts[0]}_{plus_opts[1]}'
        train_evaluate_and_save_log(net_name=net_name, plus_opts=plus_opts, save_name=save_name, note=plus_opts,
                                    show_examples=False)


# 按照传入的circle_nums,分别训练CircleUNet
def train_CircleUNets(limit):
    train_CircleUNet(limit, 2, [])


def get_CircleUNet_info_by_metric(metric_name):
    result = dict()
    for i1 in range(1, 9):
        for i2 in range(1, 9):
            dir_path = f'./logs/CircleUNet_{i1}_{i2}'
            for date_path in glob(dir_path + '/*'):
                date = date_path.split('\\')[-1]
                for name in glob(date_path + '/*'):
                    if '.txt' in name:
                        fr = open(name, 'r')
                        metrics = json.load(fr)
                        # print(metrics)
                        if len(result.items()) == 0 or result['metrics'][metric_name] < metrics[metric_name]:
                            result['date'] = date
                            result['metrics'] = metrics
    return result