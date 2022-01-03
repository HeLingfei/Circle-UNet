from utils.inference import inference
import numpy as np
from medpy import metric
import time


def average_performance(targets, segmentations, performance_func):
    arr = zip(targets, segmentations)
    s = 0
    for target, segmentation in arr:
        s += performance_func(segmentation, target)
    return s/len(targets)


def get_average_metric(net, dataset, metric_name):
    performance_func_arr = {
        'dice': metric.dc,
        'sensitivity': metric.sensitivity,
        'specificity': metric.specificity,
        'precision': metric.precision,
        'assd': metric.assd
    }

    targets = []
    segmentations = []
    print(f'开始计算 average {metric_name} 指标，共需测试{len(dataset)}个样本')
    for ct, target_mask in dataset:
        targets.append(target_mask[0])
        output = inference(net, ct[0], binary=True)
        segmentations.append(output)
    performance = average_performance(targets, segmentations, performance_func=performance_func_arr[metric_name])
    print(f'指标计算结束，average {metric_name}:{performance}')
    return performance


def get_all_metrics(net, dataset):
    performance_func_arr = {
        'dice': metric.dc,
        'sensitivity': metric.sensitivity,
        'specificity': metric.specificity,
        'precision': metric.precision,
        'assd': metric.assd
    }

    metrics = dict()

    targets = []
    segmentations = []
    print(f"开始计算指标，共需测试{len(dataset)}个样本")
    print('正在进行 性能average 指标计算')
    t = time.time()
    for ct, target_mask in dataset:
        targets.append(target_mask[0])
        output = inference(net, ct[0], binary=True)
        segmentations.append(output)

    average_time = (time.time()-t)/len(dataset)
    key = 'average inference time'
    metrics[key] = average_time

    print(f'正在进行 效果average 指标计算')
    print(f'{key}:{average_time}')
    for metric_name in performance_func_arr:
        performance = average_performance(targets, segmentations, performance_func=performance_func_arr[metric_name])
        key = 'average '+metric_name
        value = performance
        print(f'{key}:{value}')
        metrics[key] = value
    print('指标计算结束')
    return metrics


# 下面是具体的指标计算方法，输入输出都是图像的二维numpy数组


# # dice指标
# def dice(tar, seg):
#     # 用来防止分母为0
#     smooth = 1
#     intersection = (tar.flatten()*seg.flatten()).sum()
#     return (2 * intersection + smooth)/(tar.sum()+seg.sum()+smooth)
#
#
# # 精确度指标
# def PPV(tar, seg):
#     predict = np.atleast_1d(seg.astype(np.bool))
#     target = np.atleast_1d(tar.astype(np.bool))
#
#     tp = np.count_nonzero(predict & target)
#     fp = np.count_nonzero(predict & ~target)
#
#     try:
#         p = tp / float(tp + fp)
#     except ZeroDivisionError:
#         p = 0.0
#     return p


