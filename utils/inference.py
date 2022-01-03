import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_pretrained_model(model_name):
    print(f'加载{model_name}模型中...')
    net = torch.load('./pretrained_models/' + model_name + '.pkl')
    print('模型加载成功')
    return net


def inference(net, input_ct, binary=False):
    input_ct = np.expand_dims(input_ct, axis=0)
    input_ct = torch.from_numpy(np.expand_dims(input_ct, axis=0)).cuda()
    outputs = net(input_ct)
    output = outputs.cpu().detach().numpy()[0][0]
    if binary:
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
    return output


# 展示三张图
def show_test(ct, target_mask, output_mask, net_name):
    # 在我的 notebook 里，要设置下面两行才能显示中文
    plt.rcParams['font.family'] = ['sans-serif']
    # 如果是在 PyCharm 里，只要下面一行，上面的一行可以删除
    plt.rcParams['font.sans-serif'] = ['SimHei']

    ct = Image.fromarray(ct*255)
    target_mask = Image.fromarray(target_mask*255)
    output_mask = Image.fromarray(output_mask*255)

    plt.suptitle(f'随机测试图片结果展示')

    plt.subplot(131)
    plt.imshow(ct)
    plt.axis("off")
    plt.title(f'CT图')

    plt.subplot(132)
    plt.imshow(target_mask)
    plt.axis("off")
    plt.title(f'原Mask')

    plt.subplot(133)
    plt.imshow(output_mask)
    plt.axis("off")
    plt.title(f'{net_name} Mask')

    plt.show()


# 展示几张该网络的例子
def show_inference_examples(net, dataset, num=1, indexes=()):
    if len(indexes) == 0:
        # 没有指定indexes就按num随机展示
        for i in range(num):
            ct, mask = dataset.get_random_item()
            output = inference(net, ct[0], binary=True)
            show_test(ct[0], target_mask=mask[0], output_mask=output, net_name=net.__class__.__name__)
    else:
        # 否则就按指定的indexes展示
        for i in indexes:
            ct, mask = dataset[i]
            output = inference(net, ct[0], binary=True)
            show_test(ct[0], target_mask=mask[0], output_mask=output, net_name=net.__class__.__name__)