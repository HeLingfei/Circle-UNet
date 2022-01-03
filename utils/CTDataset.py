from torch.utils.data.dataset import Dataset, T_co
import numpy as np
import glob
import os
from PIL import Image


# 传入512*512图片，返回剪裁后并归一化的的numpy数组
def get_clip_arr(img):
    img = img.resize((512,512))
    img = img.convert('L')
    return np.array(img, dtype='float32')[256:, 128:384] / 255


# 从指定文件夹加载数据
def load_data_from_dir(dir_name, mode='L', positive=False):
    ct_arrs = []
    dir_names = glob.glob(f'./{dir_name}/train/*')
    print(f'正在加载train的图片中...')
    for i, name in enumerate(dir_names):
        for img_path in glob.glob(name + '/*.png'):
            img = Image.open(img_path)
            # 剪裁图片，直肠肿瘤只可能出现在直肠区域
            img_arr = get_clip_arr(img)
            ct_arrs.append(img_arr)
    print(f'加载train的图片完成')

    mask_arrs = []
    dir_names = glob.glob(f'./{dir_name}/label/*')
    print(f'正在加载label的图片中...')
    for i, name in enumerate(dir_names):
        for img_path in glob.glob(name + '/*.png'):
            img = Image.open(img_path)
            # 剪裁图片，直肠肿瘤只可能出现在直肠区域
            img_arr = get_clip_arr(img)
            mask_arrs.append(img_arr)
    print(f'加载label的图片完成')

    result_ct_arrs = []
    result_mask_arrs = []
    # 筛选有肿瘤的
    if positive:
        print('开始筛选')
        for i, img_arr in enumerate(mask_arrs):
            if np.any(img_arr > 0):
                result_ct_arrs.append(ct_arrs[i])
                result_mask_arrs.append(img_arr)
        print('筛选结束')
    else:
        result_ct_arrs = ct_arrs
        result_mask_arrs = mask_arrs

    result_ct_arrs = np.array(result_ct_arrs)
    result_ct_arrs = np.expand_dims(result_ct_arrs, axis=1)
    result_mask_arrs = np.array(result_mask_arrs)
    result_mask_arrs = np.expand_dims(result_mask_arrs, axis=1)

    return result_ct_arrs, result_mask_arrs


class CTDataset(Dataset):
    def __init__(self, kind='train', positive=False) -> None:
        self.kind = kind
        # 判断是否有现成的数据
        if os.path.exists('./dataset/dataset.npz'):
            self.data = self.load_data_from_npz()
        else:
            self.data = self.load_data_from_files(positive=positive)

    def __getitem__(self, index) -> T_co:
        return self.data['ct'][index], self.data['mask'][index]

    def get_random_item(self):
        index = np.random.randint(0, len(self) - 1)
        return self.__getitem__(index)

    def __len__(self):
        return len(self.data['ct'])

    # 从npz文件中加载数据
    def load_data_from_npz(self):
        print('正在加载.npz数据...')
        data = np.load('./dataset/dataset.npz', allow_pickle=True)
        print('数据加载完成')
        return {'ct': data[self.kind+'_ct'], 'mask': data[self.kind+'_mask']}

    # 从文件夹中加载数据
    def load_data_from_files(self, positive=False):
        ct_arrs, mask_arrs = load_data_from_dir('dataset', positive=positive)

        # 打乱数据集
        np.random.seed(116)
        np.random.shuffle(ct_arrs)
        np.random.seed(116)
        np.random.shuffle(mask_arrs)

        span = int(ct_arrs.shape[0]/10*9)
        # 9：1分割为训练集和测试集
        print(f'训练集shape:{ct_arrs[:span].shape},测试集shape:{ct_arrs[span:].shape}')
        train_data = {'ct': ct_arrs[:span], 'mask': mask_arrs[:span]}
        test_data = {'ct': ct_arrs[span:], 'mask': mask_arrs[span:]}
        np.savez('./dataset/dataset.npz', train_ct=train_data['ct'], train_mask=train_data['mask'], test_ct=test_data['ct'], test_mask=test_data['mask'],)
        if self.kind == 'train':
            return train_data
        else:
            return test_data


# dataset = CTDataset()
# print(dataset[0])
#
# a = np.array([[[1],[2]],[[2],[2]],[[3],[2]]])
# print(a.shape)
# b = np.expand_dims(a, axis=1)
# print(b.shape)
# print(b)
