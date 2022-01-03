![image](https://user-images.githubusercontent.com/96910478/147904731-a1c9fc3c-70c3-496c-9b2e-68a8b095e6d7.png){:height="50%" width="50%"}
![image](https://user-images.githubusercontent.com/96910478/147904739-71fe0c6d-e177-467e-8214-2f117c044600.png){:height="50%" width="50%"}
# Circle Unet
## Introduction

This is an end-to-end neural network based on the typical Unet. The main innovation of the architecture is the 'circle' unit which can repeat skip connection procedure to reserve more local information in high level feature maps. The data of experiment suggests that this idea did work. The code is Pytorch version. There is no other version currently.

## Pre-work

### Environment Construction

- Anaconda v1.9.0 (with Python 3.9.7 interpreter)
- CUDA 11.3
- pytorch 1.10.1(Anaconda CUDA 11.3 version)
- Medpy 0.4.0

### Dataset Download

The dataset of experiments is a lower abdominal CT dataset from 'The 7th Teddy Cup Data Mining Challenge' and was performed desensitization and properly filtered. It is available by two approaches as follows.

1. Download the pre-treating data (.npz) , which has been compressed as numpy arrays and divided into training(9/10) and test(1/10) set following random sampling.
   [Baidu Netdisk link, code: pj6i](https://pan.baidu.com/s/1IlNjk3YA9OqNpfMIJ206XA)

   Note: file need to be saved in './dataset' to make functions read correctly.

2. Download raw data(.zip) which need to refer to './CTDataset' to build your own data class.
   [Baidu Netdisk link, code: x6el](https://pan.baidu.com/s/1TOutToh1G2k8aSyknOjX9Q)
   Note: file need to be unzipped and saved in './dataset' to make functions read correctly.

### Pre-trained Model Download(Optional)

If you want to skip the procedure of training, the pre-trained model is available.

[Baidu Netdisk link, code: pfit](https://pan.baidu.com/s/1aWoYywyh41iVWi5BcHp5Mg)

Note: files need to be saved in './pretrained_models' to make functions read correctly.

## Usages

### Sample

```python
from utils.inference import *
from utils.CTDataset import *


circle_u_net = load_pretrained_model('CircleUNet')
test_dataset = CTDataset('test', positive=True)
show_inference_examples(u_net, test_dataset)
```

### Comparing Performance

```python
from utils.inference import *
from utils.CTDataset import *
from utils.performance import *


circle_u_net = load_pretrained_model('CircleUNet')
u_net = load_pretrained_model('UNet')
seg_net = load_pretrained_model('UNet')
test_dataset = CTDataset('test', positive=True)

m1 = get_all_metrics(circle_u_net, test_dataset)
m2 = get_all_metrics(u_net, test_dataset)
m3 = get_all_metrics(seg_net, test_dataset)
```

### Training, Evaluating and Saving Best Models

```python
from utils.train import train_evaluate_and_save_log


net_name = 'CircleUNet'
plus_opts = [0, 1, 3, 7]
save_name = f'{net_name}_{"".join(list(map(str,plus_opts)))}'
train_evaluate_and_save_log(net_name=net_name, plus_opts=plus_opts, save_name=save_name, note=plus_opts,
                             show_examples=False)
```

