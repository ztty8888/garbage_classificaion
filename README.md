# **一、项目背景介绍**

垃圾分类的意义在于增加对环境的保护和降低垃圾的处理成本。减少土地侵蚀、提高经济价值、减少环境污染、保护生态环境变废为宝，有效利用资源；提高垃圾的资源价值和经济价值，力争物尽其用;可以减少垃圾处理量和处理设备，降低处理成本，减少土地资源的消耗;具有社会、经济、生态等几方面的效益。我们可以将模型部署在相关的硬件上来实现落地应用。



```python

```

# **二、数据介绍**


数据集来源： [垃圾分类](https://aistudio.baidu.com/aistudio/datasetdetail/108025)

归类领域：计算机视觉、分类任务

数据类型：图片、文件

保存格式：jpg,txt

垃圾分类数据集 包含6个分类：纸板（393），玻璃（491），金属（400），纸（584），塑料（472）和垃圾（127）。

![](https://ai-studio-static-online.cdn.bcebos.com/d4b70d1cf5d9436499037eaa3e77589c41dbcadee9f74d17bc8b91b3989f047c)

![](https://ai-studio-static-online.cdn.bcebos.com/f5816c7699cf4239ad102fc07b24095b27cdf0c30dfa41f1a86758e9168d92bd)

![](https://ai-studio-static-online.cdn.bcebos.com/d728c04b91f6413297337542eedc064a8168d3d237424855ab6c5092c800afe4)

![](https://ai-studio-static-online.cdn.bcebos.com/96ae98316156427b992b1f958c23253a5038b785d4bb4a9e9cb741511e6da891)

![](https://ai-studio-static-online.cdn.bcebos.com/0f0d4d4adfd24dd388c6d408bcb0d4fa39badfbebdf94cf7ae6e8b2c5023f8e0)

![](https://ai-studio-static-online.cdn.bcebos.com/4c70fd7d359043429d33e1b64846eb37525cc034a8664e01af08693fbac37fca)


## **2.1 解压数据集**

解压后把Garbage classification改为 Garbage_classification


```python
# 样例：解压你所挂载的数据集在同级目录下
!unzip -oq /home/aistudio/data/data128034/垃圾分类.zip  -d work/laji
# 查看数据集的目录结构
!tree work/laji -d
```

    work/laji
    ├── garbage classification
    │   └── Garbage classification
    │       ├── cardboard
    │       ├── glass
    │       ├── metal
    │       ├── paper
    │       ├── plastic
    │       └── trash
    └── Garbage classification
        └── Garbage classification
            ├── cardboard
            ├── glass
            ├── metal
            ├── paper
            ├── plastic
            └── trash
    
    16 directories


## **2.2 划分数据集**

'train'： 'val'： 'test'=0.8：0.1,0.1


```python
import os
import random
import shutil
from shutil import copy2

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                test_num = test_num + 1
            
            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print("{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))


if __name__ == '__main__':
    src_data_folder = "work/laji/Garbage_classification/Garbage_classification"
    target_data_folder = "work/laji/Garbage_classification"
    data_set_split(src_data_folder, target_data_folder)
```

    开始数据集划分
    *********************************paper*************************************
    paper类按照0.8：0.1：0.1的比例划分完成，一共594张图片
    训练集work/laji/Garbage_classification/train/paper：476张
    验证集work/laji/Garbage_classification/val/paper：59张
    测试集work/laji/Garbage_classification/test/paper：59张
    *********************************trash*************************************
    trash类按照0.8：0.1：0.1的比例划分完成，一共137张图片
    训练集work/laji/Garbage_classification/train/trash：110张
    验证集work/laji/Garbage_classification/val/trash：14张
    测试集work/laji/Garbage_classification/test/trash：13张
    *********************************glass*************************************
    glass类按照0.8：0.1：0.1的比例划分完成，一共501张图片
    训练集work/laji/Garbage_classification/train/glass：401张
    验证集work/laji/Garbage_classification/val/glass：50张
    测试集work/laji/Garbage_classification/test/glass：50张
    *********************************plastic*************************************
    plastic类按照0.8：0.1：0.1的比例划分完成，一共482张图片
    训练集work/laji/Garbage_classification/train/plastic：386张
    验证集work/laji/Garbage_classification/val/plastic：48张
    测试集work/laji/Garbage_classification/test/plastic：48张
    *********************************cardboard*************************************
    cardboard类按照0.8：0.1：0.1的比例划分完成，一共403张图片
    训练集work/laji/Garbage_classification/train/cardboard：323张
    验证集work/laji/Garbage_classification/val/cardboard：40张
    测试集work/laji/Garbage_classification/test/cardboard：40张
    *********************************metal*************************************
    metal类按照0.8：0.1：0.1的比例划分完成，一共410张图片
    训练集work/laji/Garbage_classification/train/metal：329张
    验证集work/laji/Garbage_classification/val/metal：41张
    测试集work/laji/Garbage_classification/test/metal：40张


# **三、定义部分超参数**


```python
import paddle
paddle.seed(8888)
import numpy as np
from typing import Callable
#参数配置
config_parameters = {
    "class_dim": 6,  #分类数
    "target_path":"/home/aistudio/work/laji/",                     
    'train_image_dir': 'work/laji/Garbage_classification/train',
    'eval_image_dir': 'work/laji/Garbage_classification/val',
    'epochs':50,
    'batch_size': 64,
    'lr': 0.01
}
#数据集的定义
class TowerDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, transforms: Callable, mode: str ='train'):
        """
        步骤二：实现构造函数，定义数据读取方式
        """
        super(TowerDataset, self).__init__()
        
        self.mode = mode
        self.transforms = transforms

        train_image_dir = config_parameters['train_image_dir']
        eval_image_dir = config_parameters['eval_image_dir']

        train_data_folder = paddle.vision.DatasetFolder(train_image_dir)
        eval_data_folder = paddle.vision.DatasetFolder(eval_image_dir)
        if self.mode  == 'train':
            self.data = train_data_folder
        elif self.mode  == 'eval':
            self.data = eval_data_folder

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = np.array(self.data[index][0]).astype('float32')

        data = self.transforms(data)

        label = np.array([self.data[index][1]]).astype('int64')
        
        return data, label
        
    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)


from paddle.vision import transforms as T

#数据增强
transform_train =T.Compose([T.Resize((256,256)),
                            #T.RandomVerticalFlip(10),
                            #T.RandomHorizontalFlip(10),
                            T.RandomRotation(10),
                            T.Transpose(),
                            T.Normalize(mean=[0, 0, 0],                           # 像素值归一化
                                        std =[255, 255, 255]),                    # transforms.ToTensor(), # transpose操作 + (img / 255),并且数据结构变为PaddleTensor
                            T.Normalize(mean=[0.50950350, 0.54632660, 0.57409690],# 减均值 除标准差    
                                        std= [0.26059777, 0.26041326, 0.29220656])# 计算过程：output[channel] = (input[channel] - mean[channel]) / std[channel]
                            ])
transform_eval =T.Compose([ T.Resize((256,256)),
                            T.Transpose(),
                            T.Normalize(mean=[0, 0, 0],                           # 像素值归一化
                                        std =[255, 255, 255]),                    # transforms.ToTensor(), # transpose操作 + (img / 255),并且数据结构变为PaddleTensor
                            T.Normalize(mean=[0.50950350, 0.54632660, 0.57409690],# 减均值 除标准差    
                                        std= [0.26059777, 0.26041326, 0.29220656])# 计算过程：output[channel] = (input[channel] - mean[channel]) / std[channel]
                            ])

train_dataset = TowerDataset(mode='train',transforms=transform_train)
eval_dataset  = TowerDataset(mode='eval', transforms=transform_eval )
#数据异步加载
train_loader = paddle.io.DataLoader(train_dataset, 
                                    places=paddle.CUDAPlace(0), 
                                    batch_size=16, 
                                    shuffle=True,
                                    #num_workers=2,
                                    #use_shared_memory=True
                                    )
eval_loader = paddle.io.DataLoader (eval_dataset, 
                                    places=paddle.CUDAPlace(0), 
                                    batch_size=16,
                                    #num_workers=2,
                                    #use_shared_memory=True
                                    )

print('训练集样本量: {}，验证集样本量: {}'.format(len(train_loader), len(eval_loader)))
```

    训练集样本量: 127，验证集样本量: 16


# **四、模型介绍——MobileViT组网**

MobileViT：一种用于移动设备的轻量级通用视觉Transformer，据作者称，这是首个能比肩轻量级CNN网络性能的轻量级ViT工作，表现SOTA！性能优于MobileNetV3、CrossViT等网络。

轻量级卷积神经网络 (CNN) 是移动视觉任务的de-facto。他们的空间归纳偏差使他们能够在不同的视觉任务中以较少的参数学习表示。然而，这些网络在空间上是局部的。为了学习全局表示，已经采用了基于自注意力的视觉Transformer（ViT）。与 CNN 不同，ViT 是"重量级"的。在本文中，我们提出以下问题：是否有可能结合 CNNs 和 ViTs 的优势，为移动视觉任务构建一个轻量级、低延迟的网络？为此，我们推出了 MobileViT，这是一种用于移动设备的轻量级通用视觉Transformer。

结构上也非常简单，但是同样能够实现一个不错的精度表现

原论文下载：https://arxiv.org/pdf/2110.02178.pdf

MobileViT 与 Mobilenet 系列模型一样模型的结构都十分简单

MobileViT带来了一些新的结果:

1.更好的性能:在相同的参数情况下,余现有的轻量级CNN相比,mobilevit模型在不同的移动视觉任务中实现了更好的性能.

2.更好的泛化能力:泛化能力是指训练和评价指标之间的差距.对于具有相似的训练指标的两个模型,具有更好评价指标的模型更具有通用性,因为它可以更好地预测未见的数据集.与CNN相比,即使有广泛的数据增强,其泛化能力也很差,mobilevit显示出更好的泛化能力(如下图).

3.更好的鲁棒性:一个好的模型应该对超参数具有鲁棒性,因为调优这些超参数会消耗时间和资源.与大多数基于ViT的模型不同,mobilevit模型使用基于增强训练,与L2正则化不太敏感.

![](https://ai-studio-static-online.cdn.bcebos.com/4090a9935e794caab9332f03b88ddcf76692c80b3086464faacb790b118d4063)



```python
import paddle
import paddle.nn as nn




def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2D(inp, oup, kernal_size, stride, 1, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )


class PreNorm(nn.Layer):
    def __init__(self, axis, fn):
        super().__init__()
        self.norm = nn.LayerNorm(axis)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Layer):
    def __init__(self, axis, hidden_axis, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(axis, hidden_axis),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_axis, axis),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Layer):
    def __init__(self, axis, heads=8, axis_head=64, dropout=0.):
        super().__init__()
        inner_axis = axis_head *  heads
        project_out = not (heads == 1 and axis_head == axis)

        self.heads = heads
        self.scale = axis_head ** -0.5

        self.attend = nn.Softmax(axis = -1)
        self.to_qkv = nn.Linear(axis, inner_axis * 3, bias_attr = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_axis, axis),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
 
        q,k,v = self.to_qkv(x).chunk(3, axis=-1)

        b,p,n,hd = q.shape
        b,p,n,hd = k.shape
        b,p,n,hd = v.shape
        q = q.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))
        k = k.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))
        v = v.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))

        dots = paddle.matmul(q, k.transpose((0, 1, 2, 4, 3))) * self.scale
        attn = self.attend(dots)

        out = (attn.matmul(v)).transpose((0, 1, 3, 2, 4)).reshape((b, p, n,-1))
        return self.to_out(out)



class Transformer(nn.Layer):
    def __init__(self, axis, depth, heads, axis_head, mlp_axis, dropout=0.):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(axis, Attention(axis, heads, axis_head, dropout)),
                PreNorm(axis, FeedForward(axis, mlp_axis, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Layer):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_axis = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(hidden_axis, hidden_axis, 3, stride, 1, groups=hidden_axis, bias_attr=False),
                nn.BatchNorm2D(hidden_axis),
                nn.Silu(),
                # pw-linear
                nn.Conv2D(hidden_axis, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2D(inp, hidden_axis, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_axis),
                nn.Silu(),
                # dw
                nn.Conv2D(hidden_axis, hidden_axis, 3, stride, 1, groups=hidden_axis, bias_attr=False),
                nn.BatchNorm2D(hidden_axis),
                nn.Silu(),
                # pw-linear
                nn.Conv2D(hidden_axis, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Layer):
    def __init__(self, axis, depth, channel, kernel_size, patch_size, mlp_axis, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, axis)

        self.transformer = Transformer(axis, depth, 1, 32, mlp_axis, dropout)

        self.conv3 = conv_1x1_bn(axis, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        n, c, h, w = x.shape

        x = x.transpose((0,3,1,2)).reshape((n,self.ph * self.pw,-1,c))
        x = self.transformer(x)
        x = x.reshape((n,h,-1,c)).transpose((0,3,1,2))


        # Fusion
        x = self.conv3(x)
        x = paddle.concat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Layer):
    def __init__(self, image_size, axiss, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.LayerList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.LayerList([])
        self.mvit.append(MobileViTBlock(axiss[0], L[0], channels[5], kernel_size, patch_size, int(axiss[0]*2)))
        self.mvit.append(MobileViTBlock(axiss[1], L[1], channels[7], kernel_size, patch_size, int(axiss[1]*4)))
        self.mvit.append(MobileViTBlock(axiss[2], L[2], channels[9], kernel_size, patch_size, int(axiss[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2D(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias_attr=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x)
        x = x.reshape((-1, x.shape[1]))
        x = self.fc(x)
        return x


def mobilevit_xxs():
    axiss = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), axiss, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    axiss = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), axiss, channels, num_classes=1000)


def mobilevit_s():
    axiss = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), axiss, channels, num_classes=6)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

```

## **4.1 模型测试**


```python
if __name__ == '__main__':
    img = paddle.rand([5, 3, 256, 256])

    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)


    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)


    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
```

    W0226 12:44:54.582399   101 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0226 12:44:54.593559   101 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:653: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")


    [5, 1000]
    [5, 1000]
    [5, 6]


# **五、模型训练**

## **5.1实例化模型**


```python
model = mobilevit_s()
model = paddle.Model(model)
```


```python
#优化器选择
class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self, target=0.5, path='work/best_model2', verbose=0):
        self.target = target
        self.epoch = None
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        if logs.get('acc') > self.target:
            self.target = logs.get('acc')
            self.model.save(self.path)
            print('best acc is {} at epoch {}'.format(self.target, self.epoch))

callback_visualdl = paddle.callbacks.VisualDL(log_dir='work/no_SA')
callback_savebestmodel = SaveBestModel(target=0.5, path='work/best_model1')
callbacks = [callback_visualdl, callback_savebestmodel]

base_lr = config_parameters['lr']
epochs = config_parameters['epochs']

def make_optimizer(parameters=None):
    momentum = 0.9

    learning_rate= paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=epochs, verbose=False)
    weight_decay=paddle.regularizer.L2Decay(0.0001)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        parameters=parameters)
    return optimizer

optimizer = make_optimizer(model.parameters())


model.prepare(optimizer,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
```

## **5.2 训练模型需要说明：**

**lr_schedule:** CosineAnnealingDecay

**optimize:** Momentum

**epoch:** 50
 
**batch_size:** 64

**Loss function:** CrossEntropyLoss


```python
model.fit(train_loader,
          eval_loader,
          epochs=50,
          batch_size=64,     # 是否打乱样本集     
          callbacks=callbacks, 
          verbose=1)   # 日志展示格式
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/50


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and


    step 127/127 [==============================] - loss: 0.9921 - acc: 0.4785 - 160ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 3.0617 - acc: 0.5159 - 149ms/step         
    Eval samples: 252
    best acc is 0.5158730158730159 at epoch 0
    Epoch 2/50
    step 127/127 [==============================] - loss: 0.5079 - acc: 0.5714 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.5597 - acc: 0.6270 - 134ms/step         
    Eval samples: 252
    best acc is 0.626984126984127 at epoch 1
    Epoch 3/50
    step 127/127 [==============================] - loss: 1.3110 - acc: 0.5995 - 153ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.2476 - acc: 0.6429 - 133ms/step         
    Eval samples: 252
    best acc is 0.6428571428571429 at epoch 2
    Epoch 4/50
    step 127/127 [==============================] - loss: 0.6648 - acc: 0.6380 - 158ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.7209 - acc: 0.5675 - 132ms/step         
    Eval samples: 252
    Epoch 5/50
    step 127/127 [==============================] - loss: 1.1874 - acc: 0.6563 - 157ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.6133 - acc: 0.6429 - 130ms/step         
    Eval samples: 252
    Epoch 6/50
    step 127/127 [==============================] - loss: 0.9289 - acc: 0.6854 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.3293 - acc: 0.6865 - 130ms/step         
    Eval samples: 252
    best acc is 0.6865079365079365 at epoch 5
    Epoch 7/50
    step 127/127 [==============================] - loss: 0.7384 - acc: 0.6943 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.7852 - acc: 0.6032 - 130ms/step         
    Eval samples: 252
    Epoch 8/50
    step 127/127 [==============================] - loss: 1.6061 - acc: 0.7264 - 157ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.4986 - acc: 0.6944 - 130ms/step         
    Eval samples: 252
    best acc is 0.6944444444444444 at epoch 7
    Epoch 9/50
    step 127/127 [==============================] - loss: 0.9896 - acc: 0.7610 - 152ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.5544 - acc: 0.7063 - 130ms/step         
    Eval samples: 252
    best acc is 0.7063492063492064 at epoch 8
    Epoch 10/50
    step 127/127 [==============================] - loss: 0.5369 - acc: 0.7689 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.8893 - acc: 0.7103 - 130ms/step         
    Eval samples: 252
    best acc is 0.7103174603174603 at epoch 9
    Epoch 11/50
    step 127/127 [==============================] - loss: 0.8750 - acc: 0.7891 - 156ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.8423 - acc: 0.6190 - 127ms/step         
    Eval samples: 252
    Epoch 12/50
    step 127/127 [==============================] - loss: 1.2786 - acc: 0.7832 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.5781 - acc: 0.7024 - 138ms/step         
    Eval samples: 252
    Epoch 13/50
    step 127/127 [==============================] - loss: 0.4109 - acc: 0.8069 - 153ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.3971 - acc: 0.7460 - 138ms/step         
    Eval samples: 252
    best acc is 0.746031746031746 at epoch 12
    Epoch 14/50
    step 127/127 [==============================] - loss: 1.1973 - acc: 0.8178 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.0122 - acc: 0.7460 - 130ms/step         
    Eval samples: 252
    Epoch 15/50
    step 127/127 [==============================] - loss: 0.1630 - acc: 0.8242 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6996 - acc: 0.7302 - 131ms/step         
    Eval samples: 252
    Epoch 16/50
    step 127/127 [==============================] - loss: 0.3381 - acc: 0.8415 - 159ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.1308 - acc: 0.7024 - 131ms/step         
    Eval samples: 252
    Epoch 17/50
    step 127/127 [==============================] - loss: 0.2470 - acc: 0.8400 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6880 - acc: 0.7698 - 129ms/step         
    Eval samples: 252
    best acc is 0.7698412698412699 at epoch 16
    Epoch 18/50
    step 127/127 [==============================] - loss: 0.2109 - acc: 0.8583 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.5792 - acc: 0.7897 - 131ms/step         
    Eval samples: 252
    best acc is 0.7896825396825397 at epoch 17
    Epoch 19/50
    step 127/127 [==============================] - loss: 0.2365 - acc: 0.8677 - 160ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.4787 - acc: 0.5794 - 127ms/step         
    Eval samples: 252
    Epoch 20/50
    step 127/127 [==============================] - loss: 0.3634 - acc: 0.8444 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6328 - acc: 0.7579 - 129ms/step         
    Eval samples: 252
    Epoch 21/50
    step 127/127 [==============================] - loss: 0.6545 - acc: 0.8889 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.3350 - acc: 0.7619 - 130ms/step         
    Eval samples: 252
    Epoch 22/50
    step 127/127 [==============================] - loss: 0.3918 - acc: 0.9052 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 2.1700 - acc: 0.6389 - 133ms/step         
    Eval samples: 252
    Epoch 23/50
    step 127/127 [==============================] - loss: 0.4210 - acc: 0.8854 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.2249 - acc: 0.6905 - 139ms/step         
    Eval samples: 252
    Epoch 24/50
    step 127/127 [==============================] - loss: 0.4640 - acc: 0.9017 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.8004 - acc: 0.7579 - 130ms/step         
    Eval samples: 252
    Epoch 25/50
    step 127/127 [==============================] - loss: 0.8252 - acc: 0.9037 - 156ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.5250 - acc: 0.8016 - 131ms/step         
    Eval samples: 252
    best acc is 0.8015873015873016 at epoch 24
    Epoch 26/50
    step 127/127 [==============================] - loss: 0.2347 - acc: 0.9126 - 156ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.6241 - acc: 0.7341 - 131ms/step         
    Eval samples: 252
    Epoch 27/50
    step 127/127 [==============================] - loss: 0.1988 - acc: 0.9067 - 153ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.9304 - acc: 0.7381 - 132ms/step         
    Eval samples: 252
    Epoch 28/50
    step 127/127 [==============================] - loss: 0.3173 - acc: 0.9225 - 152ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.5695 - acc: 0.7738 - 131ms/step         
    Eval samples: 252
    Epoch 29/50
    step 127/127 [==============================] - loss: 0.1067 - acc: 0.9363 - 196ms/step        
    Eval begin...
    step 16/16 [==============================] - loss: 0.8192 - acc: 0.7262 - 266ms/step         
    Eval samples: 252
    Epoch 30/50
    step 127/127 [==============================] - loss: 0.1226 - acc: 0.9235 - 159ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6472 - acc: 0.7341 - 132ms/step         
    Eval samples: 252
    Epoch 31/50
    step 127/127 [==============================] - loss: 0.2043 - acc: 0.9333 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.5145 - acc: 0.7937 - 133ms/step         
    Eval samples: 252
    Epoch 32/50
    step 127/127 [==============================] - loss: 0.1612 - acc: 0.9328 - 157ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.2310 - acc: 0.7897 - 133ms/step         
    Eval samples: 252
    Epoch 33/50
    step 127/127 [==============================] - loss: 0.3547 - acc: 0.9452 - 158ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.1430 - acc: 0.8135 - 132ms/step         
    Eval samples: 252
    best acc is 0.8134920634920635 at epoch 32
    Epoch 34/50
    step 127/127 [==============================] - loss: 0.2088 - acc: 0.9472 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.0760 - acc: 0.7976 - 131ms/step         
    Eval samples: 252
    Epoch 35/50
    step 127/127 [==============================] - loss: 0.1186 - acc: 0.9506 - 152ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.9217 - acc: 0.7579 - 136ms/step         
    Eval samples: 252
    Epoch 36/50
    step 127/127 [==============================] - loss: 0.8777 - acc: 0.9610 - 152ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.3000 - acc: 0.7976 - 134ms/step         
    Eval samples: 252
    Epoch 37/50
    step 127/127 [==============================] - loss: 0.2458 - acc: 0.9506 - 158ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6378 - acc: 0.7897 - 131ms/step         
    Eval samples: 252
    Epoch 38/50
    step 127/127 [==============================] - loss: 0.0612 - acc: 0.9664 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.0060 - acc: 0.7540 - 132ms/step         
    Eval samples: 252
    Epoch 39/50
    step 127/127 [==============================] - loss: 0.0625 - acc: 0.9709 - 157ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.1978 - acc: 0.7659 - 133ms/step         
    Eval samples: 252
    Epoch 40/50
    step 127/127 [==============================] - loss: 0.0895 - acc: 0.9546 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.3882 - acc: 0.8214 - 132ms/step         
    Eval samples: 252
    best acc is 0.8214285714285714 at epoch 39
    Epoch 41/50
    step 127/127 [==============================] - loss: 0.0152 - acc: 0.9551 - 156ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.9404 - acc: 0.7659 - 132ms/step         
    Eval samples: 252
    Epoch 42/50
    step 127/127 [==============================] - loss: 0.0526 - acc: 0.9575 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.3851 - acc: 0.7619 - 126ms/step         
    Eval samples: 252
    Epoch 43/50
    step 127/127 [==============================] - loss: 0.0533 - acc: 0.9640 - 157ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.2015 - acc: 0.8175 - 132ms/step         
    Eval samples: 252
    Epoch 44/50
    step 127/127 [==============================] - loss: 0.0223 - acc: 0.9585 - 157ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.7832 - acc: 0.7817 - 128ms/step         
    Eval samples: 252
    Epoch 45/50
    step 127/127 [==============================] - loss: 0.3739 - acc: 0.9526 - 154ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6105 - acc: 0.7381 - 130ms/step         
    Eval samples: 252
    Epoch 46/50
    step 127/127 [==============================] - loss: 0.0931 - acc: 0.9511 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 1.0364 - acc: 0.7698 - 133ms/step         
    Eval samples: 252
    Epoch 47/50
    step 127/127 [==============================] - loss: 0.9378 - acc: 0.9610 - 158ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.0733 - acc: 0.7659 - 133ms/step         
    Eval samples: 252
    Epoch 48/50
    step 127/127 [==============================] - loss: 0.3155 - acc: 0.9654 - 155ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.1928 - acc: 0.7778 - 134ms/step         
    Eval samples: 252
    Epoch 49/50
    step 127/127 [==============================] - loss: 0.0611 - acc: 0.9659 - 160ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.3516 - acc: 0.7540 - 133ms/step         
    Eval samples: 252
    Epoch 50/50
    step 127/127 [==============================] - loss: 0.3301 - acc: 0.9783 - 158ms/step         
    Eval begin...
    step 16/16 [==============================] - loss: 0.6451 - acc: 0.7897 - 132ms/step         
    Eval samples: 252


# **六、模型评估**


```python
# 模型评估
# 加载训练过程保存的最后一个模型
model__state_dict = paddle.load('work/best_model1.pdparams')
model_eval = mobilevit_s()
model_eval.set_state_dict(model__state_dict) 
model_eval.eval()
accs = []
# 开始评估
for _, data in enumerate(eval_loader()):
    x_data = data[0]
    y_data = data[1]
    predicts = model_eval(x_data)
    acc = paddle.metric.accuracy(predicts, y_data)
    accs.append(acc.numpy()[0])
print('模型在验证集上的准确率为：',np.mean(accs))
```

    模型在验证集上的准确率为： 0.8203125


# **七、可视化测试模型效果**


```python
import time
import sys
from PIL import Image
import matplotlib.pyplot as plt
# 加载训练过程保存的最后一个模型
model__state_dict = paddle.load('work/best_model1.pdparams')
model_predict = mobilevit_s()
model_predict.set_state_dict(model__state_dict) 
model_predict.eval()
infer_imgs_path = os.listdir('work/laji/Garbage_classification/test/glass')
# print(infer_imgs_path)

label_dic={'0': 'cardboard', '1': 'plastic', '2': 'metal', '3': 'paper','4':' glass','5':'trash'}

def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path) 
    if img.mode != 'RGB': 
        img = img.convert('RGB') 
    img = img.resize((256, 256), Image.BILINEAR)
    img = np.array(img).astype('float32') 
    img = img.transpose((2, 0, 1)) / 255 # HWC to CHW 及归一化
    return img

# 预测所有图片
for infer_img_path in infer_imgs_path:
    infer_img = load_image('work/laji/Garbage_classification/test/glass/'+infer_img_path)
    infer_img = infer_img[np.newaxis,:, : ,:]  #reshape(-1,3,224,224)
    infer_img = paddle.to_tensor(infer_img)
    result = model_predict(infer_img)
    lab = np.argmax(result.numpy())
    print("样本: {},被预测为:{}".format(infer_img_path,label_dic[str(lab)]))
    img = Image.open('work/laji/Garbage_classification/test/glass/'+infer_img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    sys.stdout.flush()
    time.sleep(0.5)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


    样本: glass490.jpg,被预测为:plastic


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data



    <Figure size 640x480 with 1 Axes>


    样本: glass212.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass138.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass412.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass24.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass222.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass330.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass57.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass90.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass221.jpg,被预测为:cardboard



    <Figure size 640x480 with 1 Axes>


    样本: glass348.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass349.jpg,被预测为:paper



    <Figure size 640x480 with 1 Axes>


    样本: glass418.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass195.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass365.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass477.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass8.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass465.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass95.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass191.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass403.jpg,被预测为:paper



    <Figure size 640x480 with 1 Axes>


    样本: glass75.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass192.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass353.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass26.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass432.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass169.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass448.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass474.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass250.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass189.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass464.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass39.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass357.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass190.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass43.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass229.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass82.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass396.jpg,被预测为:paper



    <Figure size 640x480 with 1 Axes>


    样本: glass211.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass49.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass263.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass244.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass492.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass466.jpg,被预测为:paper



    <Figure size 640x480 with 1 Axes>


    样本: glass262.jpg,被预测为:cardboard



    <Figure size 640x480 with 1 Axes>


    样本: glass20.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


    样本: glass355.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass182.jpg,被预测为:plastic



    <Figure size 640x480 with 1 Axes>


    样本: glass279.jpg,被预测为: glass



    <Figure size 640x480 with 1 Axes>


## **7.1测试模型效果**

可以看到预测预测效果还是挺好的

![](https://ai-studio-static-online.cdn.bcebos.com/7c6e19c826054d93af6a977cd04dbf306cbafd104dba4f1bb8a39f6e638d99c8)


# **总结与升华**

分类模型Mobile-ViT实现一个垃圾分类检测即可完成，MobileViT：一种用于移动设备的轻量级通用视觉Transformer，据作者称，这是首个能比肩轻量级CNN网络性能的轻量级ViT工作，表现SOTA！性能优于MobileNetV3、CrossViT等网络。本次项目实质是属于一个6分类的问题。
还有一些项目在实践时，遇到的问题可能是环境问题，后面会在本地再测试下。早期的遇到的问题就是自己对数据进行处理后送入网络，报图片尺寸、维度错误，debug后也找不到问题，然后fork原作者项目进行操作后就可以跑的通，具体之前问题出在哪儿，仍然需要进一步测试验证。

# **参考项目：**

[Mobile-ViT：改进的一种更小更轻精度更高的模型](https://aistudio.baidu.com/aistudio/projectdetail/2683037?channelType=0&channel=0)

# **关于作者**

>- [个人主页](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/1032881)

>- 感兴趣的方向为：目标检测，图像分类，图像分割等。

>- 不定期更新感兴趣的CV比赛baseline等

>- 个人荣誉：飞桨开发者青铜奖

>- 欢迎大家有问题留言交流学习，共同进步成长。
