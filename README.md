# **一、项目背景介绍**

垃圾分类的意义在于增加对环境的保护和降低垃圾的处理成本。减少土地侵蚀、提高经济价值、减少环境污染、保护生态环境变废为宝，有效利用资源；提高垃圾的资源价值和经济价值，力争物尽其用;可以减少垃圾处理量和处理设备，降低处理成本，减少土地资源的消耗;具有社会、经济、生态等几方面的效益。我们可以将模型部署在相关的硬件上来实现落地应用。

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

\# 样例：解压你所挂载的数据集在同级目录下

!unzip -oq /home/aistudio/data/data128034/垃圾分类.zip -d work/laji

\# 查看数据集的目录结构

!tree work/laji -d

## **2.2 划分数据集**

'train'： 'val'： 'test'=0.8：0.1,0.1

# **三、定义部分超参数**

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

## **4.1 模型测试**

# **五、模型训练**

## **5.1实例化模型**

## **5.2 训练模型需要说明：**

**lr_schedule:** CosineAnnealingDecay

**optimize:** Momentum

**epoch:** 50

**batch_size:** 64

**Loss function:** CrossEntropyLoss

# **六、模型评估**

# **七、可视化测试模型效果**

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

