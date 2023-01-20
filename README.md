# DLP
## 前言：

本repo为TransUnet的代码复现，受电脑性能影响，具体代码运行平台在中科类脑的[BitaHub](https://www.bitahub.com)上，您可以上对应代码到该平台进行测试。

## 仓库文件：

1. 文件夹`TransUNet_Basic`为[官方源码](https://github.com/Beckschen/TransUNet)，但其中有部分错误，需要自行修改，您可以根据文件夹`TransUNetforBitahub`下相关代码进行修改。

   注意原代码运行在`linux`平台，在`windows`平台下，除修改错误外，还需进行相应代码适配。

2. 文件夹`TransUNetforWin`为适配Win平台的代码，运行环境为`python=3.7.12、pytorch=1.12.1`，由于为受电脑性能影响，仅简单测试了`train.py`，用于修改远程端代码，无法确保`test.py`可以正常运行。
3. 文件夹`TransUNetforBitahub`为适配[Bitahub](https://www.bitahub.com)平台的代码。请按照平台要求上传相关文件进行测试。
4. 文件夹`ViT-pytorch-main`为[ViT](https://github.com/google-research/vision_transformer)的某个**pytorch**实现，论文中Encoder结构均源于此。repo位置(https://github.com/jeonsworld/ViT-pytorch)

注意，当移植到您的环境时，还需要修改对应文件的路径等
