# Bitahub训练方式及注意事项：

## 1.环境：

`python==3.8	pytorch==1.12`

注意需要安装运行所需的库（可见requirements.txt）：

```txt
torch
torchvision
numpy
tqdm
scipy（上述库bitahub镜像已经存在（可能），可以删去）
tensorboard
tensorboardX
ml-collections
medpy
SimpleITK
h5py
```

**安装方式：**

**`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`**

## 2.程序修改：

注意在bitahub上（ `linux`  环境）， `DataLoader` 函数中 `num_worker` 可以设置大于 `0` 。

**在 `trainer.py` 中其设置为 8** ，**在 `test.py` 中设置为 1** 。训练时间可以有明显提速（4小时内）。

在 `windows` 环境中，**其均需设置为0**

**应当运行`train.py`和`test_data.py`**

已经在模型文件中保存某次训练的`.pth`文件，可以直接运行`test_data.py`来测试

## 3.运行代码：

### 1.训练模型：

`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && python train.py`

### 2.测试模型 

`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && python test_data.py` [注：该方式采用某次训练得到的模型参数文件]

### 3.训练加测试：

`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && python train.py && python test.py` [注：该方式采用训练出的模型参数计算]





# TransUNet

This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory, and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
