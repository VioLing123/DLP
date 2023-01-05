### Bitahub训练方式及注意事项：

#### 1.环境：

`python==3.8	pytorch==1.12`

注意需要安装运行所需的库（可见requirements.txt）：

torch
​torchvision
​numpy
​tqdm
​scipy（上述库bitahub镜像已经存在（可能），可以删去）
​tensorboard
​tensorboardX
​ml-collections
​medpy
​SimpleITK
​h5py

##### 安装方式：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

#### 2.程序修改：

注意在bitahub上（ `linux`  环境）， `DataLoader` 函数中 `num_worker` 可以设置大于 `0` 。

**在 `trainer.py` 中其设置为 8** ，**在 `test.py` 中设置为 1** 。训练时间可以有明显提速（4小时内）。

在 `windows` 环境中，**其均需设置为0**

**应当运行`train.py`和`test_data.py`**

已经在模型文件中保存某次训练的`.pth`文件，可以直接运行`test_data.py`来测试

#### 3.运行代码：

##### 1.训练模型：
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && python train.py`

##### 2.测试模型 
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && python test_data.py` [注：该方式采用某次训练得到的模型参数文件]

##### 3.训练加测试：
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && python train.py && python test.py` [注：该方式采用训练出的模型参数计算]

