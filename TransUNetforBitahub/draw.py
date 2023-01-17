import torch
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs')

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 9
config_vit.n_skip = 3
config_vit.patches.grid = (14, 14)

images = torch.rand(3,1,224,224)
net = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)
writer.add_graph(net, images)
writer.flush()
writer.close()