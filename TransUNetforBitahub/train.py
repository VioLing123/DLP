import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling_add import VisionTransformer_None as ViT_seg_None
from networks.vit_seg_modeling_add import VisionTransformer_1SkipinPaper as ViT_seg_Skip1
from networks.vit_seg_modeling_new import VisionTransformer_New as ViT_seg_New
from networks.vit_seg_modeling_new1 import VisionTransformer_New1 as ViT_seg_New1
from networks.vit_seg_modeling_new2 import VisionTransformer_New2 as ViT_seg_New2

from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/VioLin/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')#默认为24，本电脑只能跑3
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')#使用GPU个数
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')#使用特定的卷积方式
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')#学习率
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')#输入图像的大小
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect in CNN, default is num')
parser.add_argument('--n_skip_trans', type=int,
                    default=2, help='using number of skip-connect in trans, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')#使用预训练模型
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--decoder', type=str, default='CUP', help='whether use CUP decoder')
parser.add_argument('--con_form', type=str, default='Cat', help='the way that trans_skip connect with x')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset#数据集的名称
    #dataset实际的配置位置，
    dataset_config = {
        'Synapse': {
            'root_path': '../data/VioLin/Synapse/train_npz',#数据集的位置，从父目录开始（project_TransUNet）
            'list_dir': './lists/lists_Synapse',#list位置，从本目录开始
            'num_classes': 9,
        },
    }
    #将dataset_config的相关配置写入
    #之前通过args传入的数据实际无效
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    #调整学习率
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.is_pretrain = True

    args.exp = 'TU_' + dataset_name + str(args.img_size)#--> args.exp = "TU_Synapse224"
    
    snapshot_path = "../output/{}/{}".format(args.exp, 'TU')#保存到output文件夹里
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    #snapshot_path(训练快照保存位置)
    #位置：../output/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs3_224

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    
    '''
    设置pre_trained model
    依据对应的训练网络，设置对应的预训练模型
    '''
    # （默认为R50+ViT-B_16,注意vit_name为R50-ViT-B_16,要与CONFIGS_ViT_seg中字典key相同）
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip #TransUnet中skip的数量
    config_vit.n_skip_trans = args.n_skip_trans #New中Trans_skip的数量
    config_vit.connect_form = args.con_form #New中Trans_skip与x的连接方式
    if args.vit_name.find('R50') != -1 and (args.decoder == 'CUP' or args.decoder == 'NEW' or args.decoder == 'Skip1' or args.decoder == 'NEW1' or args.decoder == 'NEW2'): #没找到就返回 -1
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))#使用R50时，将对grid进行修改

    #创建模型网络
    if args.decoder == 'CUP':#使用论文中的CUP结构
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.decoder == 'NEW':#使用自己改进的skip-connection模型
        net = ViT_seg_New(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.decoder == 'NEW1':#结合Unet将decoder中的upsampling改为conv的形式。其中没有NEW中的trans_skip
        net = ViT_seg_New1(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.decoder == 'NEW2':#结合Unet将decoder中的upsampling改为conv的形式。其中没有NEW中的trans_skip
        net = ViT_seg_New2(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.decoder == 'Skip1':#复现论文中1skip的情况
        net = ViT_seg_Skip1(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()        
    else:#复现论文中decoder使用None的结构
        net = ViT_seg_None(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    net.load_from(weights=np.load(config_vit.pretrained_path))#提取并设置权重值（默认路径为../model/VioLin/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz）

    trainer = {'Synapse': trainer_synapse,}#创建训练模型字典，可以通过key值确定所用的训练模型，目前只有一个
    trainer[dataset_name](args, net, snapshot_path)#加载模型