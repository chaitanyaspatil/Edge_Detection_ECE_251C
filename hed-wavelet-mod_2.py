import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath, dirname
import pywt
import pywt.data

# Customized import.
from networks import HED
from datasets import BsdsDataset
from utils import Logger, AverageMeter, \
    load_checkpoint, save_checkpoint, load_vgg16_caffe, load_pretrained_caffe


# Parse arguments.
parser = argparse.ArgumentParser(description='HED training.')
# 1. Actions.
parser.add_argument('--test',             default=False,             help='Only test the model.', action='store_true')
# 2. Counts.
parser.add_argument('--train_batch_size', default=1,    type=int,   metavar='N', help='Training batch size.')
parser.add_argument('--test_batch_size',  default=1,    type=int,   metavar='N', help='Test batch size.')
parser.add_argument('--train_iter_size',  default=10,   type=int,   metavar='N', help='Training iteration size.')
parser.add_argument('--max_epoch',        default=40,   type=int,   metavar='N', help='Total epochs.')
parser.add_argument('--print_freq',       default=500,  type=int,   metavar='N', help='Print frequency.')
# 3. Optimizer settings.
parser.add_argument('--lr',               default=1e-6, type=float, metavar='F', help='Initial learning rate.')
parser.add_argument('--lr_stepsize',      default=1e4,  type=int,   metavar='N', help='Learning rate step size.')
# Note: Step size is based on number of iterations, not number of batches.
#   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L498
parser.add_argument('--lr_gamma',         default=0.1,  type=float, metavar='F', help='Learning rate decay (gamma).')
parser.add_argument('--momentum',         default=0.9,  type=float, metavar='F', help='Momentum.')
parser.add_argument('--weight_decay',     default=2e-4, type=float, metavar='F', help='Weight decay.')
# 4. Files and folders.
parser.add_argument('--vgg16_caffe',      default='',                help='Resume VGG-16 Caffe parameters.')
parser.add_argument('--checkpoint',       default='',                help='Resume the checkpoint.')
parser.add_argument('--caffe_model',      default='',                help='Resume HED Caffe model.')
parser.add_argument('--output',           default='./output',        help='Output folder.')
parser.add_argument('--dataset',          default='./data/HED-BSDS', help='HED-BSDS dataset folder.')
# 5. Others.
parser.add_argument('--cpu',              default=False,             help='Enable CPU mode.', action='store_true')
args = parser.parse_args()

# Set device.
device = torch.device('cpu' if args.cpu else 'cuda')


def main():
    ################################################
    # I. Miscellaneous.
    ################################################
    # Create the output directory.
    current_dir = abspath(dirname(__file__))
    output_dir = join(current_dir, args.output)
    if not isdir(output_dir):
        os.makedirs(output_dir)
    
    output_dir_LL = output_dir + '/LL'
    output_dir_LH = output_dir + '/LH'
    output_dir_HL = output_dir + '/HL'
    output_dir_HH = output_dir + '/HH'

    
    if not isdir(output_dir_LL):
        os.makedirs(output_dir_LL)
        
    if not isdir(output_dir_LH):
        os.makedirs(output_dir_LH)
        
    if not isdir(output_dir_HL):
        os.makedirs(output_dir_HL)
        
    if not isdir(output_dir_HH):
        os.makedirs(output_dir_HH)
        


    # Set logger.
    now_str = datetime.now().strftime('%y%m%d-%H%M%S')
    log = Logger(join(output_dir, 'log-{}.txt'.format(now_str)))
    sys.stdout = log  # Overwrite the standard output.

    ################################################
    # II. Datasets.
    ################################################
    # Datasets and dataloaders.

    train_dataset_LL = BsdsDataset(dataset_dir='./data/HED-BSDS/LL', split='train')

    train_dataset_LH = BsdsDataset(dataset_dir='./data/HED-BSDS/LH', split='train')

    train_dataset_HL = BsdsDataset(dataset_dir='./data/HED-BSDS/HL', split='train')

    train_dataset_HH = BsdsDataset(dataset_dir='./data/HED-BSDS/HH', split='train')

    train_loader_LL  = DataLoader(train_dataset_LL, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    train_loader_LH  = DataLoader(train_dataset_LH, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    train_loader_HL  = DataLoader(train_dataset_HL, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    train_loader_HH  = DataLoader(train_dataset_HH, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    
    
    
    test_dataset_LL  = BsdsDataset(dataset_dir='./data/HED-BSDS/LL', split='test')
    test_dataset_LH  = BsdsDataset(dataset_dir='./data/HED-BSDS/LH', split='test')
    test_dataset_HL  = BsdsDataset(dataset_dir='./data/HED-BSDS/HL', split='test')
    test_dataset_HH  = BsdsDataset(dataset_dir='./data/HED-BSDS/HH', split='test')


    test_loader_LL   = DataLoader(test_dataset_LL,  batch_size=args.test_batch_size, num_workers=4, drop_last=False, shuffle=False)
    
    test_loader_LH   = DataLoader(test_dataset_LH,  batch_size=args.test_batch_size, num_workers=4, drop_last=False, shuffle=False)
    
    test_loader_HL   = DataLoader(test_dataset_HL,  batch_size=args.test_batch_size, num_workers=4, drop_last=False, shuffle=False)
    
    test_loader_HH   = DataLoader(test_dataset_HH,  batch_size=args.test_batch_size, num_workers=4, drop_last=False, shuffle=False)

    ################################################
    # III. Network and optimizer.
    ################################################
    # Create the network in GPU.
    net_LL = nn.DataParallel(HED(device))
    net_LL.to(device)
    
    net_LH = nn.DataParallel(HED(device))
    net_LH.to(device)
    
    net_HL = nn.DataParallel(HED(device))
    net_HL.to(device)
    
    net_HH = nn.DataParallel(HED(device))
    net_HH.to(device)

    # Initialize the weights for HED model.
    def weights_init(m):
        """ Weight initialization function. """
        if isinstance(m, nn.Conv2d):
            # Initialize: m.weight.
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                # Constant initialization for fusion layer in HED network.
                torch.nn.init.constant_(m.weight, 0.2)
            else:
                # Zero initialization following official repository.
                # Reference: hed/docs/tutorial/layers.md
                m.weight.data.zero_()
            # Initialize: m.bias.
            if m.bias is not None:
                # Zero initialization.
                m.bias.data.zero_()
                
    net_LL.apply(weights_init)
    net_LH.apply(weights_init)
    net_HL.apply(weights_init)
    net_HH.apply(weights_init)

    # Optimizer settings.
    net_parameters_id_LL = defaultdict(list)
    net_parameters_id_LH = defaultdict(list)
    net_parameters_id_HL = defaultdict(list)
    net_parameters_id_HH = defaultdict(list)
    
    for name, param in net_LL.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id_LL['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id_LL['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id_LL['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias'] :
            print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id_LL['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id_LL['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id_LL['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id_LL['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id_LL['score_final.bias'].append(param)

    
    for name, param in net_LH.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id_LH['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id_LH['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id_LH['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias'] :
            print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id_LH['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id_LH['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id_LH['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id_LH['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id_LH['score_final.bias'].append(param)
        

    for name, param in net_HL.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id_HL['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id_HL['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id_HL['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias'] :
            print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id_HL['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id_HL['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id_HL['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id_HL['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id_HL['score_final.bias'].append(param)
            
    for name, param in net_HH.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id_HH['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id_HH['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id_HH['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias'] :
            print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id_HH['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id_HH['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id_HH['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id_HH['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id_HH['score_final.bias'].append(param)
        
          
            
            
            
            
            
            
            
            
    # Create optimizer.
    opt_LL = torch.optim.SGD([
        {'params': net_parameters_id_LL['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LL['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id_LL['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LL['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id_LL['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LL['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id_LL['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LL['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.

    # Create optimizer.
    opt_LH = torch.optim.SGD([
        {'params': net_parameters_id_LH['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LH['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id_LH['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LH['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id_LH['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LH['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id_LH['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_LH['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.

    
        # Create optimizer.
    opt_HL = torch.optim.SGD([
        {'params': net_parameters_id_HL['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HL['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id_HL['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HL['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id_HL['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HL['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id_HL['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HL['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.
    
    
            # Create optimizer.
    opt_HH = torch.optim.SGD([
        {'params': net_parameters_id_HH['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HH['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id_HH['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HH['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id_HH['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HH['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id_HH['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
        {'params': net_parameters_id_HH['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.

    # Learning rate scheduler.
    lr_schd_LL = lr_scheduler.StepLR(opt_LL, step_size=args.lr_stepsize, gamma=args.lr_gamma)
    lr_schd_LH = lr_scheduler.StepLR(opt_LH, step_size=args.lr_stepsize, gamma=args.lr_gamma)
    lr_schd_HL = lr_scheduler.StepLR(opt_HL, step_size=args.lr_stepsize, gamma=args.lr_gamma)  
    lr_schd_HH = lr_scheduler.StepLR(opt_HH, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    ################################################
    # IV. Pre-trained parameters.
    ################################################
    # Load parameters from pre-trained VGG-16 Caffe model.
    if args.vgg16_caffe:
        load_vgg16_caffe(net_LL, args.vgg16_caffe)
    if args.vgg16_caffe:
        load_vgg16_caffe(net_LH, args.vgg16_caffe)
    if args.vgg16_caffe:
        load_vgg16_caffe(net_HL, args.vgg16_caffe)
    if args.vgg16_caffe:
        load_vgg16_caffe(net_HH, args.vgg16_caffe)
        
        
    # Resume the checkpoint.
    if args.checkpoint:
        load_checkpoint(net_LL, opt_LL, args.checkpoint)  # Omit the returned values.
    if args.checkpoint:
        load_checkpoint(net_LH, opt_LH, args.checkpoint) 
    if args.checkpoint:
        load_checkpoint(net_HL, opt_HL, args.checkpoint)
    if args.checkpoint:
        load_checkpoint(net_HH, opt_HH, args.checkpoint) 
        
        
    # Resume the HED Caffe model.
    if args.caffe_model:
        load_pretrained_caffe(net_LL, args.caffe_model)
    if args.caffe_model:
        load_pretrained_caffe(net_LH, args.caffe_model)
    if args.caffe_model:
        load_pretrained_caffe(net_HL, args.caffe_model)
    if args.caffe_model:
        load_pretrained_caffe(net_HH, args.caffe_model)
        
#     net_LL = net
#     net_LH = net
#     net_HL = net
#     net_HH = net
    
#     opt_LL = opt
#     opt_LH = opt
#     opt_HL = opt
#     opt_HH = opt

    ################################################
    # V. Training / testing.
    ################################################
    if args.test is True:
        
        # Only test.       
        test(test_loader_LL, test_loader_LH, test_loader_HL, test_loader_HH, net_LL, net_LH, net_HL, net_HH, save_dir_LL=join(output_dir_LL, 'test'), save_dir_LH=join(output_dir_LH, 'test'), save_dir_HL=join(output_dir_HL, 'test'), save_dir_HH=join(output_dir_HH, 'test'))
        
        
    else:
        
        # Train.
        train_epoch_losses = []
        
        for epoch in range(args.max_epoch):
            # Initial test.
            if epoch == 0:
                print('Initial test...')
                test(test_loader_LL, test_loader_LH, test_loader_HL, test_loader_HH, net_LL, net_LH, net_HL, net_HH, save_dir_LL=join(output_dir_LL, 'initial-test'), save_dir_LH=join(output_dir_LH, 'initial-test'), save_dir_HL=join(output_dir_HL, 'initial-test'), save_dir_HH=join(output_dir_HH, 'initial-test'))
            # Epoch training and test.
            train_epoch_loss = \
                train(train_loader_LL, train_loader_LH, train_loader_HL, train_loader_HH, net_LL, net_LH, net_HL, net_HH, opt_LL, opt_LH, opt_HL, opt_HH, lr_schd_LL, lr_schd_LH, lr_schd_HL, lr_schd_HH, epoch, save_dir_LL=join(output_dir_LL, 'epoch-{}-train'.format(epoch)), save_dir_LH=join(output_dir_LH, 'epoch-{}-train'.format(epoch)), save_dir_HL=join(output_dir_HL, 'epoch-{}-train'.format(epoch)), save_dir_HH=join(output_dir_HH, 'epoch-{}-train'.format(epoch)))

            
            test(test_loader_LL, test_loader_LH, test_loader_HL, test_loader_HH, net_LL, net_LH, net_HL, net_HH, save_dir_LL=join(output_dir_LL, 'epoch-{}-test'), save_dir_LH=join(output_dir_LH, 'epoch-{}-test'), save_dir_HL=join(output_dir_HL, 'epoch-{}-test'), save_dir_HH=join(output_dir_HH, 'epoch-{}-test'))

            # Write log.
            log.flush()
            
            # Save checkpoint.
            save_checkpoint(state={'net_LL': net_LL.state_dict(), 'opt_LL': opt_LL.state_dict(), 'epoch': epoch}, path=os.path.join(output_dir_LL, 'epoch-{}-checkpoint.pt'.format(epoch)))
            save_checkpoint(state={'net_LH': net_LH.state_dict(), 'opt_LH': opt_LH.state_dict(), 'epoch': epoch}, path=os.path.join(output_dir_LH, 'epoch-{}-checkpoint.pt'.format(epoch)))
            save_checkpoint(state={'net_HL': net_HL.state_dict(), 'opt_HL': opt_HL.state_dict(), 'epoch': epoch}, path=os.path.join(output_dir_HL, 'epoch-{}-checkpoint.pt'.format(epoch)))
            save_checkpoint(state={'net_HH': net_HH.state_dict(), 'opt_HH': opt_HH.state_dict(), 'epoch': epoch}, path=os.path.join(output_dir_HH, 'epoch-{}-checkpoint.pt'.format(epoch)))
            
            # Collect losses.
            train_epoch_losses.append(train_epoch_loss)


            
            
def train(train_loader_LL, train_loader_LH, train_loader_HL, train_loader_HH, net_LL, net_LH, net_HL, net_HH, opt_LL, opt_LH, opt_HL, opt_HH, lr_schd_LL, lr_schd_LH, lr_schd_HL, lr_schd_HH, epoch, save_dir_LL, save_dir_LH, save_dir_HL, save_dir_HH):
    """ Training procedure. """

    dataiter_LL = iter(train_loader_LL)
    dataiter_LH = iter(train_loader_LH)
    dataiter_HL = iter(train_loader_HL)
    dataiter_HH = iter(train_loader_HH)
    
    # Create the directory.
    if not isdir(save_dir_LL):
        os.makedirs(save_dir_LL)
    # Create the directory.
    if not isdir(save_dir_LH):
        os.makedirs(save_dir_LH)
    # Create the directory.
    if not isdir(save_dir_HL):
        os.makedirs(save_dir_HL)
    # Create the directory.
    if not isdir(save_dir_HH):
        os.makedirs(save_dir_HH)
        
        
    # Switch to train mode and clear the gradient.
    net_LL.train()
    opt_LL.zero_grad()
    
    net_LH.train()
    opt_LH.zero_grad()
    
    net_HL.train()
    opt_HL.zero_grad()
    
    net_HH.train()
    opt_HH.zero_grad()
    
    
    # Initialize meter and list.
    batch_loss_meter = AverageMeter()
    # Note: The counter is used here to record number of batches in current training iteration has been processed.
    #       It aims to have large training iteration number even if GPU memory is not enough. However, such trick
    #       can be used because batch normalization is not used in the network architecture.
    
    
    counter = 0
    
    for i in tqdm(range(28800)):
        
        images_LL, edges = dataiter_LL.next()
        images_LH, edges = dataiter_LH.next()
        images_HL, edges = dataiter_HL.next()
        images_HH, edges = dataiter_HH.next()
        
        batch_index = i
        
        # Adjust learning rate and modify counter following Caffe's way.
        if counter == 0:
            lr_schd_LL.step()  # Step at the beginning of the iteration.
            lr_schd_LH.step()
            lr_schd_HL.step()
            lr_schd_HH.step()
        counter += 1
        # Get images and edges from current batch.
        
        images_LL, edges = images_LL.to(device), edges.to(device)
        images_LH, edges = images_LH.to(device), edges.to(device)
        images_HL, edges = images_HL.to(device), edges.to(device)
        images_HH, edges = images_HH.to(device), edges.to(device)


        # Generate predictions.
        preds_list_LL = net_LL(images_LL)
        preds_list_LH = net_LH(images_LH)
        preds_list_HL = net_HL(images_HL)
        preds_list_HH = net_LL(images_HH)
        # Calculate the loss of current batch (sum of all scales and fused).
        # Note: Here we mimic the "iteration" in official repository: iter_size batches will be considered together
        #       to perform one gradient update. To achieve the goal, we calculate the equivalent iteration loss
        #       eqv_iter_loss of current batch and generate the gradient. Then, instead of updating the weights,
        #       we continue to calculate eqv_iter_loss and add the newly generated gradient to current gradient.
        #       After iter_size batches, we will update the weights using the accumulated gradients and then zero
        #       the gradients.
        # Reference:
        #   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L230
        #   https://www.zhihu.com/question/37270367
        
        # major change here
        # use IDWT to generate image
        # call that preds and pass it to loss function
        
        preds_list = []
        for i in range(6):
            LL = preds_list_LL[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            LH = preds_list_LH[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            HL = preds_list_HL[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            HH = preds_list_HH[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            full_image = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
            full_image = torch.reshape(torch.from_numpy(full_image), (1,1,preds_list_LL[i].shape[2]*2, preds_list_LL[i].shape[3]*2))
            full_image = full_image.to(device)
            preds_list.append(full_image)
        
        batch_loss = sum([weighted_cross_entropy_loss(preds, edges) for preds in preds_list])
        
        batch_loss.requires_grad = True
        
        eqv_iter_loss = batch_loss / args.train_iter_size

                
        
        
        # Generate the gradient and accumulate (using equivalent average loss).
        eqv_iter_loss.backward()
        
        
        
        if counter == args.train_iter_size:
            opt_LL.step()
            opt_LH.step()
            opt_HL.step()
            opt_HH.step()
            opt_LL.zero_grad()
            opt_LH.zero_grad()
            opt_HL.zero_grad()
            opt_HH.zero_grad()
            counter = 0  # Reset the counter.
            
        # Record loss.
        batch_loss_meter.update(batch_loss.item())
        
        
        # Log and save intermediate images.
        if batch_index % args.print_freq == args.print_freq - 1:
            # Log.
            print(('Training epoch:{}/{}, batch:{}/{} current iteration:{}, ' +
                   'current batch batch_loss:{}, epoch average batch_loss:{}, learning rate list:{}.').format(
                   epoch, args.max_epoch, batch_index, len(train_loader_LL), lr_schd_LL.last_epoch,
                   batch_loss_meter.val, batch_loss_meter.avg, lr_schd_LL.get_lr()))
            
            # Generate intermediate images.
            preds_list_and_edges = preds_list + [edges]
            _, _, h, w = preds_list_and_edges[0].shape
            interm_images = torch.zeros((len(preds_list_and_edges), 1, h, w))
            print(interm_images[i, 0, :, :].shape)
            print(preds_list_and_edges[i][0, 0, :, :].shape)
            for i in range(len(preds_list_and_edges)):
                # Only fetch the first image in the batch.
                
                interm_images[i, 0, :, :] = preds_list_and_edges[i][0, 0, :, :]
                
                
                
            # Save the images.
            torchvision.utils.save_image(interm_images, join(save_dir_LL, 'batch-{}-1st-image.png'.format(batch_index)))
    # Return the epoch average batch_loss.
    return batch_loss_meter.avg

def test(test_loader_LL, test_loader_LH, test_loader_HL, test_loader_HH,  net_LL, net_LH, net_HL, net_HH, save_dir_LL, save_dir_LH, save_dir_HL, save_dir_HH):
    """ Test procedure. """
    


    dataiter_test_LL = iter(test_loader_LL)
    dataiter_test_LH = iter(test_loader_LH)
    dataiter_test_HL = iter(test_loader_HL)
    dataiter_test_HH = iter(test_loader_HH)


    # Create the directories.
    if not isdir(save_dir_LL):
        os.makedirs(save_dir_LL)
    if not isdir(save_dir_LH):
        os.makedirs(save_dir_LH)
    if not isdir(save_dir_HL):
        os.makedirs(save_dir_HL)
    if not isdir(save_dir_HH):
        os.makedirs(save_dir_HH)
        
    save_png_dir_LL = join(save_dir_LL, 'png')
    save_png_dir_LH = join(save_dir_LH, 'png')
    save_png_dir_HL = join(save_dir_HL, 'png')
    save_png_dir_HH = join(save_dir_HH, 'png')

    
    
    if not isdir(save_png_dir_LL):
        os.makedirs(save_png_dir_LL)
    if not isdir(save_png_dir_LH):
        os.makedirs(save_png_dir_LH)
    if not isdir(save_png_dir_HL):
        os.makedirs(save_png_dir_HL)
    if not isdir(save_png_dir_HH):
        os.makedirs(save_png_dir_HH)
        
    
    save_mat_dir_LL = join(save_dir_LL, 'mat')
    save_mat_dir_LH = join(save_dir_LH, 'mat')
    save_mat_dir_HL = join(save_dir_HL, 'mat')
    save_mat_dir_HH = join(save_dir_HH, 'mat')

    
    
    if not isdir(save_mat_dir_LL):
        os.makedirs(save_mat_dir_LL)
    if not isdir(save_mat_dir_LH):
        os.makedirs(save_mat_dir_LH)
    if not isdir(save_mat_dir_HL):
        os.makedirs(save_mat_dir_HL)
    if not isdir(save_mat_dir_HH):
        os.makedirs(save_mat_dir_HH)
        
        
    # Switch to evaluation mode.
    net_LL.eval()
    net_LH.eval()
    net_HL.eval()
    net_HH.eval()
    # Generate predictions and save.
    assert args.test_batch_size == 1  # Currently only support test batch size 1.
          

        
    for i in tqdm(range(200)): # hard-coding temporarily
        images_LL = dataiter_test_LL.next()
        images_LH = dataiter_test_LH.next()
        images_HL = dataiter_test_HL.next()
        images_HH = dataiter_test_HH.next()
        
        batch_index = i
        
        images_LL = images_LL.cuda()
        images_LH = images_LH.cuda()
        images_HL = images_HL.cuda()
        images_HH = images_HH.cuda()
        
        _, _, h, w = images_LL.shape
        
        preds_list_LL = net_LL(images_LL)
        preds_list_LH = net_LH(images_LH)
        preds_list_HL = net_HL(images_HL)
        preds_list_HH = net_HH(images_HH)
        
        preds_list = []
        for i in range(6):
            LL = preds_list_LL[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            LH = preds_list_LH[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            HL = preds_list_HL[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            HH = preds_list_HH[i].cpu().detach().numpy().reshape(preds_list_LL[i].shape[2], preds_list_LL[i].shape[3])
            full_image = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
            full_image = torch.reshape(torch.from_numpy(full_image), (1,1,preds_list_LL[i].shape[2]*2, preds_list_LL[i].shape[3]*2))
            full_image = full_image.to(device)
            preds_list.append(full_image)
        
        # generate preds_list (IDWT)
        
        fuse       = preds_list[-1].detach().cpu().numpy()[0, 0]  # Shape: [h, w].
        name       = test_loader_LL.dataset.images_name[batch_index]
        
        sio.savemat(join(save_mat_dir_LL, '{}.mat'.format(name)), {'result': fuse})
        sio.savemat(join(save_mat_dir_LH, '{}.mat'.format(name)), {'result': fuse})
        sio.savemat(join(save_mat_dir_HL, '{}.mat'.format(name)), {'result': fuse})
        sio.savemat(join(save_mat_dir_HH, '{}.mat'.format(name)), {'result': fuse})

        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_png_dir_LL, '{}.png'.format(name)))
        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_png_dir_LH, '{}.png'.format(name)))
        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_png_dir_HL, '{}.png'.format(name)))
        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_png_dir_HH, '{}.png'.format(name)))

        # print('Test batch {}/{}.'.format(batch_index + 1, len(test_loader)))


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].
    weight = torch.zeros_like(mask)
    weight[edges > 0.5]  = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    # is there a size mismatch (due to DWT)?
    if(preds.shape!=edges.shape):
        #is there a rows mismatch?
        if(preds.shape[2]!=edges.shape[2] and preds.shape[3]!=edges.shape[3]):
            preds = preds[:,:,:preds.shape[2]-1,:]
            preds = preds[:,:,:,:preds.shape[3]-1]
        elif(preds.shape[2]!=edges.shape[2]):
            preds = preds[:,:,:preds.shape[2]-1,:]
        else:
            preds = preds[:,:,:,:preds.shape[3]-1]
    losses = torch.nn.functional.binary_cross_entropy(preds.float(), edges.float(), weight=weight, reduction='none')

    loss   = torch.sum(losses) / b
    return loss


if __name__ == '__main__':
    main()
