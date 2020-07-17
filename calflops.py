from __future__ import print_function
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import argparse
import os
import sys
# from resnet_1d import ResNet50_1d
# from resnet_1d_lite import ResNet50_1d_shrink
from thop import profile
import yaml
import wdsr_b
from option2 import parser
from wdsr_b import *
#from args import *
import math
# parser = argparse.ArgumentParser(description='Load Models')
# parser.add_argument('--slice_size', type=int, default=198, help='input size')
# parser.add_argument('--devices', type=int, default=500, help='number of classes')
import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.utils.resnet_utils import myResnet
from captioning.utils import resnet

#with torch.cuda.device(6):
	# args = parser.parse_args()
	# shrink = 0.547
	# base_path = os.getcwd()
	# model_folder = ['1C_wifi_raw', '1C_adsb', '1C_mixture','1C_wifi_eq','1A_wifi_eq']
	# slice_sizes = [512,512,512,198,198]
	# devices = [50,50,50,50,500]
	# for folder, slice_size, device in zip(model_folder,slice_sizes,devices):
	# 	print(folder)
	# model = ResNet50_1d(args.slice_size,args.devices)
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()
cnn_model = 'resnet101'
my_resnet = getattr(resnet, cnn_model)()
my_resnet.load_state_dict(torch.load('data/imagenet_weights/'+ cnn_model+'.pth'))
model = myResnet(my_resnet)
input = torch.randn(3, 640, 427)

model.train(False)
model.eval()
macs, params = profile(model, inputs=(input, ))
# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs/pow(10,9))) # GMACs
print('{:<30}  {:<8}'.format('Number of parameters: ', params/pow(10,6))) # M



