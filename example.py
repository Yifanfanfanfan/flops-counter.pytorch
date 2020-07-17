#import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
# import wdsr_b
from option2 import parser
#from wdsr_b import *
#from image_captioning import *
import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.utils.resnet_utils import myResnet
from captioning.utils import resnet
import argparse

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

#with torch.cuda.device():
#args = get_args()
#net = WDSR_B(args)
# net = models.densenet161()
'''
################################
# Build dataloader
################################
loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length
##########################
# Build model
##########################
opt.vocab = loader.get_vocab()
model = models.setup(opt).cuda()
del opt.vocab
'''
cnn_model = 'resnet101'
my_resnet = getattr(resnet, cnn_model)()
my_resnet.load_state_dict(torch.load('data/imagenet_weights/'+ cnn_model+'.pth'))
net = myResnet(my_resnet)
flops, params = get_model_complexity_info(net, (3, 640, 427), as_strings=True, print_per_layer_stat=True)
# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# 144P(256×144) 240p(426×240) 360P(640×360) 480P(854×480)