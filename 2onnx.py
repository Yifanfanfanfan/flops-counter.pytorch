import os, sys, time, shutil, argparse
from functools import partial
import pickle
sys.path.append('../')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
#import torchvision.models as models
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from collections import OrderedDict
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx as torch_onnx
import onnx

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage import io

# import prune_util
# from prune_util import GradualWarmupScheduler
# from prune_util import CrossEntropyLossMaybeSmooth
# from prune_util import mixup_data, mixup_criterion

# from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
# from model import ColorNet
#from wdsr_b import *
#from args import *
import captioning.utils.opts as opts
import captioning.models as models
import captioning.utils.misc as utils
def main():

    use_gpu = torch.cuda.is_available()
    # Create model  
    # models.resnet18(num_classes=365)
    # model = ColorNet()
    #args = get_args()
    #model = MODEL(args)
    # state_dict = torch.load("./checkpoint/checkpoint6/model_epoch133_step1.pth")
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     k = k.replace('module.', '')
    #     new_state_dict[k] = v

    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(new_state_dict)
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
    parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
    parser.add_argument('--force', type=int, default=0,
                help='force to evaluate no matter if there are results available')
    opts.add_eval_options(parser)
    opts.add_diversity_opts(parser)
    opt = parser.parse_args()
    opt.caption_model = 'newfc'
    opt.infos_path = '/home/zzgyf/github_yifan/ImageCaptioning.pytorch/models/infos_fc_nsc-best.pkl'
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    vocab = infos['vocab'] # ix -> word mapping

    opt.vocab = vocab
    model = models.setup(opt)

    checkpoint = torch.load("/home/zzgyf/github_yifan/ImageCaptioning.pytorch/models/model-best.pth")
    model.load_state_dict(checkpoint)

    # print(model)
    #input_shape = (1, 256, 256)
    cocotest_bu_fc_size = (10, 2048)
    cocotest_bu_att_size = (10, 0, 0)
    labels_size = (10, 5, 18)
    masks_size = (10, 5, 18)
    model_onnx_path = "./image_captioning.onnx"
    model.train(False)

    # Export the model to an ONNX file
    # dummy_input = Variable(torch.randn(1, *input_shape))
    dummy_input = Variable(torch.randn(10, 2048), torch.randn(10, 0, 0), torch.randint(5200, (10, 5, 18)) torch.randint(1, (10, 5, 18)))
    # dummy_cocotest_bu_fc = Variable(torch.randn(10, 2048))
    # dummy_cocotest_bu_att = Variable(torch.randn(10, 0, 0))
    # dummy_labels = Variable(torch.randint(5200, (10, 5, 18)))
    # dummy_masks = Variable(torch.randint(1, (10, 5, 18)))
    #output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    print("Export of torch_model.onnx complete!")


def check():

    # Load the ONNX model
    model = onnx.load("image_captioning.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

if __name__ == '__main__':
    main()
    check()
