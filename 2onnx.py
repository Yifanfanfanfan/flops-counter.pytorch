import os, time, shutil, argparse
from functools import partial
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
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
from wdsr_b import *
from args import *

def main():

    use_gpu = torch.cuda.is_available()
    # Create model  
    # models.resnet18(num_classes=365)
    # model = ColorNet()
    args = get_args()
    model = MODEL(args)
    # state_dict = torch.load("./checkpoint/checkpoint6/model_epoch133_step1.pth")
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     k = k.replace('module.', '')
    #     new_state_dict[k] = v

    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(new_state_dict)
    checkpoint = torch.load("/home/zzgyf/github_yifan/ImageCaptioning.pytorch/models/model-best.pth")
    model.load_state_dict(checkpoint["model"].state_dict())

    # print(model)
    #input_shape = (1, 256, 256)
    cocotest_bu_fc_size = (10, 2048)
    cocotest_bu_att_size = (10, 0, 0)
    labels_size = (10, 5, 18)
    masks_size = (10, 5, 18)
    model_onnx_path = "./wdsr_b.onnx"
    model.train(False)

    # Export the model to an ONNX file
    #dummy_input = Variable(torch.randn(1, *input_shape))
    dummy_cocotest_bu_fc = Variable(torch.randn(10, 2048))
    dummy_cocotest_bu_att = Variable(torch.randn(10, 0, 0))
    dummy_labels = Variable(torch.randint(5200, (10, 5, 18)))
    dummy_masks = Variable(torch.randint(1, (10, 5, 18)))
    #output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    output = torch_onnx.export(model, dummy_cocotest_bu_fc, dummy_cocotest_bu_att, dummy_labels, dummy_masks, model_onnx_path, verbose=False)
    print("Export of torch_model.onnx complete!")


def check():

    # Load the ONNX model
    model = onnx.load("wdsr_b.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

if __name__ == '__main__':
    main()
    check()
