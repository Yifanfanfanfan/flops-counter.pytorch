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
from captioning.data.dataloader import *
from captioning.utils.resnet_utils import myResnet
from captioning.utils import resnet

import onnxruntime

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
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
    opts.add_eval_options(parser)
    opts.add_diversity_opts(parser)
    opt = parser.parse_args()
    cnn_model = 'resnet101'
    my_resnet = getattr(resnet, cnn_model)()
    
    my_resnet.load_state_dict(torch.load('/home/zzgyf/github_yifan/ImageCaptioning.pytorch/data/imagenet_weights/'+ cnn_model+'.pth'))

    model = myResnet(my_resnet)

    if isinstance(model, torch.nn.DataParallel):         
        model = model.module

    #checkpoint = torch.load("/home/zzgyf/github_yifan/ImageCaptioning.pytorch/models/model-best.pth")
    #model.load_state_dict(checkpoint)


    # cocotest_bu_fc_size = (10, 2048)
    # cocotest_bu_att_size = (10, 0, 0)
    # labels_size = (10, 5, 18)
    # masks_size = (10, 5, 18)
    model_onnx_path = "./resnet101.onnx"
    input_shape = (3, 640, 480)
    model.train(False)
    model.eval()

    # Export the model to an ONNX file
    # dummy_input = Variable(torch.randn(1, *input_shape))
    dummy_input = Variable(torch.randn(*input_shape))
    #output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=True)
    print("Export of torch_model.onnx complete!")


def check():
    
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
    opts.add_eval_options(parser)
    opts.add_diversity_opts(parser)
    opt = parser.parse_args()
    cnn_model = 'resnet101'
    my_resnet = getattr(resnet, cnn_model)()
    my_resnet.load_state_dict(torch.load('/home/zzgyf/github_yifan/ImageCaptioning.pytorch/data/imagenet_weights/'+ cnn_model+'.pth'))
    model = myResnet(my_resnet)
    #checkpoint = torch.load("/home/zzgyf/github_yifan/ImageCaptioning.pytorch/models/model-best.pth")
    #model.load_state_dict(checkpoint)

    # torch.nn.utils.remove_weight_norm(model.head[0])
    # for i in range(2):
    #     for j in [0,2,3]:
    #         torch.nn.utils.remove_weight_norm(model.body[i].body[j])
    # torch.nn.utils.remove_weight_norm(model.tail[0])
    # torch.nn.utils.remove_weight_norm(model.skip[0])

    model.eval()
    ort_session = onnxruntime.InferenceSession("resnet101.onnx")

 
    x = Variable(torch.randn(3, 640, 480))
    #x = torch.randn(1, 3, 392, 392, requires_grad=False)
    #torch_out = model(x)
    # # Load the ONNX model
    # model = onnx.load("wdsr_b.onnx")

    # # Check that the IR is well formed
    # onnx.checker.check_model(model)

    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

if __name__ == '__main__':
    main()
    check()
