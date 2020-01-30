import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import wdsr
from option import args

with torch.cuda.device(6):
	net = wdsr.MODEL(args)
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (3, 124, 118), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

