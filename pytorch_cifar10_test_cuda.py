import torch
from torch.autograd import Variable
from utee import selector

import numpy as np

# from pytorch_hessian_test import hessian_diagonal, hessian, total_derivative

# from tensordot import tensordot_pytorch, contraction_pytorch

from utils.hessian_utils import *

from itertools import product

import datetime

# NOTE: Even with cuda=False the selector still tried to map the variables to GPUs.  I had to change code in mnist/model.py to force mapping to CPU.
cuda = torch.device('cuda')
model_raw, ds_fetcher, is_imagenet = selector.select('cifar10', cuda=True)

# ps = list(model_raw.parameters())

# batch_size = 13 caused CUDA OOM on a K80

batch_size = 15

ds_val = ds_fetcher(batch_size=batch_size, train=False, val=True)

for idx, (data, target) in enumerate(ds_val):
    print(idx)

    z = data.to(device=cuda)
    target = target.to(device=cuda)

    for layer in model_raw.features:
        z = layer(z)

    z = z.reshape([batch_size, 1024])

    output = model_raw.classifier(z)
    loss = torch.nn.CrossEntropyLoss()
    f = loss(output, target) ** 2

    # break

    print('start: {}'.format(datetime.datetime.now()))

    dhs = diagonal_hessian_multi(f, output, model_raw.parameters())  

    print('end:   {}'.format(datetime.datetime.now()))
    print('memory: {}, {}'.format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))

    torch.cuda.empty_cache()

    if idx > 10:
        break



