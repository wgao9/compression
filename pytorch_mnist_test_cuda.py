import torch
from torch.autograd import Variable
from utee import selector

import numpy as np

# from pytorch_hessian_test import hessian_diagonal, hessian, total_derivative

# from tensordot import tensordot_pytorch, contraction_pytorch

from utils.hessian_utils import *

from itertools import product

import datetime

batch_size = 200

# NOTE: Even with cuda=False the selector still tried to map the variables to GPUs.  I had to change code in mnist/model.py to force mapping to CPU.
cuda = torch.device('cuda')

model_raw, ds_fetcher, is_imagenet = selector.select('mnist', cuda=True)

# ps = list(model_raw.parameters())

layer1 = model_raw.model[:3]
layer2 = model_raw.model[3:6]
layer3 = model_raw.model[6]

ds_val = ds_fetcher(batch_size=batch_size, train=False, val=True)
for idx, (data, target) in enumerate(ds_val):
    # data =  Variable(torch.FloatTensor(data))
    # output = model_raw(data)

    data = data.to(device=cuda)
    target = target.to(device=cuda)

    flattened = data.reshape((-1, 28*28))
    z1 = layer1(flattened)
    z2 = layer2(z1)
    z3 = layer3(z2)

    output = z3

    loss = torch.nn.CrossEntropyLoss()

    f = loss(output, target) ** 2

    print('start: {}'.format(datetime.datetime.now()))

    dhs = diagonal_hessian_multi(f, output, model_raw.parameters())  

    print('end:   {}'.format(datetime.datetime.now()))
    print('memory: {}, {}'.format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))

    torch.cuda.empty_cache()

    if idx > 5:
        break

# # the kernels
# w0 = ps[0]
# w1 = ps[2]
# w2 = ps[4]

# print(w0.shape, w1.shape, w2.shape)
# print(w0.device, w1.device, w2.device)


# on one V100: (also, not really faster on 4 K80s)
# In [9]: %time fw2w2_diagonal = diagonal_hessian(f, [z3], w2)
# CPU times: user 340 ms, sys: 52 ms, total: 392 ms
# Wall time: 385 ms
# In [10]: %time fw1w1_diagonal = diagonal_hessian(f, [z3, z2], w1)
# CPU times: user 5.96 s, sys: 936 ms, total: 6.89 s
# Wall time: 6.81 s
# In [11]: %time fw0w0_diagonal = diagonal_hessian(f, [z3, z2, z1], w0)
# CPU times: user 16 s, sys: 2.68 s, total: 18.7 s
# Wall time: 18.5 s

# not really faster!
# In [12]: %time fw0w0_diagonal = diagonal_hessian(f, [z3], w0)
# CPU times: user 15.1 s, sys: 2.29 s, total: 17.4 s
# Wall time: 17.4 s


# fw2w2_diagonal = diagonal_hessian(f, [z3], w2)

# print(fw2w2_diagonal.shape, np.max(fw2w2_diagonal.cpu().data.numpy()), np.min(fw2w2_diagonal.cpu().data.numpy()))


# # on one V100 with the slow implementation
# # RuntimeError: CUDA error: out of memory
# fw1w1_diagonal = diagonal_hessian(f, [z3, z2], w1)

# print(fw1w1_diagonal.shape, np.max(fw1w1_diagonal.cpu().data.numpy()), np.min(fw1w1_diagonal.cpu().data.numpy()))



# fw0w0_diagonal = diagonal_hessian(f, [z3, z2, z1], w0)

# print(fw0w0_diagonal.shape, np.max(fw0w0_diagonal.cpu().data.numpy()), np.min(fw0w0_diagonal.cpu().data.numpy()))

