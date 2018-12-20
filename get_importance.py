import argparse
import shutil
import tqdm

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable, grad

from collections import OrderedDict
from utee import misc, quant, selector
from utils.utils import *
from utils.utils import accuracy
from utils.prune_utils import *
from utils.quantize_utils import *
from utils.hessian_utils import *

import time

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--mode', default='normal', help='mode of weight importance: normal/gradient/hessian')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for computing importance')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature for model calibration')
parser.add_argument('--hessian_ssr', type=float, default=0.25, help='subsample rate for computing hessian')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--save_root', default='weight_importances/', help='folder for retrained models')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
args = parser.parse_args()

args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
'''
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")
'''

assert torch.cuda.is_available(), 'no cuda'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

valid_layer_types = [nn.modules.conv.Conv2d, nn.modules.linear.Linear]

def get_all_one_importance(model, valid_ind, is_imagenet):
    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        m = m_list[ix]
        importances[ix] = torch.ones_like(m.weight) 
    return importances

def get_gradient_importance(model, ds_for_importance, valid_ind, is_imagenet):
    criterion = nn.CrossEntropyLoss()
    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        importances[ix] = 0.
    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    for i, (input, target) in enumerate(tqdm.tqdm(ds_for_importance, total=len(ds_for_importance))):
        optimizer.zero_grad()
        if is_imagenet:
            input = torch.from_numpy(input)
            target = torch.from_numpy(target)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        if 'inception' in args.type:
            output, aux_output = model(input_var)
            loss = criterion(output / args.temperature, target_var) + criterion(aux_output / args.temperature, target_var)
        else:
            output = model(input_var)
            loss = criterion(output / args.temperature, target_var)
        
        loss.backward()

        for ix in valid_ind:
            m = m_list[ix]
            importances[ix] += m.weight.grad.data**2

    for ix in valid_ind:
        importances[ix] = importances[ix] / importances[ix].mean()
        
    return importances

def get_hessian_importance(model, ds_for_importance, valid_ind, is_imagenet):
    criterion = nn.CrossEntropyLoss()
    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        importances[ix] = 0.
    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    for i, (input, target) in enumerate(tqdm.tqdm(ds_for_importance, total=len(ds_for_importance))):
        optimizer.zero_grad()
        if is_imagenet:
            input = torch.from_numpy(input)
            target = torch.from_numpy(target)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        if 'inception' in args.type:
            output, aux_output = model(input_var)
            loss = (criterion(output / args.temperature, target_var) + criterion(aux_output / args.temperature, target_var))**2
        else:
            output = model(input_var)
            loss = criterion(output / args.temperature, target_var)**2

       # dhs = diagonal_hessian_multi(loss, output, model.parameters()) 
      
        for ii in range(len(valid_ind)):
            ix = valid_ind[ii]
            m = m_list[ix]
            dhs = diagonal_hessian_multi(loss, output, [m.weight])
            importances[ix] += dhs[0] 
            
    for ix in valid_ind:
        importances[ix] = importances[ix] / importances[ix].mean()
        
    return importances

def main():
    # load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)
    args.ngpu = args.ngpu if is_imagenet else 1
    training_size = 60000 if args.type=='mnist' else 50000

    # get valid layers
    valid_ind = []
    layer_type_list = []
    for i, layer in enumerate(model_raw.modules()):
        if type(layer) in valid_layer_types:
            valid_ind.append(i)
            layer_type_list.append(type(layer))

    #get training dataset and validation dataset and dataset for computing importance
    if args.mode == 'hessian' and args.hessian_ssr < 1.0:
        indices = np.random.choice(training_size, int(args.hessian_ssr*training_size/args.batch_size)*args.batch_size, replace=True)
        ds_for_importance = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, subsample=True, indices=indices, input_size=args.input_size)
    else:
        ds_for_importance = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, input_size=args.input_size)


    #get weight importance
    if args.mode == 'normal':
        weight_importance = get_all_one_importance(model_raw,valid_ind, is_imagenet)
    elif args.mode == 'hessian':
        weight_importance = get_hessian_importance(model_raw, ds_for_importance, valid_ind, is_imagenet)
    elif args.mode == 'gradient':
        weight_importance = get_gradient_importance(model_raw, ds_for_importance, valid_ind, is_imagenet)

    #write to file
    filename = args.type+"_"+args.mode
    if args.temperature > 1.0:
        filename += "_t="+str(int(args.temperature))
    filename += ".pth"
    pathname = args.save_root+args.type
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    filepath = os.path.join(pathname, filename)
    with open(filepath, "wb") as f:
        torch.save(weight_importance, f)

if __name__ == '__main__':
    main()

