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

#General arguments
parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--importance_root', default='weight_importances/', help='folder to save weight importances')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
#Quantization arguments
parser.add_argument('--mode', default='normal', type=str, help='quantization mode: none/normal/gradient/hessian')
parser.add_argument('--loss', default='cross_entropy', type=str, help='loss funciton')
parser.add_argument('--bits', default=None, type=str, help='number of bits of quantization')
parser.add_argument('--fix_bit', default=None, type=float, help='fix bit for every layer')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for estimating gradient')
parser.add_argument('--ha', default=0.0, type=float, help='multiplier of hessian')
parser.add_argument('--mu', default=0.0, type=float, help='stabilizer of hessian')
parser.add_argument('--entropy_reg', default=0.0, type=float, help='entropy regularizer')
parser.add_argument('--centroids_init', default='quantile', type=str, help='initialization method of centroids: linear/quantile')
parser.add_argument('--max_iter', default=30, type=int, help='max iteration for quantization')
parser.add_argument('--quant_finetune', default=False, type=bool, help='finetune after quantization or not')
parser.add_argument('--quant_finetune_lr', default=1e-3, type=float, help='learning rate for after-quant finetune')
parser.add_argument('--quant_finetune_epoch', default=12, type=int, help='num of epochs for after-quant finetune')
parser.add_argument('--is_pruned', default=False, type=bool, help='is the model pruned or not')
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

def quantize(model, weight_importance, weight_hessian, valid_ind, n_clusters, is_imagenet):
    assert len(n_clusters) == len(valid_ind)
    print('==>quantization clusters: {}'.format(n_clusters))

    quantize_layer_size = []
    for i, layer in enumerate(model.modules()):
        if i in valid_ind:
            quantize_layer_size.append([np.prod(layer.weight.size())])

    centroid_label_dict = quantize_model(model, weight_importance, weight_hessian, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=args.is_pruned, ha=args.ha, entropy_reg = args.entropy_reg)

    #Now get the overall compression rate
    huffman_size = get_huffmaned_weight_size(centroid_label_dict, quantize_layer_size, n_clusters)
    org_size = get_original_weight_size(quantize_layer_size)
    return huffman_size * 1.0 / org_size, centroid_label_dict

def get_importance(importance_type, t=1.0):
    #load file
    filename = args.type+"_"+importance_type
    if args.loss != 'cross_entropy' and importance_type in ['gradient', 'hessian']:
        filename += "_"+args.loss
    if t > 1.0:
        filename += "_t="+str(int(t))
    filename += ".pth"
    pathname = args.importance_root+args.type
    filepath = os.path.join(pathname, filename)

    assert os.path.isfile(filepath), "Please check whether importance file exist"

    with open(filepath, "rb") as f:
        weight_importance = torch.load(f)
    return weight_importance

def train(train_ds, model, criterion, optimizer, epoch, valid_ind, mask_list, is_imagenet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(tqdm.tqdm(train_ds, total=len(train_ds))):
        # measure data loading time
        data_time.update(time.time() - end)

        if is_imagenet:
            input = torch.from_numpy(input)
            target = torch.from_numpy(target)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        if 'inception' in args.type:
            output, aux_output = model(input_var)
            loss = criterion(output, target_var) + criterion(aux_output, target_var)
        else:
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # mask_gradient(model, prunable_ind, mask_list)

        optimizer.step()

        # mask_list = prune_fine_grained(model, prunable_ind, ratios)
        mask_weight(model, valid_ind, mask_list)

        batch_time.update(time.time() - end)
        end = time.time()

        #progress_bar(i, len(train_ds), 'Loss: %.3f | Acc1: %.3f%% | Acc5: %.3f%%'
        #             % (losses.avg, top1.avg, top5.avg))

    print('* Train epoch # %d    top1:  %.3f  top5:  %.3f' % (epoch, top1.avg, top5.avg))


def mask_gradient(model, valid_ind, mask_list):
    for i, m in enumerate(model.modules()):
        if i in valid_ind:
            mask = mask_list[valid_ind.index(i)]
            m.weight.grad.data *= mask.float()  # in-place mask of grad

def mask_weight(model, valid_ind, mask_list):
    if len(mask_list) == 0:
        return
    for i, m in enumerate(model.modules()):
        if i in valid_ind:
            mask = mask_list[valid_ind.index(i)]
            m.weight.data *= mask.float()  # in-place mask of grad

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.9 ** (epoch // (args.epoch // 4)))
    print('==> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def finetune(model, centroid_label_dict, train_ds, val_ds, valid_ind, is_imagenet):
    # fine-tune to preserve accuracy
    criterion = nn.CrossEntropyLoss().cuda()

    if args.fast_grad:
        # to speed up training, save the index tensor first
        weight_index_vars = []
        for key in valid_ind:
            cl_list = centroid_label_dict[key]
            centroids, labels = cl_list[0]
            ind_vars = []
            for i in range(centroids.size(1)):
                ind_vars.append((labels == i).float().cuda())
            weight_index_vars.append(ind_vars)

    for epoch in range(args.quant_finetune_epoch):
        print('Fine-tune epoch {}: '.format(i))
        if args.fast_grad:
            train(train_ds, model, criterion, epoch, valid_ind, centroid_label_dict, weight_index_vars, cycle=args.cycle)
        else:
            train(train_ds, model, criterion, epoch, quantizable_ind, centroid_label_dict, cycle=args.cycle)
        if (epoch+1)%args.eval_epoch == 0: 
            eval_and_print(model, train_ds, val_ds, is_imagenet, prefix_str="retraining epoch {}".format(epoch+1))

def eval_and_print(model, ds, is_imagenet, is_train,  prefix_str=""):
    if is_train:
        acc1, acc5, loss = misc.eval_model(model, ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
        print(prefix_str+" model, type={}, training acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1, acc5, loss))
    else:
        acc1, acc5, loss = misc.eval_model(model, ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
        print(prefix_str+" model, type={}, validation acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1, acc5, loss))
    return acc1, acc5, loss

def main():
    # load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)
    args.ngpu = args.ngpu if is_imagenet else 1

    # get valid layers
    valid_ind = []
    layer_type_list = []
    for i, layer in enumerate(model_raw.modules()):
        if type(layer) in valid_layer_types:
            valid_ind.append(i)
            layer_type_list.append(type(layer))

    #get training dataset and validation dataset and dataset for computing importance
    train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, input_size=args.input_size)
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    
    # eval raw model
    if not is_imagenet:
        acc1_train, acc5_train, loss_train  = eval_and_print(model_raw, train_ds, is_imagenet, is_train=True, prefix_str="Raw")
    acc1_val, acc5_val, loss_val = eval_and_print(model_raw, val_ds, is_imagenet, is_train=False, prefix_str="Raw")

    # get quantize ratios
    if args.bits is not None:
        clusters = [int(math.pow(2,r)) for r in eval(args.bits)]  # the actual ratio
    else:
        if args.fix_bit is not None:
            clusters = [int(math.pow(2,args.fix_bit))] * len(valid_ind)
        else:
            raise NotImplementedError

    #get weight importance
    if args.mode == 'normal':
        weight_importance = get_importance(importance_type='normal')
    elif args.mode == 'hessian':
        weight_importance = get_importance(importance_type='hessian', t=args.temperature)
        weight_importance_id = get_importance(importance_type='normal')
        for ix in weight_importance:
            weight_importance[ix] = weight_importance[ix] + args.mu*weight_importance_id[ix]
    elif args.mode == 'gradient':
        weight_importance = get_importance(importance_type='gradient', t=args.temperature)
    elif args.mode == 'KL':
        weight_importance = get_importance(importance_type='KL', t=args.temperature)
    if args.type in ['mnist', 'cifar10'] and args.mode != 'KL' and args.ha > 0.0:
        #get weight hessian
        weight_hessian = get_importance(importance_type='hessian', t=args.temperature)
        weight_hessian_id = get_importance(importance_type='normal')
        for ix in weight_hessian:
            weight_hessian[ix] = weight_hessian[ix] + args.mu*weight_hessian_id[ix]
    else:
        #TODO: delete this after hessian of cifar100 and alexnet were implemented
        weight_hessian = get_importance(importance_type='normal')

    #quantize
    compress_ratio, cl_list = quantize(model_raw, weight_importance, weight_hessian, valid_ind, clusters, is_imagenet)
    print("Quantization, ratio={:.4f}".format(compress_ratio))
    if not is_imagenet:
        acc1_train, acc5_train, loss_train  = eval_and_print(model_raw, train_ds, is_imagenet, is_train=True, prefix_str="Quantization")
    acc1_val, acc5_val, loss_val = eval_and_print(model_raw, val_ds, is_imagenet, is_train=False, prefix_str="Quantization")

    if args.quant_finetune:
        finetune(model_raw, cl_list, train_ds, val_ds, valid_ind, is_imagenet)
        print("Quantization and finetune, ratio={:.4f}".format(compress_ratio))
        if not is_imagenet:
            acc1_train, acc5_train, loss_train  = eval_and_print(model_raw, train_ds, is_imagenet, is_train=True, prefix_str="Quantization and finetune")
        acc1_val, acc5_val, loss_val = eval_and_print(model_raw, val_ds, is_imagenet, is_train=False, prefix_str="Quantization and finetune")


if __name__ == '__main__':
    main()

