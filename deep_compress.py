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

import time

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
#Pruning arguments
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for estimating gradient')
parser.add_argument('--hessian_average', default=0.0, type=float, help='estimate of average of H_ii')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epoch', default=12, type=int, help='num of epochs for fine-tuning')
parser.add_argument('--modify_dropout', action='store_true', help='modify the rate of dropout')
parser.add_argument('--stage', default=1, type=int, help='retrain stage 1/2/3')
parser.add_argument('--ratios', default=None, type=str, help='ratios of pruning')
parser.add_argument('--prune_mode', default='none', type=str, help='pruning mode: none/normal/wng')
parser.add_argument('--fix_ratio', default=None, type=float, help='fix ratio for every layer')
parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
parser.add_argument('--bits', default=None, type=str, help='number of bits of pruning')
parser.add_argument('--quantization_mode', default='normal', type=str, help='quantization mode: none/normal/variational')
parser.add_argument('--fix_bit', default=None, type=float, help='fix bit for every layer')
parser.add_argument('--variational_lambda', default=1e-9, type=float, help='lambda for variational update')
parser.add_argument('--pretrained', default=True, type=bool, help='if use pretrained model')
parser.add_argument('--max_iter', default=30, type=int, help='max iteration for quantization')
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

def prune(model, ds_fetcher, valid_ind, ratios, is_imagenet):
    assert len(ratios) == len(valid_ind), (len(ratios), len(valid_ind))
    print('=> Given pruning ratios: {}'.format(np.array(ratios).round(3)))

    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    #Pruning
    if args.prune_mode == 'normal':
        mask_list = prune_fine_grained(model, valid_ind, ratios, criteria='magnitude')  # get the mask
    elif args.prune_mode == 'wng':
        weight_importance = get_importance(model, ds_fetcher, optimizer, valid_ind, is_imagenet)
        mask_list = prune_fine_grained(model, valid_ind, ratios, criteria='w_and_g', importances=weight_importance)
        optimizer.zero_grad()
    else:
        raise NotImplementedError

    # now get the overall compression rate
    full_size = 0.
    pruned_size = 0.
    m_list = list(model.modules())
    for ix in valid_ind:
        full_size += m_list[ix].weight.data.numel()
        pruned_size += torch.sum(m_list[ix].weight.data.eq(0)).item()
    total_compress_ratio = (full_size - pruned_size) * 1.0 / full_size
    return mask_list, total_compress_ratio

def finetune(model, ds_fetcher, valid_ind, mask_list, is_imagenet):
    print('==> Beginning fine-tuning...')
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    best_acc, best_acc5, best_loss = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    best_model = model
    criterion = nn.CrossEntropyLoss()
    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch)
        train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, input_size=args.input_size)
        train(train_ds, model, criterion, optimizer, epoch, valid_ind, mask_list, is_imagenet)
        val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
        acc1, acc5, loss = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

        res_str = "type={}, epoch={}, acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, epoch, acc1, acc5, loss)
        print(res_str)
        if acc1 > best_acc:
            best_acc = acc1
            best_model = model

    model = best_model

def quantize(model, ds_fetcher, valid_ind, n_clusters, is_imagenet):
    assert len(n_clusters) == len(valid_ind)
    print('==>quantization clusters: {}'.format(n_clusters))

    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    is_pruned = True if args.prune_mode == 'normal' or args.prune_mode == 'wng' else False

    quantize_layer_size = []
    for i, layer in enumerate(model.modules()):
        if i in valid_ind:
            quantize_layer_size.append([np.prod(layer.weight.size())])

    if args.quantization_mode == 'normal':
        centroid_label_dict = quantize_model(model, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=is_pruned)
    elif args.quantization_mode == 'variational':
        weight_importance = get_importance(model, ds_fetcher, optimizer, valid_ind, is_imagenet)
        centroid_label_dict = quantize_model(model, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=is_pruned, is_variational=True, importances=weight_importance, variational_lambda=args.variational_lambda)
        optimizer.zero_grad()
    else:
        raise NotImplementedError

    huffman_size = get_huffmaned_weight_size(centroid_label_dict, quantize_layer_size, n_clusters)
    org_size = get_original_weight_size(quantize_layer_size)
    return huffman_size * 1.0 / org_size
'''
        # fine-tune to preserve accuracy
        criterion = nn.CrossEntropyLoss().cuda()

        if args.fast_grad:
            # to speed up training, save the index tensor first
            weight_index_vars = []
            for key in quantizable_ind:
                cl_list = centroid_label_dict[key]
                centroids, labels = cl_list[0]
                ind_vars = []
                for i in range(centroids.size(1)):
                    ind_vars.append((labels == i).float().cuda())
                weight_index_vars.append(ind_vars)

        for i in range(args.finetune_epoch):
            print('Fine-tune epoch {}: '.format(i))
            if args.fast_grad:
                train(train_loader, model, criterion, i, quantizable_ind, centroid_label_dict, weight_index_vars,
                      cycle=args.cycle)
            else:
                train(train_loader, model, criterion, i, quantizable_ind, centroid_label_dict, cycle=args.cycle)
            top1 = validate(val_loader, model)
'''

def get_importance(model, ds_fetcher, optimizer, valid_ind, is_imagenet):
    gbs = 100
    if 'mnist' in args.type:
        gbs = 1
    elif 'cifar' in args.type:
        gbs = 10

    criterion = nn.CrossEntropyLoss()
    ds_for_gradients = ds_fetcher(gbs, data_root=args.data_root, train=False, input_size=args.input_size) 
    m_list = list(model.modules())
    expectations = {}
    for ix in valid_ind:
        expectations[ix] = 0.
    for i, (input, target) in enumerate(tqdm.tqdm(ds_for_gradients, total=len(ds_for_gradients))):
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
            expectations[ix] += m.weight.grad.data**2+0.5*args.hessian_average*m.weight.data**2
            
    for ix in valid_ind:
        zero_rate = 100.0 - (expectations[ix].nonzero().size(0) * 100.0 / expectations[ix].numel())
        print("Layer {}, {:4f} percent importance are zero".format(ix, zero_rate))
        expectations[ix] = expectations[ix] / expectations[ix].mean()
        
    return expectations

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

def retrain(model, ds_fetcher, valid_ind, is_imagenet):
    for i, m in enumerate(model.modules()):
        if i in valid_ind:
            torch.nn.init.xavier_normal_(m.weight)
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    acc1, acc5, loss = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    print("Self-trained model, type={}, acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1, acc5, loss))
    finetune(model, ds_fetcher, valid_ind, [], is_imagenet)

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
    
    # eval pretrainedmodel
   # val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
   # acc1, acc5, loss = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
   # print("Pretrained model, type={}, acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1, acc5, loss))
    
    # retrain if we want self trained model
    if args.pretrained == False:
        retrain(model_raw, ds_fetcher, valid_ind, is_imagenet)

    #Pruning
    if args.prune_mode != 'none':
        #get prune ratios
        
        for i in range(args.stage):
            if args.ratios is not None:
                ratios = [math.pow(r, (i+1.0)/args.stage) for r in eval(args.ratios)]  # the actual ratio
            else:
                if args.fix_ratio is not None:
                    ratios = [math.pow(args.fix_ratio, (i+1.0)/args.stage)] * len(valid_ind)
                else:
                    raise NotImplementedError

            #prune
            mask_list, compression_ratio = prune(model_raw, ds_fetcher, valid_ind, ratios, is_imagenet)
            val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
            acc1, acc5, loss = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
            print("Prune stage {}, type={}, acc1={:.4f}, acc5={:.4f}, loss={:.6f}, ratio={:.4f}".format(i+1, args.type, acc1, acc5, loss, compression_ratio))

            #and finetune
   #         finetune(model_raw, ds_fetcher, valid_ind, mask_list, is_imagenet)
   #         val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
   #         acc1, acc5, loss = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
   #         print("Prune and finetune stage {}, type={}, acc1={:.4f}, acc5={:.4f}, loss={:.6f}, ratio={:.4f}".format(i+1, args.type, acc1, acc5, loss, compression_ratio))
    
    if args.quantization_mode != 'none':
        # get quantize ratios
        if args.bits is not None:
            clusters = [int(math.pow(2,r)) for r in eval(args.bits)]  # the actual ratio
        else:
            if args.fix_bit is not None:
                clusters = [int(math.pow(2,args.fix_bit))] * len(valid_ind)
            else:
                raise NotImplementedError

        #quantize
        compress_ratio = quantize(model_raw, ds_fetcher, valid_ind, clusters, is_imagenet)
        val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
        acc1, acc5, loss = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
        print("Quantization, type={}, acc1={:.4f}, acc5={:.4f}, loss={:.6f}, ratio={:.4f}".format(args.type, acc1, acc5, loss, compress_ratio))

if __name__ == '__main__':
    main()

