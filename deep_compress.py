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
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--gradient_batch_size', type=int, default=10, help='batch size for computing gradient square')
parser.add_argument('--hessian_subsample_rate', type=float, default=1.0, help='subsample rate for computing hessian')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
#Pruning arguments
parser.add_argument('--prune_mode', default='none', type=str, help='pruning mode: none/normal/gradient/hessian')
parser.add_argument('--ratios', default=None, type=str, help='ratios of pruning')
parser.add_argument('--fix_ratio', default=None, type=float, help='fix ratio for every layer')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for estimating gradient')
parser.add_argument('--hessian_average', default=0.0, type=float, help='estimate of average of H_ii')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--modify_dropout', action='store_true', help='modify the rate of dropout')
parser.add_argument('--stage', default=1, type=int, help='retrain stage 1/2/3')
parser.add_argument('--prune_finetune', default=False, type=bool, help='finetune after prune or not')
parser.add_argument('--prune_finetune_lr', default=1e-3, type=float, help='learning rate for after-prune finetune')
parser.add_argument('--prune_finetune_epoch', default=12, type=int, help='num of epochs for after-prune finetune')
#Quantization arguments
parser.add_argument('--quantization_mode', default='normal', type=str, help='quantization mode: none/normal/gradient/hessian')
parser.add_argument('--bits', default=None, type=str, help='number of bits of quantization')
parser.add_argument('--fix_bit', default=None, type=float, help='fix bit for every layer')
parser.add_argument('--entropy_reg', default=0.0, type=float, help='entropy regularizer')
parser.add_argument('--diameter_reg', default=0.0, type=float, help='diameter regularizer')
parser.add_argument('--centroids_init', default='quantile', type=str, help='initialization method of centroids: linear/quantile')
parser.add_argument('--max_iter', default=30, type=int, help='max iteration for quantization')
parser.add_argument('--quant_finetune', default=False, type=bool, help='finetune after quantization or not')
parser.add_argument('--quant_finetune_lr', default=1e-3, type=float, help='learning rate for after-quant finetune')
parser.add_argument('--quant_finetune_epoch', default=12, type=int, help='num of epochs for after-quant finetune')
#Retrain argument
parser.add_argument('--retrain', default=False, type=bool, help='if use pretrained model')
parser.add_argument('--retrain_lr', default=1e-2, type=float, help='learning rate for retrain')
parser.add_argument('--retrain_epoch', default=25, type=int, help='num of epochs for retrain')
parser.add_argument('--eval_epoch', default=5, type=int, help='evaluate performance per how many epochs')
parser.add_argument('--subsample_rate', default=1.0, type=float, help='subsample_rate for retrain')
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

def prune(model, weight_importance, valid_ind, ratios, is_imagenet):
    assert len(ratios) == len(valid_ind), (len(ratios), len(valid_ind))
    print('=> Given pruning ratios: {}'.format(np.array(ratios).round(3)))

    #Pruning
    if args.prune_mode == 'normal':
        mask_list = prune_fine_grained(model, valid_ind, ratios, criteria='normal')  # get the mask
    elif args.prune_mode in ['gradient', 'hessian']:
        mask_list = prune_fine_grained(model, valid_ind, ratios, criteria='importance', importances=weight_importance)
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

def quantize(model, weight_importance, valid_ind, n_clusters, is_imagenet):
    assert len(n_clusters) == len(valid_ind)
    print('==>quantization clusters: {}'.format(n_clusters))

    is_pruned = False if args.prune_mode == 'none' else True
    is_entropy_reg = True if args.entropy_reg > 0.0 else False
    is_diameter_reg = True if args.diameter_reg > 0.0 else False

    quantize_layer_size = []
    for i, layer in enumerate(model.modules()):
        if i in valid_ind:
            quantize_layer_size.append([np.prod(layer.weight.size())])

    if not is_entropy_reg and not is_diameter_reg:
        centroid_label_dict = quantize_model(model, weight_importance, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=is_pruned, ha=args.hessian_average)
    elif is_entropy_reg:
        centroid_label_dict = quantize_model(model, weight_importance, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=is_pruned, ha=args.hessian_average, entropy_reg = args.entropy_reg)
    elif is_diameter_reg:
        centroid_label_dict = quantize_model(model, weight_importance, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=is_pruned, ha=args.hessian_average, diameter_reg = args.diameter_reg)
    else:
        raise NotImplementedError

    #Now get the overall compression rate
    huffman_size = get_huffmaned_weight_size(centroid_label_dict, quantize_layer_size, n_clusters)
    org_size = get_original_weight_size(quantize_layer_size)
    return huffman_size * 1.0 / org_size

def get_importance(model, ds_for_importance, valid_ind, is_imagenet, importance_type, ha=0.0):
    if importance_type == 'normal':
        return get_all_one_importance(model, valid_ind, is_imagenet, ha=ha)
    elif importance_type == 'gradient':
        return get_gradient_importance(model, ds_for_importance, valid_ind, is_imagenet, ha=ha)
    elif importance_type == 'hessian':
        return get_hessian_importance(model, ds_for_importance, valid_ind, is_imagenet, ha=ha)

def get_all_one_importance(model, valid_ind, is_imagenet, ha=0.0):
    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        m = m_list[ix]
        importances[ix] = torch.ones_like(m.weight) + ha*m.weight.data**2
    return importances

def get_gradient_importance(model, ds_for_importance, valid_ind, is_imagenet, ha=0.0):
    criterion = nn.CrossEntropyLoss()
    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        importances[ix] = 0.

    for i, (input, target) in enumerate(tqdm.tqdm(ds_for_importance, total=len(ds_for_importance))):
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
            importances[ix] += m.weight.grad.data**2 + ha*m.weight.data**2
            
    for ix in valid_ind:
        importances[ix] = importances[ix] / importances[ix].mean()
        
    return importances

def get_hessian_importance(model, ds_for_importance, valid_ind, is_imagenet, ha=0.0):
    criterion = nn.CrossEntropyLoss()
    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        importances[ix] = 0.

    for i, (input, target) in enumerate(tqdm.tqdm(ds_for_importance, total=len(ds_for_importance))):
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

        dhs = diagonal_hessian_multi(loss, output, model.parameters()) 
        
        for ii in range(len(valid_ind)):
            ix = valid_ind[ii]
            m = m_list[ix]
            importances[ix] += dhs[2*ii] + ha*m.weight.data**2
            
    for ix in valid_ind:
        importances[ix] = importances[ix] / importances[ix].mean()
        
    return importances

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

def retrain(model, train_ds, val_ds, valid_ind, mask_list, is_imagenet, is_retrain = False):
    if is_retrain:
        for i, m in enumerate(model.modules()):
            if i in valid_ind:
                torch.nn.init.xavier_normal_(m.weight)

    best_acc, best_acc5, best_loss = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    best_model = model
    criterion = nn.CrossEntropyLoss()
    epochs = args.retrain_epoch if is_retrain else args.prune_finetune_epoch
    lrs = args.retrain_lr if is_retrain else args.prune_finetune_lr

    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lrs, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lrs, momentum=0.9, weight_decay=args.decay)

    for epoch in range(epochs):
        #adjust_learning_rate(optimizer, epoch)
        train(train_ds, model, criterion, optimizer, epoch, valid_ind, mask_list, is_imagenet)
        if (epoch+1)%args.eval_epoch == 0: 
            eval_and_print(model, train_ds, val_ds, is_imagenet, prefix_str="retraining epoch {}".format(epoch+1))

        #if acc1 > best_acc:
        #    best_acc = acc1
        #    best_model = model

    model = best_model

def eval_and_print(model, train_ds, val_ds, is_imagenet, prefix_str=""):
    acc1_train, acc5_train, loss_train = misc.eval_model(model, train_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    acc1_val, acc5_val, loss_val = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    print(prefix_str+" model, type={}, training acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1_train, acc5_train, loss_train))
    print(prefix_str+" model, type={}, validation acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1_val, acc5_val, loss_val))

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

    #get training dataset and validation dataset
    if args.subsample_rate < 1.0 and args.type == 'mnist':
        indices = np.random.choice(60000, int(args.subsample_rate*60000/args.batch_size)*args.batch_size, replace=True)
        train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, subsample=True, indices=indices, input_size=args.input_size)
    else:
        train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, input_size=args.input_size)
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    
    # eval raw model
    eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Raw")
    
    # retrain if we want self trained model
    if args.retrain == True:
        retrain(model_raw, train_ds, val_ds, valid_ind, [], is_imagenet, is_retrain=True)
        eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Retrained")
        

    #Pruning
    if args.prune_mode != 'none':
        #stage by stage
        for stage in range(args.stage):
            #get pruning ratios
            if args.ratios is not None:
                ratios = [math.pow(r, (stage+1.0)/args.stage) for r in eval(args.ratios)]  # the actual ratio
            else:
                if args.fix_ratio is not None:
                    ratios = [math.pow(args.fix_ratio, (stage+1.0)/args.stage)] * len(valid_ind)
                else:
                    raise NotImplementedError

            #get weight importance
            if args.prune_mode == 'normal':
                weight_importance = get_importance(model_raw, train_ds, valid_ind, is_imagenet, importance_type='normal', ha=args.hessian_average)
            elif args.prune_mode == 'hessian':
                weight_importance = get_importance(model_raw, train_ds, valid_ind, is_imagenet, importance_type='hessian', ha=args.hessian_average)
            elif args.prune_mode == 'gradient':
                ds_for_gradient = ds_fetcher(args.gradient_batch_size, data_root=args.data_root, val=False, input_size=args.input_size)
                weight_importance = get_importance(model_raw, ds_for_gradient, valid_ind, is_imagenet, importance_type='gradient', ha=args.hessian_average)

            #prune
            mask_list, compression_ratio = prune(model_raw, weight_importance, valid_ind, ratios, is_imagenet)
            print("Pruning stage {}, compression ratio {:.4f}".format(stage+1, compression_ratio))
            eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Prune stage {}".format(stage+1))

            #and finetune
            if args.prune_finetune:
                retrain(model_raw, train_ds, val_ds, valid_ind, mask_list, is_imagenet, is_retrain=False)
                eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Prune and finetune stage {}".format(stage+1))
   
    #Quantization
    if args.quantization_mode != 'none':
        # get quantize ratios
        if args.bits is not None:
            clusters = [int(math.pow(2,r)) for r in eval(args.bits)]  # the actual ratio
        else:
            if args.fix_bit is not None:
                clusters = [int(math.pow(2,args.fix_bit))] * len(valid_ind)
            else:
                raise NotImplementedError

        #get weight importance
        if args.quantization_mode == 'normal':
            weight_importance = get_importance(model_raw, train_ds, valid_ind, is_imagenet, importance_type='normal')
        elif args.quantization_mode == 'hessian':
            weight_importance = get_importance(model_raw, train_ds, valid_ind, is_imagenet, importance_type='hessian')
        elif args.quantization_mode == 'gradient':
            ds_for_gradient = ds_fetcher(args.gradient_batch_size, data_root=args.data_root, val=False, input_size=args.input_size)
            weight_importance = get_importance(model_raw, ds_for_gradient, valid_ind, is_imagenet, importance_type='gradient')

        #quantize
        compress_ratio = quantize(model_raw, weight_importance, valid_ind, clusters, is_imagenet)
        print("Quantization, ratio={:.4f}".format(compress_ratio))
        eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Quantization")

        if args.quant_finetune:
            #FIXME: old version, need to update
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

            for i in range(args.quant_finetune_epoch):
                print('Fine-tune epoch {}: '.format(i))
                if args.fast_grad:
                    train(train_loader, model, criterion, i, quantizable_ind, centroid_label_dict, weight_index_vars,
                        cycle=args.cycle)
                else:
                    train(train_loader, model, criterion, i, quantizable_ind, centroid_label_dict, cycle=args.cycle)
                top1 = validate(val_loader, model)


if __name__ == '__main__':
    main()

