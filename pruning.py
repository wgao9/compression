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
#Pruning arguments
parser.add_argument('--mode', default='none', type=str, help='pruning mode: none/normal/gradient/hessian')
parser.add_argument('--loss', default='cross_entropy', type=str, help='loss funciton')
parser.add_argument('--ratios', default=None, type=str, help='ratios of pruning')
parser.add_argument('--fix_ratio', default=None, type=float, help='fix ratio for every layer')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for estimating gradient')
parser.add_argument('--hessian_mode', default='fix', type=str, help='mode of hessian term: fix/true')
parser.add_argument('--ha', default=0.0, type=float, help='estimate of average of H_ii/multiplier of true hessian')
parser.add_argument('--modify_dropout', action='store_true', help='modify the rate of dropout')
parser.add_argument('--stage', default=1, type=int, help='retrain stage 1/2/3')
parser.add_argument('--prune_finetune', default=False, type=bool, help='finetune after prune or not')
parser.add_argument('--prune_finetune_lr', default=1e-3, type=float, help='learning rate for after-prune finetune')
parser.add_argument('--prune_finetune_epoch', default=12, type=int, help='num of epochs for after-prune finetune')
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

def prune(model, weight_importance, weight_hessian, valid_ind, ratios, is_imagenet):
    assert len(ratios) == len(valid_ind), (len(ratios), len(valid_ind))
    print('=> Given pruning ratios: {}'.format(np.array(ratios).round(3)))

    m_list = list(model.modules())
    importances = {}
    for ix in valid_ind:
        m = m_list[ix]
        importances[ix] = weight_importance[ix] + (args.ha*weight_hessian[ix]*m.weight.data)**2

    #Pruning
    mask_list = prune_fine_grained(model, valid_ind, ratios, criteria='importance', importances=importances)

    # now get the overall compression rate
    full_size = 0.
    pruned_size = 0.
    m_list = list(model.modules())
    for ix in valid_ind:
        full_size += m_list[ix].weight.data.numel()
        pruned_size += torch.sum(m_list[ix].weight.data.eq(0)).item()
    total_compress_ratio = (full_size - pruned_size) * 1.0 / full_size
    return mask_list, total_compress_ratio

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

    assert os.path.isfile(filepath), "Please check whether importance file exists"

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

def retrain(model, train_ds, val_ds, valid_ind, mask_list, is_imagenet):
    best_acc, best_acc5, best_loss = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    best_model = model
    criterion = nn.CrossEntropyLoss()
    epochs = args.prune_finetune_epoch
    lrs = args.prune_finetune_lr

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
    return acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val

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
    #eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Raw")
    
    #Pruning stage by stage
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
        if args.mode == 'normal':
            weight_importance = get_importance(importance_type='normal')
        elif args.mode == 'hessian':
            weight_importance = get_importance(importance_type='hessian', t=args.temperature)
        elif args.mode == 'gradient':
            weight_importance = get_importance(importance_type='gradient', t=args.temperature)
        #get weight hessian
        if args.hessian_mode == 'true':
            weight_hessian = get_importance(importance_type='hessian', t=args.temperature)
        else:
            weight_hessian = get_importance(importance_type='normal')

        #prune
        mask_list, compression_ratio = prune(model_raw, weight_importance, weight_hessian, valid_ind, ratios, is_imagenet)
        print("Pruning stage {}, compression ratio {:.4f}".format(stage+1, compression_ratio))
        acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Prune stage {}".format(stage+1))

        #and finetune
        if args.prune_finetune:
            retrain(model_raw, train_ds, val_ds, valid_ind, mask_list, is_imagenet, is_retrain=False)
            acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Prune and finetune stage {}".format(stage+1))


if __name__ == '__main__':
    main()

