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
import sys
import os
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for estimating gradient')
#Quantization arguments
parser.add_argument('--mode', default='normal', type=str, help='quantization mode: none/normal/gradient/hessian')
parser.add_argument('--bits', default=None, type=str, help='number of bits of quantization')
parser.add_argument('--fix_bit', default=None, type=float, help='fix bit for every layer')
parser.add_argument('--diameter_reg', default=0.0, type=float, help='diameter regularizer')
parser.add_argument('--diameter_entropy_reg', default=0.0, type=float, help='diameter times entropyregularizer')
parser.add_argument('--centroids_init', default='quantile', type=str, help='initialization method of centroids: linear/quantile')
parser.add_argument('--max_iter', default=30, type=int, help='max iteration for quantization')
parser.add_argument('--quant_finetune', default=False, type=bool, help='finetune after quantization or not')
parser.add_argument('--quant_finetune_lr', default=1e-3, type=float, help='learning rate for after-quant finetune')
parser.add_argument('--quant_finetune_epoch', default=12, type=int, help='num of epochs for after-quant finetune')
parser.add_argument('--is_pruned', default=False, type=bool, help='Is pruned model or not?')
#Retrain argument
parser.add_argument('--number_of_models', default=5, type=int, help='number of models to use')
parser.add_argument('--subsample_rate', default=0.05, type=float, help='subsampling rate')
parser.add_argument('--save_root', default='sub_models/', type=str, help='folder for retrained models')
parser.add_argument('--result_root', default=None, type=str, help='folder to store results')
parser.add_argument('--result_name', default='', type=str, help='folder to store results')
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

    centroid_label_dict = quantize_model(model, weight_importance, weight_hessian, valid_ind, n_clusters, max_iter=args.max_iter, mode='cpu', is_pruned=args.is_pruned, diameter_reg = args.diameter_reg, diameter_entropy_reg = args.diameter_entropy_reg)

    #Now get the overall compression rate
    huffman_size = get_huffmaned_weight_size(centroid_label_dict, quantize_layer_size, n_clusters)
    org_size = get_original_weight_size(quantize_layer_size)
    return huffman_size * 1.0 / org_size, centroid_label_dict

def get_importance(importance_type, index, t=1.0):
    #load file
    filename = args.type+"_"+importance_type+"_"+str(index)
    if args.temperature > 1.0:
        filename += "_t="+str(int(args.temperature))
    filename += ".pth"
    pathname = args.save_root+args.type+"/ssr="+str(int(args.subsample_rate*1000))
    filepath = os.path.join(pathname, filename)

    assert os.path.isfile(filepath), "Please check "+filepath+" exists"

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

    # get quantize ratios
    if args.bits is not None:
        clusters = [int(math.pow(2,r)) for r in eval(args.bits)]  # the actual ratio
    else:
        if args.fix_bit is not None:
            clusters = [int(math.pow(2,args.fix_bit))] * len(valid_ind)
        else:
            raise NotImplementedError

    metrics = np.zeros((19,args.number_of_models))
    #iterate over all retrained models
    for i in range(args.number_of_models):
        #load retrained model
        filename = args.type+"_model_"+str(i)+".pth.tar"
        pathname = args.save_root+args.type+"/ssr="+str(int(args.subsample_rate*1000))
        filepath = os.path.join(pathname, filename)
        assert os.path.isfile(filepath), "Can not find model"

        with open(filepath, "rb") as f:
            print("Loading model parameters from"+filepath)
            checkpoint = torch.load(f)
            model_raw.load_state_dict(checkpoint['model_state_dict'])
            ds_indices = checkpoint['ds_indices']

        #get training dataset and validation dataset
        train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, subsample=True, indices=ds_indices, input_size=args.input_size)
        val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    
        #get weight importance
        if args.mode == 'hessian':
            weight_importance = get_importance('hessian', i, t=args.temperature)
        elif args.mode == 'gradient':
            weight_importance = get_importance('gradient', i, t=args.temperature)
        else:
            weight_importance = get_importance('normal', i)
        weight_hessian = get_importance('normal', i)
        
        # eval raw model
        metrics[1,i], metrics[2,i], metrics[3,i], metrics[4,i], metrics[5,i], metrics[6,i] = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Retrained model number %d"%i)

        #quantize
        metrics[0,i], cl_list = quantize(model_raw, weight_importance, weight_hessian, valid_ind, clusters, is_imagenet)
        #print("Quantization, ratio={:.4f}".format(metrics[0,i]))
        metrics[7,i], metrics[8,i], metrics[9,i], metrics[10,i], metrics[11,i], metrics[12,i] = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="After quantization number %d"%i)

        if args.quant_finetune:
            finetune(model_raw, cl_list, train_ds, val_ds, valid_ind, is_imagenet)
            #print("Quantization and finetune, ratio={:.4f}".format(metrics[0,i]))
            metrics[13,i], metrics[14,i], metrics[15,i], metrics[16,i], metrics[17,i], metrics[18,i] = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="After finetune number %d"%i)

    #print average performance
    print("Before quantization, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f}".format(args.type, np.mean(metrics[1]), np.std(metrics[1]), np.mean(metrics[2]), np.std(metrics[2]), np.mean(metrics[3]), np.std(metrics[3])))
    print("Before quantization, type={}, validation acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f}".format(args.type, np.mean(metrics[4]), np.std(metrics[4]), np.mean(metrics[5]), np.std(metrics[5]), np.mean(metrics[6]), np.std(metrics[6])))
    print("Compression ratio = {:.4f}+-{:.4f}".format(np.mean(metrics[0]), np.std(metrics[0])))
    print("After quantization, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f}".format(args.type, np.mean(metrics[7]), np.std(metrics[7]), np.mean(metrics[8]), np.std(metrics[8]), np.mean(metrics[9]), np.std(metrics[9])))
    print("After quantization, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f}".format(args.type, np.mean(metrics[10]), np.std(metrics[10]), np.mean(metrics[11]), np.std(metrics[11]), np.mean(metrics[12]), np.std(metrics[12])))
    if args.quant_finetune:
        print("After finetune, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f}".format(args.type, np.mean(metrics[13]), np.std(metrics[13]), np.mean(metrics[14]), np.std(metrics[14]), np.mean(metrics[15]), np.std(metrics[15])))
        print("After finetune, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f}".format(args.type, np.mean(metrics[16]), np.std(metrics[16]), np.mean(metrics[17]), np.std(metrics[17]), np.mean(metrics[18]), np.std(metrics[18])))

    if args.result_root != None:
        filename = args.type+"_"+args.mode+"_"+args.result_name
        pathname = args.result_root
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        filepath = os.path.join(pathname, filename)

        with open(filepath, "w") as f:
            for arg in sys.argv:
                f.write(arg+" ")
            f.write("\n")
            f.write("\n")
            f.write("Before quantization, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f} \n".format(args.type, np.mean(metrics[1]), np.std(metrics[1]), np.mean(metrics[2]), np.std(metrics[2]), np.mean(metrics[3]), np.std(metrics[3])))
            f.write("Before quantization, type={}, validation acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f} \n".format(args.type, np.mean(metrics[4]), np.std(metrics[4]), np.mean(metrics[5]), np.std(metrics[5]), np.mean(metrics[6]), np.std(metrics[6])))
            f.write("Compression ratio = {:.4f}+-{:.4f} \n".format(np.mean(metrics[0]), np.std(metrics[0])))
            f.write("After quantization, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f} \n".format(args.type, np.mean(metrics[7]), np.std(metrics[7]), np.mean(metrics[8]), np.std(metrics[8]), np.mean(metrics[9]), np.std(metrics[9])))
            f.write("After quantization, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f} \n".format(args.type, np.mean(metrics[10]), np.std(metrics[10]), np.mean(metrics[11]), np.std(metrics[11]), np.mean(metrics[12]), np.std(metrics[12])))
            if args.quant_finetune:
                f.write("After finetune, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f} \n".format(args.type, np.mean(metrics[13]), np.std(metrics[13]), np.mean(metrics[14]), np.std(metrics[14]), np.mean(metrics[15]), np.std(metrics[15])))
                f.write("After finetune, type={}, training acc1={:.4f}+-{:.4f}, acc5={:.4f}+-{:.4f}, loss={:.6f}+-{:.6f} \n".format(args.type, np.mean(metrics[16]), np.std(metrics[16]), np.mean(metrics[17]), np.std(metrics[17]), np.mean(metrics[18]), np.std(metrics[18])))
            

if __name__ == '__main__':
    main()

