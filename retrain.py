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
parser = argparse.ArgumentParser(description='Retrain model with fewer samples')
parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--gbs', type=int, default=10, help='input batch size for evaluating gradient')
parser.add_argument('--hbs', type=int, default=10, help='input batch size for evaluating hessian')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=2, help='number of gpus to use')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--optimizer', default='sgd', type=str, help='type of optimizer')
parser.add_argument('--save_root', default='sub_models/', help='folder for retrained models')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
#Retrain argument
parser.add_argument('--number_of_models', default=20, type=int, help='number of independent models to retrain')
parser.add_argument('--starting_index', default=0, type=int, help='start index for naming models')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate for retrain')
parser.add_argument('--epoch', default=25, type=int, help='num of epochs for retrain')
parser.add_argument('--eval_epoch', default=25, type=int, help='evaluate performance per how many epochs')
parser.add_argument('--subsample_rate', default=1.0, type=float, help='subsample_rate for retrain')
parser.add_argument('--compute_gradient', default=False, type=bool, help='compute gradient for retrained model')
parser.add_argument('--compute_hessian', default=False, type=bool, help='compute hessian for retrained model')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for model calibration')
args = parser.parse_args()

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

def train(train_ds, model, criterion, optimizer, epoch, is_imagenet):
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

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print('* Train epoch # %d    top1:  %.3f  top5:  %.3f' % (epoch, top1.avg, top5.avg))

def eval_and_print(model, train_ds, val_ds, is_imagenet, prefix_str=""):
    acc1_train, acc5_train, loss_train = misc.eval_model(model, train_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    acc1_val, acc5_val, loss_val = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    print(prefix_str+" model, type={}, training acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1_train, acc5_train, loss_train))
    print(prefix_str+" model, type={}, validation acc1={:.4f}, acc5={:.4f}, loss={:.6f}".format(args.type, acc1_val, acc5_val, loss_val))
    return acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.9 ** (epoch // (args.epoch // 4)))
    print('==> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def retrain(model, train_ds, val_ds, valid_ind, is_imagenet):
    for i, m in enumerate(model.modules()):
        if i in valid_ind:
            torch.nn.init.xavier_normal_(m.weight)

    best_acc, best_acc5, best_loss = misc.eval_model(model, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    best_model = model
    criterion = nn.CrossEntropyLoss()

    if 'inception' in args.type or args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    for epoch in range(args.epoch):
        #adjust_learning_rate(optimizer, epoch)
        train(train_ds, model, criterion, optimizer, epoch, is_imagenet)
        if (epoch+1)%args.eval_epoch == 0: 
            acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val = eval_and_print(model, train_ds, val_ds, is_imagenet, prefix_str="retraining epoch {}".format(epoch+1))


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

    for i in range(args.number_of_models):
        #get training dataset and validation dataset
        if args.subsample_rate < 1.0:
            indices = np.random.choice(training_size, int(args.subsample_rate*training_size/args.batch_size)*args.batch_size, replace=True)
            train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, subsample=True, indices=indices, input_size=args.input_size)
        else:
            train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, val=False, input_size=args.input_size)
            indices = np.arange(training_size)
        val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    
        # eval raw model
        acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Raw")
    
        # retrain model
        retrain(model_raw, train_ds, val_ds, valid_ind, is_imagenet)
        acc1_train, acc5_train, loss_train, acc1_val, acc5_val, loss_val = eval_and_print(model_raw, train_ds, val_ds, is_imagenet, prefix_str="Retrained "+str(i+args.starting_index))

        #save retrained model
        filename = args.type+"_model_"+str(i+args.starting_index)+".pth.tar"
        pathname = args.save_root+args.type+"/ssr="+str(int(args.subsample_rate*1000))
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        filepath = os.path.join(pathname, filename)
        with open(filepath, "wb") as f:
            torch.save({
                'number': i,
                'subsample_rate': args.subsample_rate,
                'ds_indices': indices,
                'model_state_dict': model_raw.state_dict(),
                }, f)

        #compute importance and write to file
        weight_importance = get_all_one_importance(model_raw, valid_ind, is_imagenet)
        filename = args.type+"_normal_"+str(i+args.starting_index)
        if args.temperature > 1.0:
            filename += "_t="+str(int(args.temperature))
        filename += ".pth"
        filepath = os.path.join(pathname, filename)
        with open(filepath, "wb") as f:
            torch.save(weight_importance, f)

        if args.compute_gradient:
            #compute importance and write to file
            ds_for_importance = ds_fetcher(args.gbs, data_root=args.data_root, val=False, subsample=True, indices=indices, input_size=args.input_size)
            weight_importance = get_gradient_importance(model_raw, ds_for_importance, valid_ind, is_imagenet)
            filename = args.type+"_gradient_"+str(i+args.starting_index)
            if args.temperature > 1.0:
                filename += "_t="+str(int(args.temperature))
            filename += ".pth"
            filepath = os.path.join(pathname, filename)
            with open(filepath, "wb") as f:
                torch.save(weight_importance, f)

        if args.compute_hessian:
            #compute importance and write to file
            ds_for_hessian = ds_fetcher(args.hbs, data_root=args.data_root, val=False, subsample=True, indices=indices, input_size=args.input_size)
            weight_importance = get_hessian_importance(model_raw, ds_for_hessian, valid_ind, is_imagenet)
            filename = args.type+"_hessian_"+str(i+args.starting_index)
            if args.temperature > 1.0:
                filename += "_t="+str(int(args.temperature))
            filename += ".pth"
            filepath = os.path.join(pathname, filename)
            with open(filepath, "wb") as f:
                torch.save(weight_importance, f)


if __name__ == '__main__':
    main()
