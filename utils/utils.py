import os
import time
import sys
import torch
import math
import torch.nn.functional as F
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class Logger(object):
    """Write log immediately to the disk"""
    def __init__(self, filepath):
        self.f = open(filepath, 'w')
        self.fid = self.f.fileno()
        self.filepath = filepath

    def close(self):
        self.f.close()

    def write(self, content):
        self.f.write(content)
        self.f.flush()
        os.fsync(self.fid)

    def write_buf(self, content):
        self.f.write(content)

    def print_and_write(self, content):
        print(content)
        self.write(content+'\n')


def print_section(content, up=True, down=True):
    if up:
        print('='*len(content))
    elif up > 0:
        print('='*up)
    print(content)
    if down:
        print('='*len(content))
    elif down > 0:
        print('='*down)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    num = output.size(1)
    target_topk = []
    appendices = []
    for k in topk:
        if k <= num:
            target_topk.append(k)
        else:
            appendices.append([0.0])
    topk = target_topk
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res + appendices


# Custom progress bar
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 40.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_model_param(model):
    # import numpy as np
    # n_param = 0
    # for i in model.parameters():
    #     if hasattr(i, 'require_grad') and i.require_grad:
    #         p_size = i.size()
    #         n_param += np.prod(p_size)
    # return n_param

    import operator
    import functools
    return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])


def get_circulant_model_param(model):
    import numpy as np
    n_param = 0
    for i in model.parameters():
        p_size = i.size()
        if len(p_size) == 4 and p_size[2] == 1 and p_size[3] == 1:  # 1*1 conv
            n_param += p_size[0]
        else:
            n_param += np.prod(p_size)
    return n_param


def get_toeplitz_model_param(model):
    import numpy as np
    n_param = 0
    for i in model.parameters():
        p_size = i.size()
        if len(p_size) == 4 and p_size[2] == 1 and p_size[3] == 1:  # 1*1 conv
            n_param += p_size[0] * 2 - 1
        else:
            n_param += np.prod(p_size)
    return n_param


def get_bernoulli_diagonal(dim):
    m = torch.diag(torch.ones(dim)) * 0.5
    m = torch.bernoulli(m)
    for i in range(dim):
        m[i, i] = m[i, i] * 2 - 1
    return m.float()


def get_normalized_hadamard(dim):
    import scipy.linalg as la
    h = la.hadamard(dim) * pow(dim, -0.5)
    return torch.from_numpy(h).float()


def get_random_projection(inp, oup):
    return torch.FloatTensor(oup, inp).normal_(0, math.sqrt(2. / inp))


def get_sparse_random_projection(inp, oup):
    import numpy as np
    filled = torch.FloatTensor(oup, inp).normal_(0, math.sqrt(2. / inp))
    q = (np.log(128) ** 2 / inp).item()
    w = torch.FloatTensor(oup, inp).zero_()
    r = torch.rand(oup, inp)
    w[r < q] = filled[r < q]

    return w


def get_right_inverse(w):
    return torch.mm(w.t(), torch.inverse(w.mm(w.t())))


def get_rht_mat(cin, cout):
    return get_random_projection(cin, cout).mm(get_normalized_hadamard(cin)).mm(
        get_bernoulli_diagonal(cin))


def get_srht_mat():
    return get_sparse_random_projection(inp, inp / n_div).mm(get_normalized_hadamard(inp)).mm(
        get_bernoulli_diagonal(inp))


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
from torch.autograd import Variable

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


# logging
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    import operator
    import functools

    return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])


def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    # ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel() / x.size(0)
        delta_params = get_layer_param(layer)

    # ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = weight_ops + bias_ops
        delta_params = get_layer_param(layer)

    # ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    # unknown layer type
    else:
        delta_params = get_layer_param(layer)

    count_ops += delta_ops
    count_params += delta_params

    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W)).cuda()

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params


def measure_layer_for_pruning(layer, x):
    multi_add = 1
    type_name = get_layer_info(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        layer.flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        layer.params = get_layer_param(layer)

    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        layer.flops = weight_ops + bias_ops
        layer.params = get_layer_param(layer)

    return


def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5, temperature=5.):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    import torch.nn as nn
    import torch.nn.functional as F
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/temperature, dim=1),
                             F.softmax(teacher_outputs/temperature, dim=1)) * (alpha * temperature * temperature) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


# the original kl diviergence has changed
class KLloss(torch.nn.modules.loss._Loss):
    """The KL-Divergence loss for the model and refined labels output.
    output must be a pair of (model_output, refined_labels), both NxC tensors.
    The rows of refined_labels must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, output, target):
        assert output.size() == target.size(), "output must a pair of tensors of same size."

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        model_output, refined_labels = output, target
        if refined_labels.requires_grad:
            raise ValueError("Refined labels should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        del model_output

        # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
        # for batch matrix multiplicatio
        refined_labels = refined_labels.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(refined_labels, model_output_log_prob)
        if self.size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        return cross_entropy_loss


def convert_to_one_hot(y, n_class=10):
    return np.eye(n_class)[y.reshape(-1)].astype(np.float32)
