import numpy as np

import torch
import torch.nn as nn

import math
import random


def prune_select_layer_simple(model, x, given_index, sparsity, prune_method='activation', recon_method='least_square'):
    '''
    prune filters in a conv layer with given index and given sparsity
    model: the model class, requires that it has members: features, conv_index
    x: a batch of input images
    given_index: index for the conv layer to prune
    sparsity: sparsity for the conv layer to prune
    '''

    for forward_index in range(0, given_index):
        x = model.features[forward_index](x)
    x_next = model.features[given_index](x)
    x_buffer = x_next
    # forward to get the feature map before conv layer next(used in calibration)
    next_conv_index = model.conv_index[model.conv_index.index(given_index) + 1]
    for forward_index in range(given_index + 1, next_conv_index + 1):
        x_next = model.features[forward_index](x_next)
    # find channel to prune
    # 0. x: the feature map before the current conv layer
    # 1. x_buffer: the feature map output by the conv layer to be pruned
    # 2. x_next: the feature map output by the next conv layer
    channel_index_left = find_prune_channel(model, x_buffer, x_next, given_index, next_conv_index, sparsity, prune_method)
    del x_buffer
    net_surgery(model, given_index, model.conv_index[model.conv_index.index(given_index) + 1], channel_index_left)

    if recon_method == 'least_square':  # the original least square method
        x = model.features[given_index](x)  # TODO: maybe possible to merge this command with the next ones
        for forward_index in range(given_index + 1, next_conv_index):
            x = model.features[forward_index](x)  # NOTE: conv is the feature map before the next conv layer
        # reconstruct conv filter
        # solve_conv_kernel_numpy(model, x, x_next, next_conv_index) # use cpu implementation to allow a larger batch
        solve_conv_kernel_torch(model, x, x_next, next_conv_index)
    elif recon_method == 'bp_current':  # my gradient based method
        x_before_current = x.clone()
        adjust_current_kernel(model, x_before_current, x_next, given_index, next_conv_index)
    elif recon_method == 'hybrid':
        x_before_current = x.clone()
        x = model.features[given_index](x)  # TODO: maybe possible to merge this command with the next ones
        for forward_index in range(given_index + 1, next_conv_index):
            x = model.features[forward_index](x)  # NOTE: conv is the feature map before the next conv layer
        solve_conv_kernel_torch(model, x, x_next, next_conv_index)
        adjust_current_kernel(model, x_before_current, x_next, given_index, next_conv_index)
    else:
        raise Exception('Not supported!')

def prune_select_layer(model, x, given_index, sparsity, prune_method='activation', recon_method='least_square'):
    '''
    prune filters in a conv layer with given index and given sparsity
    model: the model class, requires that it has members: features, conv_index
    x: a batch of input images
    given_index: index for the conv layer to prune
    sparsity: sparsity for the conv layer to prune
    '''

    for forward_index in range(0, given_index):
        x = model.features[forward_index](x)
    x_next = model.features[given_index](x)
    x_buffer = x_next
    # forward to get the feature map before conv layer next(used in calibration)
    next_conv_index = model.conv_index[model.conv_index.index(given_index) + 1]
    for forward_index in range(given_index + 1, next_conv_index + 1):
        x_next = model.features[forward_index](x_next)
    # find channel to prune
    # 0. x: the feature map before the current conv layer
    # 1. x_buffer: the feature map output by the conv layer to be pruned
    # 2. x_next: the feature map output by the next conv layer
    channel_index_left = find_prune_channel(model, x_buffer, x_next, given_index, next_conv_index, sparsity, prune_method)
    del x_buffer
    net_surgery(model, given_index, model.conv_index[model.conv_index.index(given_index) + 1], channel_index_left)

    if recon_method == 'least_square':  # the original least square method
        x = model.features[given_index](x)  # TODO: maybe possible to merge this command with the next ones
        for forward_index in range(given_index + 1, next_conv_index):
            x = model.features[forward_index](x)  # NOTE: conv is the feature map before the next conv layer
        # reconstruct conv filter
        # solve_conv_kernel_numpy(model, x, x_next, next_conv_index) # use cpu implementation to allow a larger batch
        solve_conv_kernel_torch(model, x, x_next, next_conv_index)
    elif recon_method == 'bp_current':  # my gradient based method
        x_before_current = x.clone()
        adjust_current_kernel(model, x_before_current, x_next, given_index, next_conv_index)
    elif recon_method == 'hybrid':
        x_before_current = x.clone()
        x = model.features[given_index](x)  # TODO: maybe possible to merge this command with the next ones
        for forward_index in range(given_index + 1, next_conv_index):
            x = model.features[forward_index](x)  # NOTE: conv is the feature map before the next conv layer
        solve_conv_kernel_torch(model, x, x_next, next_conv_index)
        adjust_current_kernel(model, x_before_current, x_next, given_index, next_conv_index)
    else:
        raise Exception('Not supported!')

def find_prune_channel(model, conv_feature, conv_feature_next, conv_index, conv_index_next, sparsity, prune_method):
    '''
    find channels to prune with a given metric

    conv_featrue: feature map of the layer to prune
    conv_feature_next: feature map of the next conv layer (cuz we find layer to prune based on activation of the layer next)
    conv_index: index of the layer to prune
    conv_index_next: index of the next conv layer
    sparsity: sparsity assigned

    prune_method: will be elaborated later
    'random': we choose kernel to prune randomly
    'weight': we sorted importance of kernels by l2 norm of each kernel
    'greedy': we find one kernel contributed to the smallest activation first, and then find the next, and so on ...
    'activation': we sorted importance of kernels by l2 norm of activation of next layer
    'sample': we sample given number of possible combinations, and find the one with least activation difference of next layer
    '''

    assert prune_method in ['random', 'weight', 'greedy', 'activation', 'sample', 'activation_remain']
    channel_num = conv_feature.size(1)
    prune_num = int(math.floor(channel_num * sparsity))
    channel_index_to_prune = []
    if prune_method == 'greedy':
        while len(channel_index_to_prune) < prune_num:
            min_diff = 1e10
            min_index = 0
            for index in range(channel_num):
                if index in channel_index_to_prune:
                    continue
                index_to_try = channel_index_to_prune + [index]
                sliced_feature = conv_feature.clone()
                zero_index = list(set(range(channel_num)) - set(index_to_try))
                sliced_feature[:, zero_index, ...] = 0
                for forward_index in range(conv_index + 1, conv_index_next + 1):
                    sliced_feature = model.features[forward_index](sliced_feature)
                sliced_norm = sliced_feature.norm(2)
                if (sliced_norm.data < min_diff).all():  # TODO: .all() may not be necessary here
                    min_diff = sliced_norm.data
                    min_index = index
            channel_index_to_prune += [min_index]
    elif prune_method == 'activation':  # calculate the activation for each kernel
        norm_list = list()
        for index in range(channel_num):
            sliced_feature = conv_feature.clone()
            zero_index = list(set(range(channel_num)) - set([index]))
            sliced_feature[:, zero_index, ...] = 0
            for forward_index in range(conv_index + 1, conv_index_next + 1):
                sliced_feature = model.features[forward_index](sliced_feature)
            sliced_norm = sliced_feature.norm(2)
            norm_list.append(sliced_norm.data.cpu().numpy())
        norm_list = np.hstack(norm_list)
        sorted_index = np.argsort(norm_list, axis=0)
        channel_index_to_prune = sorted_index[0:prune_num]
    elif prune_method == 'activation_remain':  # calculate the activation for each kernel
        norm_list = list()
        for index in range(channel_num):
            sliced_feature = conv_feature.clone()
            zero_index = [index]
            sliced_feature[:, zero_index, ...] = 0
            for forward_index in range(conv_index + 1, conv_index_next + 1):
                sliced_feature = model.features[forward_index](sliced_feature)
            sliced_norm = (sliced_feature - conv_feature_next).norm(2)
            norm_list.append(sliced_norm.data.cpu().numpy())
        norm_list = np.hstack(norm_list)
        sorted_index = np.argsort(norm_list, axis=0)
        channel_index_to_prune = sorted_index[0:prune_num]
    elif prune_method == 'weight':
        conv_weight = model.features[conv_index].weight.data
        conv_weight_flatten = conv_weight.transpose(0, 1).contiguous().view(channel_num, -1)
        filter_norm = conv_weight_flatten.norm(2, dim=1)
        _, sorted_index = torch.sort(filter_norm)
        channel_index_to_prune = sorted_index[0:prune_num].cpu().numpy()
        del conv_weight
    elif prune_method == 'random':
        channel_index_to_prune = random.sample(range(channel_num), prune_num)
    elif prune_method == 'sample':
        min_diff = 1e10
        for i in range(1000):
            zero_index_to_try = random.sample(range(channel_num), prune_num)
            sliced_feature = conv_feature.clone()
            sliced_feature[:, zero_index_to_try, ...] = 0
            for forward_index in range(conv_index + 1, conv_index_next + 1):
                sliced_feature = model.features[forward_index](sliced_feature)
            sliced_norm = (sliced_feature - conv_feature_next).norm(2)
            if (sliced_norm.data < min_diff).all():
                min_diff = sliced_norm.data
                channel_index_to_prune = zero_index_to_try
    channel_index_left = list(set(range(channel_num)) - set(channel_index_to_prune))
    return channel_index_left


def net_surgery(model, conv_index, conv_index_next, channel_index_left):
    '''
    delete redundant weigths given channel index left

    conv_index: conv_index of the layer we are pruning
    conv_index_next: conv_index of the next conv layer (it also contains redundant weights)
    channel_index_left: channels we want to preserve
    '''
    # TODO: Note that this function only prunes the current conv layer and the next conv layer, only supports
    # TODO: ReLU intermediate layer, not BN etc.
    conv_kernel = model.features[conv_index]
    conv_kernel_next = model.features[conv_index_next]
    conv_weight = conv_kernel.weight.data
    conv_bias = conv_kernel.bias.data
    conv_next_weight = conv_kernel_next.weight.data
    conv_next_bias = conv_kernel_next.bias.data
    conv_weight = conv_weight[channel_index_left, ...]
    conv_bias = conv_bias[torch.cuda.LongTensor(channel_index_left)]
    conv_next_weight = conv_next_weight[:, channel_index_left, ...]
    model.features[conv_index] = nn.Conv2d(conv_weight.size(1), conv_weight.size(0), kernel_size=3, padding=1)
    del model.features[conv_index].weight
    model.features[conv_index].weight = nn.Parameter(conv_weight)
    del model.features[conv_index].bias
    model.features[conv_index].bias = nn.Parameter(conv_bias)
    model.features[conv_index_next] = nn.Conv2d(conv_next_weight.size(1), conv_next_weight.size(0), kernel_size=3, padding=1)
    del model.features[conv_index_next].weight
    model.features[conv_index_next].weight = nn.Parameter(conv_next_weight)
    del model.features[conv_index_next].bias
    model.features[conv_index_next].bias = nn.Parameter(conv_next_bias)
    del conv_kernel
    del conv_kernel_next


def solve_conv_kernel_torch(model, X, Y, conv_index):
    '''
    calibrationi using torch functionanlity (running on GPU)
    X: input of conv kernel
    Y: output of conv kernel
    conv_index: the index we are pruning
    '''
    # X shape: [bs, c_in, h, w]
    # Y shape: [bs, c_out, h, w]
    # im2col shape: [bs * h * w, c_in * k * k]
    print('X shape: {}, y shape: {}'.format(X.size(), Y.size()))
    this_X = im2col_X(model, X, conv_index)
    print('Im2Col shape: {}'.format(this_X.size()))
    bias = model.features[conv_index].bias.data
    this_Y = Y.data
    this_Y = this_Y.permute(0, 2, 3, 1)
    this_Y = this_Y.contiguous().view(-1, this_Y.size(3))
    this_Y_no_bias = this_Y - bias.view(1, -1) # we don't solve bias term

    W, _ = torch.gels(this_Y_no_bias, this_X) # torch function to solve least square problem
    W = W[0:this_X.size(1), :]  # Pytorch bug, size does not match
    W = W.transpose(0, 1).contiguous().view(model.features[conv_index].weight.data.size())
    del model.features[conv_index].weight
    del this_X, this_Y, this_Y_no_bias
    model.features[conv_index].weight = nn.Parameter(W)


def solve_conv_kernel_numpy(model, X, Y, conv_index):
    '''
    calibrationi using numpy functionanlity (running on CPU)
    X: input of conv kernel
    Y: output of conv kernel
    conv_index: the index we are pruning
    '''

    N, C, H, W = X.size()
    weight = model.features[conv_index].weight.data
    bias = model.features[conv_index].bias.data.cpu().numpy()
    F, _, HH, WW = weight.size()
    stride = model.features[conv_index].stride[0]
    pad = model.features[conv_index].padding[0]

    # Check dimensions
    assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    X = X.data.cpu().numpy()
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = X.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
    X_cols = np.ascontiguousarray(x_stride)
    X_cols.shape = (C * HH * WW, N * out_h * out_w)

    # Now we generate matrix of Y
    Y = Y.data.cpu().numpy()
    Y = Y.transpose(1, 0, 2, 3)
    Y = Y.reshape(Y.shape[0], -1)
    Y_no_bias = Y - bias.reshape(-1, 1)

    W = np.linalg.lstsq(X_cols.T, Y_no_bias.T)[0].T

    W = W.reshape(weight.size())
    del model.features[conv_index].weight
    model.features[conv_index].weight = nn.Parameter(torch.from_numpy(W).cuda())


def im2col_X(model, X, conv_index):
    # GPU implementation of im2col
    import pyinn as P
    this_X = P.im2col(X, model.features[conv_index].kernel_size[0], model.features[conv_index].stride[0], model.features[conv_index].padding[0]).detach()
    this_X_data = this_X.data
    this_X_data = this_X_data.permute(0, 4, 5, 1, 2, 3)
    this_X_data_flatten = this_X_data.contiguous().view(this_X_data.size(0) * this_X_data.size(1) * this_X_data.size(2), -1)
    del this_X, this_X_data
    return this_X_data_flatten


# adjust current kernel with back-propagation
def adjust_current_kernel(model, x_before_current, x_after_next, given_index, next_conv_index, n_iter=1000):
    init_lr = 1e-3
    optimizer = torch.optim.SGD([model.features[given_index].weight, model.features[given_index].bias], init_lr,
                                momentum=0.9,
                                weight_decay=0)
    ori_weight = model.features[given_index].weight.clone()
    ori_bias = model.features[given_index].bias.clone()
    criterion = nn.MSELoss().cuda()
    for iter in range(n_iter):
        optimizer.zero_grad()
        x = model.features[given_index](x_before_current.detach())
        for i in range(given_index + 1, next_conv_index + 1):
            x = model.features[i](x)
        diff_loss = criterion(x, x_after_next.detach())
        change_loss = criterion(model.features[given_index].weight, ori_weight.detach()) + \
                      criterion(model.features[given_index].bias, ori_bias.detach())
        loss = diff_loss + 0.05 * change_loss
        if iter == 0 or iter == n_iter - 1:
            print('==> reconstruction loss: ', loss.data.cpu().numpy()[0])

        loss.backward()
        optimizer.step()

        if iter < n_iter / 3.:
            current_lr = init_lr
        elif iter < n_iter * 2 / 3.:
            current_lr = init_lr * 0.1
        else:
            current_lr = init_lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr


def get_density_list(model, prunable_idx):
    density_list = []
    total_dense = 0.
    total_weight = 0.
    modules = list(model.modules())
    for i in prunable_idx:
        m = modules[i]
        density = 1 - torch.sum(m.weight.data.eq(0).float()) / m.weight.data.numel()
        total_dense += density * m.weight.data.numel()
        total_weight += m.weight.data.numel()
        density_list.append(density)
    return density_list, total_dense * 1. / total_weight


# fine-grained pruning
def prune_fine_grained(model, prunable_idx, preserve_ratio_list, criteria='normal', importances=None):
    if criteria=='importance':
        assert importances is not None, 'Please provide weight importance'
    mask_list = []
    modules = list(model.modules())
    # print(preserve_ratio_list)
    for i, preserve_ratio in zip(prunable_idx, preserve_ratio_list):
        m = modules[i]
        # mask = prune_element_wise_in_place(m.weight.data, 1. - preserve_ratio)

        w_size = m.weight.data.numel()
        n_preserve = int(preserve_ratio * w_size)
        if n_preserve == 0:
            n_preserve = 1

        if criteria == 'normal':
            significance = m.weight.data ** 2
        elif criteria == 'importance':  # here we save the expectations in gradient
            significance = importances[i] * m.weight.data**2  # m.weight.data ** 2 * m.weight.grad.data ** 2
        else:
            raise NotImplementedError
        val, idx = torch.topk(significance.view(-1), n_preserve)
        threshold = val[-1]
        mask = significance >= threshold  # ByteTensor to reduce storage
        m.weight.data[significance < threshold] = 0.  # prune away the small values

        mask_list.append(mask)
    return mask_list


def get_sparsity(t):
    # get sparsity of tensor t
    return torch.sum(t.ne(0)).item() * 1. / t.numel()


def least_square_sklearn(X, Y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    return reg.coef_


def prepare_model_for_maintain_mask(model):
    def mask_weight(self, input):
        self.weight.data *= self.mask
    n_nz = 0
    n_weights = 0
    for m in model.modules():
        if type(m) in [nn.Conv2d, nn.Linear]:
            mask = (~m.weight.data.eq(0.)).float()
            m.mask = mask.cuda()
            m.register_forward_pre_hook(mask_weight)
            n_nz += torch.sum(mask)
            n_weights += mask.numel()
    return 1 - n_nz * 1. / n_weights
