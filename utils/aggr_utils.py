import numpy as np

import torch
import torch.nn as nn


def aggr_by_one(model, index_list=None):
    if not hasattr(model, 'aggr_mask'):
        model.aggr_mask = dict()
    if index_list is None:
        index_list = model.conv_index[1:-1]
    for ind in index_list:
        W = model.features[ind].weight.data
        W_arr = W.cpu().numpy()
        if ind not in model.aggr_mask.keys():
            model.aggr_mask[ind] = np.ones_like(W_arr)
        ch_out, ch_in, ksize, _ = W_arr.shape
        for i in range(ch_out):
            for j in range(ch_in):
                this_kernel = np.squeeze(np.abs(W_arr[i, j, ...]))
                this_kernel[this_kernel == 0] = 1000.
                m_ind = np.argmin(this_kernel)
                m_row = int(m_ind / ksize)
                m_col = m_ind % ksize
                W_arr[i, j, m_row, m_col] = 0.
                model.aggr_mask[ind][i, j, m_row, m_col] = 0.
        model.features[ind].weight = nn.Parameter(torch.from_numpy(W_arr).cuda())


def mask_aggr_gradient(model, index_list=None):
    if index_list is None:
        index_list = model.conv_index[1:-1]  # we do not operate on the first and last conv layer

    for ind in index_list:
        if ind not in model.aggr_mask.keys():  # not yet aggr.
            continue
        # print(type(self.features[ind].weight.grad))
        # print(type(self.aggr_mask[ind]))
        mask = model.aggr_mask[ind]
        if type(mask) == np.ndarray:
            mask = torch.from_numpy(mask).cuda()
        model.features[ind].weight.grad.data = torch.mul(model.features[ind].weight.grad.data, mask)


def aggr_select_layer(model, index, aggr_method='max', mode='cpu', get_mask=False):
    if not hasattr(model, 'aggr_mask'):
        model.aggr_mask = dict()
    W = model.features[index].weight.data
    if mode == 'cpu':
        W_arr = W.cpu().numpy()
        if get_mask:
            mask = np.zeros_like(W_arr)
        ch_out, ch_in, ksize, _ = W_arr.shape
        assert ksize == 3
        for i in range(ch_out):
            for j in range(ch_in):
                m_ind = np.argmax(np.abs(W_arr[i, j, ...]))
                m_row = int(m_ind / ksize)
                m_col = m_ind % ksize
                if aggr_method == 'max':
                    m_val = W_arr[i, j, m_row, m_col]
                elif aggr_method == 'sum':
                    m_val = np.sum(W_arr[i, j, ...])  # TODO
                elif aggr_method == 'weighted':
                    ss_x = 0.
                    ss_y = 0.
                    for k_i in range(ksize):
                        for k_j in range(ksize):
                            ss_x += k_i * W_arr[i, j, k_i, k_j]
                            ss_y += k_j * W_arr[i, j, k_i, k_j]
                    ss_x /= np.sum(W_arr[i, j, ...])
                    ss_y /= np.sum(W_arr[i, j, ...])
                    m_row = int(round(ss_x))
                    m_col = int(round(ss_y))
                    m_val = np.sum(W_arr[i, j, ...])
                else:
                    raise NotImplementedError

                if get_mask:
                    mask[i, j, m_row, m_col] = 1.  # only largest value is preseved
                W_arr[i, j, ...] = np.zeros([ksize, ksize])
                W_arr[i, j, m_row, m_col] = m_val
        del model.features[index].weight
        model.features[index].weight = nn.Parameter(torch.from_numpy(W_arr).cuda())
        if get_mask:
            model.aggr_mask[index] = torch.from_numpy(mask).cuda()

    else:
        raise NotImplementedError


def reset_aggr_mask(model):
    if hasattr(model, 'aggr_mask'):
        del model.aggr_mask
    model.aggr_mask = dict()
