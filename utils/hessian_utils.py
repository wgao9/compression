from itertools import product
import numpy as np
import torch
from torch.autograd import grad

import torch.multiprocessing as mp
import functools


def all_indices(a):
    return product(*[range(s) for s in a.shape])


def flat_total_derivative(output, input_, create_graph=True):
    derivatives = []

    for i, ind in enumerate(all_indices(output)):
        z = torch.zeros_like(output)
        z[ind] = 1.0

        d, = grad(output, input_, grad_outputs=z, create_graph=create_graph)

        # d, = grad(output[ind], input_, create_graph=create_graph)

        derivatives.append(d)

    return torch.stack(derivatives, dim=0).reshape(output.shape + input_.shape)


def total_derivative(output, input_, create_graph=True):
    return flat_total_derivative(output, input_)


def diagonal_contraction(a, b, dims=2):
    # contracts the first dims dimensions
    assert a.shape[:dims] == b.shape[:dims]
    assert a.shape[dims:] == b.shape[dims:]

    m = torch.mul(a, b)

    result = torch.sum(m, dim=list(range(a.dim()))[:dims])
    return result

# FIXME: missing a term
# FIXME: working only for 2d tensors in path, need to fix wherever tensordot_pytorch or torch.permute is used
# FIXME: divide by the batch size
def flat_hessian(f, path, w, diagonal=False):
    # path is a list of intermediate tensors from f to w
    # it should not include f or w
    if len(path) == 0:
        raise NotImplementedError

    # the middle factor
    z = path[0]
    fz = total_derivative(f, z)
    fzz = total_derivative(fz, z)

    # the right factor
    pathw = path + [w] # length >= 2
    zw = total_derivative(pathw[0], pathw[1])
    for i in range(len(path))[1:]:
        next_ = total_derivative(pathw[i], pathw[i+1])
        zw = tensordot_pytorch(zw, next_)


    # FIXME: divide by batch_size
    fzw = tensordot_pytorch(fzz, zw)

    if diagonal:
        # FIXME: divide by batch_size
        flattened = diagonal_contraction(fzw, zw)

        return flattened.reshape(w.shape)
    else:
        # full hessian
        a = range(z.dim())
        fww = tensordot_pytorch(fzw, zw, axes=[a, a])

        return fww


def diagonal_hessian(f, path, w):
    return flat_hessian(f, path, w, diagonal=True)


# FIXME: missing a term
def _diagonal_hessian_multi_inner(fzz, z, w):
    with torch.no_grad():
        zw = total_derivative(z, w, create_graph=False)
        fzw = tensordot_pytorch(fzz, zw)
        fww_diagonal = diagonal_contraction(fzw, zw)

    # TODO: add fz * diagonal(zww)

    assert fww_diagonal.shape == w.shape

    return fww_diagonal


def diagonal_hessian_multi(f, z, ws):
    fz = total_derivative(f, z)
    with torch.no_grad():
        fzz = total_derivative(fz, z, create_graph=False).detach()

    fww_diagonals = []

    for w in ws:
        fww_diagonal = _diagonal_hessian_multi_inner(fzz, z, w)
                
        assert fww_diagonal.shape == w.shape

        fww_diagonals.append(fww_diagonal.detach())

    return fww_diagonals


def _diagonal_hessian_multi(f, z, ws, trained=True):
    fz = total_derivative(f, z)
    with torch.no_grad():
        fzz = total_derivative(fz, z, create_graph=False).detach()

    fww_diagonals = []

    for w in ws:
        # fww_diagonal = _diagonal_hessian_multi_inner(fz, z, w)
        zw = total_derivative(z, w)
        with torch.no_grad(): 
            zw_detached = zw.detach()
            fzw = tensordot_pytorch(fzz, zw_detached)
            fww_diagonal = diagonal_contraction(fzw, zw_detached)

        # if trained then fz is assumed to be negligible
        # this is VERY slow
        # NOTE: if z(w) factors through a ReLU then this additional term is not needed.
        if not trained:
            # calculate fz * diagonal(zww)
            zww_diagonal = []
            w_dim = w.dim()
            for zw_ind in all_indices(zw):
                w_ind = zw_ind[-w_dim:]
                
                zz = torch.zeros_like(zw)
                zz[zw_ind] = 1.0

                # shape = w.shape
                zww_slice, = grad(zw, w, grad_outputs=zz, create_graph=True)

                # import ipdb; ipdb.set_trace()

                zww_entry = zww_slice[w_ind]
                zww_diagonal.append(zww_entry.item())

            zww_diagonal = torch.tensor(zww_diagonal).reshape(zw.shape)

            a = range(z.dim())
            fww_diagonal += tensordot_pytorch(zw, zww_diagonal, axes=[a, a])
                
        assert fww_diagonal.shape == w.shape

        fww_diagonals.append(fww_diagonal.detach())

    return fww_diagonals


# from: https://gist.github.com/deanmark/9aec75b7dc9fa71c93c4bc85c5438777
def tensordot_pytorch(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)
