import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import time
# import tensorflow as tf

import torch
import torch.nn as nn

from utils.utils import AverageMeter


def k_means_cpu(weight, n_clusters, init='linear', max_iter=50):
    '''
    Run k-means on weights of models.
    :param weight: the given weight to quantize
    :param n_clusters: number of centroids (k in k-means)
    :param init: ways to init: [linear, random, density]
    :param max_iter: max iteration of running k-means, default to 300
    :return: centroids and labels
    '''
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature

    if init == 'linear':  # using the linear initial centroids
        linspace = np.linspace(weight.min(), weight.max(), n_clusters + 1)
        init_centroids = [(linspace[i] + linspace[i + 1]) / 2. for i in range(n_clusters)]
        init_centroids = np.array(init_centroids).reshape(-1, 1)  # single feature
    elif init == 'random':
        init_centroids = 'random'
    elif init == 'density':
        raise NotImplementedError  # TODO
    else:
        raise NotImplementedError

    k_means = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1, max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).cuda().view(1, -1), torch.from_numpy(labels).int().cuda()

def weighted_k_means_cpu(weight, importance, hessian, n_clusters, init='linear', max_iter=50, ha=0.0, entropy_reg=0.0, diameter_reg=0.0, diameter_entropy_reg=0.0):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    n_weights = weight.size
    weight = weight.reshape(-1)  # single feature
    importance = importance.reshape(-1)
    hessian = hessian.reshape(-1)
    is_quartic = True if ha > 0.0 else False
    is_entropy_reg = True if entropy_reg > 0.0 else False
    is_diameter_reg = True if diameter_reg > 0.0 else False
    is_diameter_entropy_reg = True if diameter_entropy_reg > 0.0 else False

    if init == 'linear':  # using the linear initial centroids
        linspace = np.linspace(weight.min(), weight.max(), n_clusters + 1)
        centroids = [(linspace[i] + linspace[i + 1]) / 2. for i in range(n_clusters)]
        centroids = np.array(centroids).reshape(-1)  # single feature
    elif init == 'quantile':
        centroids = [np.quantile(weight, 0.5*(2*i+1)/n_clusters) for i in range(n_clusters)]
        centroids = np.array(centroids).reshape(-1)  # single feature
    else:
        raise NotImplementedError

    #precomputed features
    importance_weight = np.multiply(importance, weight)
    hessian_2 = np.multiply(hessian, hessian)
    hessian_2_weight = np.multiply(hessian_2, weight)
    hessian_2_weight_2 = np.multiply(hessian_2_weight, weight)
    hessian_2_weight_3 = np.multiply(hessian_2_weight_2, weight)
    theta = np.log(1./n_clusters)*np.ones(n_clusters)
    labels = np.zeros(n_weights, dtype='int')

    for iters in range(max_iter):
        labels = np.zeros(n_weights, dtype='int')
        coeffs = np.zeros((n_clusters, 4))

        #compute labels
        for j in range(n_clusters-1):
            criteria = 2*(centroids[j+1]-centroids[j])*importance_weight - (centroids[j+1]**2 - centroids[j]**2)*importance
            thr = 0
            if is_entropy_reg:
                thr += entropy_reg*(theta[j+1]-theta[j])
            elif is_diameter_entropy_reg:
                thr += diameter_entropy_reg*(theta[j+1]-theta[j])*(centroids[-1]-centroids[0])**2 
            if is_quartic:
                criteria += ha*(4*(centroids[j+1]-centroids[j])*hessian_2_weight_3 - 6*(centroids[j+1]**2-centroids[j]**2)*hessian_2_weight_2 + 4*ha*(centroids[j+1]**3-centroids[j]**3)*hessian_2_weight - (centroids[j+1]**4-centroids[j]**4)*hessian_2)

            leq_j = (criteria < thr).astype(int)

            labels += 1-leq_j

            #compute coefficients
            coeffs[j,0] -= 2*np.inner(importance_weight, leq_j)
            coeffs[j,1] += 2*np.inner(importance, leq_j)
            if is_quartic:
                coeffs[j,0] -= 4*ha*np.inner(hessian_2_weight_3, leq_j)
                coeffs[j,1] += 12*ha*np.inner(hessian_2_weight_2, leq_j)
                coeffs[j,2] -= 12*ha*np.inner(hessian_2_weight, leq_j)
                coeffs[j,3] += 4*ha*np.inner(hessian_2, leq_j)

        leq_minus_1 = np.ones(n_weights, dtype='int')
        coeffs[-1,0] -= 2*np.inner(importance_weight, leq_minus_1)
        coeffs[-1,1] += 2*np.inner(importance, leq_minus_1)
        if is_quartic:
            coeffs[-1,0] -= 4*ha*np.inner(hessian_2_weight_3, leq_minus_1)
            coeffs[-1,1] += 12*ha*np.inner(hessian_2_weight_2, leq_minus_1)
            coeffs[-1,2] -= 12*ha*np.inner(hessian_2_weight, leq_minus_1)
            coeffs[-1,3] += 4*ha*np.inner(hessian_2, leq_minus_1)

        for j in range(n_clusters-1):
            for p in range(4):
                coeffs[n_clusters-1-j,p] = coeffs[n_clusters-1-j,p] - coeffs[n_clusters-2-j,p]
        
        #compute new cluster centers
        for j in range(n_clusters):
            if is_quartic and coeffs[j,3] > 0.0:
                root = np.roots([coeffs[j,3], coeffs[j,2], coeffs[j,1], coeffs[j,0]])
                centroids[j] = np.real(root[2])
            elif coeffs[j,1] > 0.0:
                centroids[j] = -coeffs[j,0]/coeffs[j,1]

        #compute new thetas for entropy_reg and diameter_entropy_reg
        entropy = 0.0
        if is_entropy_reg or is_diameter_entropy_reg:
            probs = np.bincount(labels, minlength=n_clusters)
            for j in range(n_clusters):
                theta[j] = np.log(n_weights+n_clusters) - np.log(probs[j]+1)
                entropy += (probs[j]+1)*theta[j]/(n_weights+n_clusters)

        #update leftmost and rightmost centroids for diameter_reg or diameter_entropy_reg
        if is_diameter_reg and n_clusters > 1:
            centroids[0] += 0.5*diameter_reg/coeffs[0,1]
            if centroids[0] > centroids[1]:
                centroids[0] = centroids[1]-1e-8
            centroids[-1] -= 0.5*diameter_reg/coeffs[-1,1]
            if centroids[-1] < centroids[-2]:
                centroids[-1] = centroids[-2]+1e-8
        elif is_diameter_entropy_reg and n_clusters > 1:
            new_cen_0 = (-coeffs[0,0]+2*diameter_entropy_reg*entropy*centroids[-1])/(coeffs[0,1]+2*diameter_entropy_reg*entropy)
            new_cen_minus_1 = (-coeffs[-1,0]+2*diameter_entropy_reg*entropy*centroids[0])/(coeffs[-1,1]+2*diameter_entropy_reg*entropy)
            if new_cen_0 < centroids[1]:
                centroids[0] = new_cen_0
            else:
                centroids[0] = centroids[1]-1e-8
            if new_cen_minus_1 > centroids[-2]:
                centroids[-1] = new_cen_minus_1
            else:
                centroids[-1] = centroids[-2]+1e-8


    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).cuda().view(1, -1), torch.from_numpy(labels).int().cuda()

def k_means_torch(weight, n_clusters, init='linear', max_iter=50):
    org_shape = weight.size()
    weight = weight.view(-1, 1)  # single feature
    n_data_samples = weight.size(0)

    if init == 'linear':  # using the linear initial centroids
        linspace = np.linspace(weight.min(), weight.max(), n_clusters + 1)
        init_centroids = [(linspace[i] + linspace[i + 1]) / 2. for i in range(n_clusters)]
        init_centroids = np.array(init_centroids).reshape(-1, 1)  # single feature
    else:
        raise NotImplementedError

    # Initialize cluster means
    mean_vec = torch.from_numpy(init_centroids).cuda().float().view(1, -1)

    # Initialize variables
    u_zeros = torch.zeros(n_data_samples, n_clusters).cuda()  # Empty membership matrix

    for _ in range(max_iter):
        dist = weight - mean_vec
        # Per row of dist_sq get the index of the min value
        min_indices = torch.min((dist * dist), 1)[1]
        del dist

        # Change dim of tensor to n_data_samplesx1
        min_indices = min_indices.view(-1, 1)
        # Empty clusters could be present in membership matrix
        U = u_zeros.scatter_(1, min_indices, 1)
        # Empty Cluster Check
        empty_cluster_check = []  # number of data not belonging to this cluster
        for i in range(n_clusters):
            n_zero_elem_i = n_data_samples - torch.numel(torch.nonzero(U[:, i]))
            empty_cluster_check.append(n_zero_elem_i)

        del U
        # START OF CPU
        empty_indices = np.argwhere(np.array(empty_cluster_check) == n_data_samples)  # no data belong to the cluster
        len_empty_indices = len(empty_indices)
        min_indices_arr = min_indices.cpu().numpy()
        if len_empty_indices > 0:  # if one cluster has no corresponding data sample
            min_zeros = min(empty_cluster_check)  # has min zeros, which means has most points
            min_zeros_index = empty_cluster_check.index(min_zeros)
            extra_points = n_data_samples - min_zeros - 1  # number of elements - 1
            min_data_points = np.where(min_indices_arr == min_zeros_index)[0]
            if len_empty_indices <= extra_points:
                index = 0
                for i in empty_indices:
                    min_indices_arr[min_data_points[index]] = i[0]
                    index += 1
                    empty_cluster_check[i[0]] = empty_cluster_check[i[0]] - 1
                    empty_cluster_check[min_zeros_index] = empty_cluster_check[min_zeros_index] + 1
            else:
                pass  # very rare case, ignore
                # print("Error")
        # END OF CPU
        del empty_cluster_check
        min_indices = torch.from_numpy(min_indices_arr).cuda()
        min_indices = min_indices.view([n_data_samples, 1])

        # Update membership matrix, no empty clusters
        U = u_zeros.scatter_(1, min_indices, 1)
        del min_indices

        n_cluster_points = []
        # Get number of data points in each cluster
        for i in range(n_clusters):
            n_points = torch.numel(torch.nonzero(U[:, i]))
            n_cluster_points.append(n_points)
        # Create tensor
        n_cluster_points_tensor = torch.Tensor(n_cluster_points).cuda()
        del n_cluster_points

        # Update the means
        cluster_mat_col_wise_sum = torch.sum((U * weight), 0)
        new_means = torch.div(cluster_mat_col_wise_sum, n_cluster_points_tensor)
        del cluster_mat_col_wise_sum
        del n_cluster_points_tensor

        del mean_vec

        mean_vec = new_means.view(1, n_clusters)
        del new_means
    del u_zeros
    return mean_vec.view(1, -1), torch.from_numpy(min_indices_arr).cuda().view(org_shape)


def k_means_torch_compact(weight, n_clusters, init='linear', use_gpu=True):
    # a very simple linear nearest neighbor algorithm using PyTorch
    # NOTICE that the return is gpu variables
    org_shape = weight.size()
    weight = weight.view(-1, 1)  # single feature

    if init == 'linear':  # using the linear initial centroids
        linspace = np.linspace(weight.min(), weight.max(), n_clusters + 1)
        init_centroids = [(linspace[i] + linspace[i + 1]) / 2. for i in range(n_clusters)]
        init_centroids = np.array(init_centroids).reshape(-1, 1)  # single feature
    elif init == 'linear_include':
        linspace = np.linspace(weight.min(), weight.max(), n_clusters)
        init_centroids = np.array(linspace).reshape(-1, 1)
    elif init == 'linear_zero':  # using the linear initial centroids, and adjust to include zero
        linspace = np.linspace(weight.min(), weight.max(), n_clusters + 1)
        init_centroids = [(linspace[i] + linspace[i + 1]) / 2. for i in range(n_clusters)]
        init_centroids = np.array(init_centroids).reshape(-1, 1)  # single feature
        smallest_bias_ind = np.argmin(np.abs(init_centroids))
        init_centroids = init_centroids - init_centroids[smallest_bias_ind]
        assert 0 in init_centroids
    else:
        raise NotImplementedError

    # Initialize cluster means
    mean_vec = torch.from_numpy(init_centroids).float().view(1, -1)
    if use_gpu:
        mean_vec = mean_vec.cuda()
    else:  # run with cpu
        weight = weight.cpu()

    dist = weight - mean_vec
    # Per row of dist_sq get the index of the min value
    min_indices = torch.min((dist * dist), 1)[1]

    if not use_gpu:
        mean_vec = mean_vec.cuda()
        min_indices = min_indices.cuda()
    return mean_vec.view(1, -1), min_indices.view(org_shape)


def reconstruct_weight_from_k_means_result(centroids, labels):
    '''
    Construct the weight from centroids and labels obtained by k-means
    :param centroids:
    :param labels:
    :return:
    '''
    # weight = np.zeros_like(labels, dtype='float64')
    # for i, c in enumerate(np.squeeze(centroids)):
    #     weight[labels == i] = c
    # return weight

    if centroids.size()[1] == 1:
        value = centroids.cpu().numpy().item()
        print(value)
        weight = value*torch.ones_like(labels)
        return weight.float().cuda()

    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        # print(labels==i)
        weight[labels == i] = c.item()
    # print weight
    return weight


def fast_reconstruct_weight_from_k_means_result(centroids, ind_mats):
    '''
    Construct the weight from centroids and index matrix
    '''
    centroids_list = centroids.cpu().numpy().squeeze().tolist()
    # weight = torch.zeros_like(ind_mats[0]).float().cuda()
    # for c, ind_mat in zip(centroids_list, ind_mats):
    #     weight += c * ind_mat
    # return weight
    return sum([c * ind_mat for c, ind_mat in zip(centroids_list, ind_mats)])

def quantize_model(model, importances, hessians, quantize_index, quantize_clusters, max_iter=50, mode='cpu', quantize_bias=False, centroids_init='quantile', is_pruned=False, ha=0.0, entropy_reg=0.0, diameter_reg=0.0, diameter_entropy_reg=0.0):
    assert len(quantize_index) == len(quantize_clusters), \
        'You should provide the same number of bit setting as layer list!'
    assert importances is not None, 'Please provide weight importances'
    assert hessians is not None or ha==0.0, 'please provide hessians or set hessian to be zero'
    quantize_layer_cluster_dict = {n: b for n, b in zip(quantize_index, quantize_clusters)}
    centroid_label_dict = {}

    start_time = time.time()
    for i, layer in enumerate(model.modules()):
        if i not in quantize_index:
            continue
        this_cl_list = []
        n_cluster = quantize_layer_cluster_dict[i]
        if n_cluster < 1:  # if -1, do not quantize
            continue
        if type(n_cluster) == list:  # given both the bit of weight and bias
            assert len(n_cluster) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_cluster = [n_cluster, n_cluster]  # using same setting for W and b

        # quantize weight
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            importance = importances[i]
            hessian = hessians[i]
            nonzero_rate = importance.nonzero().size(0) * 100.0 / importance.numel()
            if is_pruned:
                nz_mask = w.ne(0)
                print('*** pruned density: {:.4f}'.format(torch.sum(nz_mask) * 1.0/ w.numel()))
                ori_shape = w.size()
                w = w[nz_mask]
                importance = importance[nz_mask]
                hessian = hessian[nz_mask]

            if mode == 'cpu':
                #centroids, labels = k_means_cpu(w.cpu().numpy(), n_cluster[0], init=centroids_init, max_iter=max_iter)
                centroids, labels = weighted_k_means_cpu(w.cpu().numpy(), importance.cpu().numpy(), hessian.cpu().numpy(), n_cluster[0], init=centroids_init, max_iter=max_iter, ha=ha, entropy_reg=entropy_reg, diameter_reg=diameter_reg, diameter_entropy_reg=diameter_entropy_reg)
            elif mode == 'gpu':
                centroids, labels = k_means_torch(w, n_cluster[0], init=centroids_init, max_iter=max_iter)
                #TODO gpu algorithm for weighted k means
            elif mode == 'compact':
                centroids, labels = k_means_torch_compact(w, n_cluster[0], init=centroids_init, use_gpu=True)
                #TODO compact algorithm for weighed k means

            if is_pruned:
                full_labels = labels.new(ori_shape).zero_() - 1  # use -1 for pruned elements
                full_labels[nz_mask] = labels
                labels = full_labels

            this_cl_list.append([centroids, labels])
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)
            
            #print("layer {} weight, L_2 loss {:.6f}".format(i,torch.mul(w_q - w, w_q - w).mean()))
            del layer.weight
            # layer.weight = nn.Parameter(torch.from_numpy(w_q).float().cuda())
            layer.weight = nn.Parameter(w_q.float())
            del w_q

        # quantize bias
        if hasattr(layer, 'bias') and quantize_bias:
            w = layer.bias.data
            importance = importances[i]
            hessian = hessians[i]

            if mode == 'cpu':
                #centroids, labels = k_means_cpu(w.cpu().numpy(), n_cluster[0], init=centroids_init, max_iter=max_iter)
                centroids, labels = weighted_k_means_cpu(w.cpu().numpy(), importance.cpu().numpy(), hessian.cpu().numpy(), n_cluster[1], init=centroids_init, max_iter=max_iter, ha=ha, entropy_reg=entropy_reg, diameter_reg=diameter_reg, diameter_entropy_reg=diameter_entropy_reg)
            elif mode == 'gpu':
                centroids, labels = k_means_torch(w, n_cluster[1], init=centroids_init, max_iter=max_iter)
                #TODO gpu algorithm for different weighted k means
            elif mode == 'compact':
                centroids, labels = k_means_torch_compact(w, n_cluster[1], init=centroids_init, use_gpu=True)
                #TODO compact algorithm for different weighted k means

            this_cl_list.append([centroids, labels])
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)

            
            #print("layer {} bias, L_2 loss {:.6f}".format(i,torch.mul(w_q - w, w_q - w).mean()))
            del layer.bias
            # layer.bias = nn.Parameter(torch.from_numpy(w_q).float().cuda())
            layer.bias = nn.Parameter(w_q.float())

        centroid_label_dict[i] = this_cl_list
        print("Finished quantize layer %d (%d params) with time %.4f"%(i, np.size(w.cpu().numpy()), time.time()-start_time))
    return centroid_label_dict

#Deprecated
'''
def variational_update(centroids_, labels_, importance_, w_, lamb=1e-7):
    centroids, labels, importance, w = centroids_.cpu().numpy(), labels_.cpu().numpy(), importance_.cpu().numpy(), w_.cpu().numpy()
    m, k = np.size(w), np.size(centroids)
    w_shape, i_shape = np.shape(w), np.shape(importance)
    assert np.array_equal(w_shape, i_shape)
    w_1d, i_1d, label_1d = w.flatten(), importance.flatten(), labels.flatten()
    if np.mean(i_1d) > 1e-20:
        i_1d = i_1d/np.mean(i_1d)
    else:
        return torch.from_numpy(labels).int().cuda()

    prob = np.bincount(np.concatenate((label_1d, np.arange(k))))
    theta = -np.log(prob*(1.0/(m+k)))
    theta_order = np.argsort(theta)
    valid_clusters = []
    for j in range(k):
        tmp = []
        for j_1 in theta_order:
            is_valid = True
            for j_2 in tmp:
                if (centroids[0][j_1]-centroids[0][j])*(centroids[0][j_2]-centroids[0][j]) > 0 and np.abs(centroids[0][j_2]-centroids[0][j]) < np.abs(centroids[0][j_1] - centroids[0][j]):
                    is_valid = False
                    break
            if is_valid:
                tmp.append(j_1)
            if j_1 == j:
                break
        valid_clusters.append(tmp)
    new_labels = label_1d

    for i in range(m):
        if label_1d[i] != theta_order[0]:
            valid_cluster = valid_clusters[label_1d[i]]
            label_1d[i] = valid_cluster[np.argmin([i_1d[i]*(w_1d[i]-centroids[0][j])**2 + lamb*theta[j] for j in valid_cluster])]

    new_labels = new_labels.reshape(w_shape)
    return torch.from_numpy(new_labels).int().cuda()
'''

def prune_weight_incremental(weight, mask, delta, qvalue_list):
    ori_shape = weight.shape
    n_weight = np.prod(ori_shape)
    n_quantized = np.sum(1 - mask)  # if 0, quantized
    assert n_quantized < n_weight * delta
    # sort by abs to find the weights to prune
    active_weight = np.multiply(np.abs(weight), mask)  # set the quantized weights to 0
    active_weight_flat = active_weight.reshape(-1)
    active_weight_flat.sort()
    split_ind = int(n_weight * delta) - n_quantized  # the number of weights to quantize this time
    split_val = active_weight_flat[-split_ind]
    new_mask = mask.copy()
    new_mask[active_weight < split_val] = 0

    incremental_mask = mask - new_mask  # the 1s are the weight pruned in this iter
    assert incremental_mask.min() == 0.
    assert incremental_mask.max() == 1.


## For adaptive quantization

def adaptive_quantization(model, val_loader, checkpoint, quantizable_ind, quantize_bit_choice):
    module_list = list(model.modules())

    model.load_state_dict(checkpoint)  # restore weight first
    org_acc = validate(val_loader, model)
    # 1. Calculate t_i
    # calculate the mean value of adversarial noise for the dataset, notice that we do not add Softmax to network

    mean_adv_noise_meter = AverageMeter()
    with torch.no_grad():
        for input_w, target in val_loader:
            input_var = torch.autograd.Variable(input_w).cuda()
            output = model(input_var)
            top2, inds = torch.topk(output, 2, dim=1)
            mean_adv_noise = torch.mean((top2[:, 0] - top2[:, 1]) ** 2 / 2.)
            mean_adv_noise_meter.update(mean_adv_noise.cpu().data[0], output.size(0))
    mean_adv_noise_dset = mean_adv_noise_meter.avg
    print('Mean adversarial noise for the dataset is: {:.4f}'.format(mean_adv_noise_dset))

    d_acc = 10.  # choose 10% for delta_acc. Does not matter
    t_i_list = []
    for ind in quantizable_ind:
        layer = module_list[ind]
        assert hasattr(layer, 'weight')
        r_W_i_pi = torch.rand(layer.weight.data.size()).cuda() - 0.5  # re-normalize to [-0.5, 0.5]
        k = k_min = 1e-5
        k_max = 1e1
        # get initial acc'
        model.load_state_dict(checkpoint)  # restore weight first
        # assert validate(val_loader, model) == org_acc  # removed to accelerate
        layer.weight.data += k * r_W_i_pi
        new_acc = validate(val_loader, model)
        while not np.abs(org_acc - new_acc - d_acc) < 0.1:
            if org_acc - new_acc < d_acc:
                k_min = k
            else:
                k_max = k
            k = np.sqrt(k_min * k_max).item()
            model.load_state_dict(checkpoint)  # restore weight first
            layer.weight.data += k * r_W_i_pi
            new_acc = validate(val_loader, model)
            print('Layer {} current acc degradation: {:.3f}'.format(ind, org_acc - new_acc))
        mean_r_z_i = AverageMeter()
        with torch.no_grad():
            for i, (input_d, target) in enumerate(val_loader):
                input_var = torch.autograd.Variable(input_d).cuda()
                # compute output
                model.load_state_dict(checkpoint)  # restore weight first
                output1 = model(input_var)
                layer.weight.data += k * r_W_i_pi
                output2 = model(input_var)
                norm_r = torch.norm(output1 - output2, p=2, dim=1) ** 2
                mean_r_z_i.update(torch.mean(norm_r).cpu().data[0], output1.size(0))
        t_i = mean_r_z_i.avg / mean_adv_noise_dset
        print('==> t_i for layer {}: {}'.format(ind, t_i))
        t_i_list.append(t_i)
    print('t_i_list: ')
    print(t_i_list)
    # t_i_list = [113.314645623, 108.02437323, 111.228006385, 109.585273767, 115.362011096, 111.136186881, 114.150737539,
    #             106.789374298, 135.436417323, 118.175965146, 136.776404035, 162.089406771, 224.905060191, 340.589419784,
    #             904.878690392, 256.250864841]

    # 2. Calculate p_i
    fix_b_i = 6
    p_i_list = []
    for i, ind in enumerate(quantizable_ind):
        model.load_state_dict(checkpoint)  # restore weight first
        centroids, labels = k_means_torch_compact(module_list[ind].weight.data, 2 ** fix_b_i, init='linear', use_gpu=False)
        w_q = reconstruct_weight_from_k_means_result(centroids, labels)
        del centroids, labels
        mean_r_z_i = AverageMeter()
        with torch.no_grad():
            for input_d, target in val_loader:
                input_var = torch.autograd.Variable(input_d).cuda()
                # compute output
                model.load_state_dict(checkpoint)  # restore weight first
                output1 = model(input_var)
                del module_list[ind].weight
                module_list[ind].weight = nn.Parameter(w_q.float())
                output2 = model(input_var)
                norm_r = torch.norm(output1 - output2, p=2, dim=1) ** 2
                mean_r_z_i.update(torch.mean(norm_r).cpu().data[0], output1.size(0))
        del w_q
        p_i = mean_r_z_i.avg  # / np.exp(-np.log(4) * fix_b_i)
        print('==> p_i for layer {}: {}'.format(ind, p_i))
        p_i_list.append(p_i)

    # 3. Calculate b_i
    b1 = 8
    assert len(p_i_list) == len(t_i_list) == len(quantizable_ind)
    layer1_size = np.prod(module_list[quantizable_ind[0]].weight.data.size())
    bits_list = [b1]
    print('==> Layer1 size: {}, bits: {}'.format(layer1_size, b1))
    for i, ind in enumerate(quantizable_ind):
        if i == 0:
            continue
        this_size = np.prod(module_list[ind].weight.data.size())
        b_i = b1 + (1 / np.log(4)) * np.log(p_i_list[i] * t_i_list[0] * layer1_size / (p_i_list[0] * t_i_list[i] * this_size))
        print('Optimal bits for layer {}: {}'.format(ind, b_i))
        bits_list.append(b_i)

    print('Final result: {}'.format(bits_list))


def validate(val_loader, model):
    from utils import accuracy
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target).cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

    return top1.avg


def get_quantized_weight_size(quantizable_layer_size, quantize_clusters, float_bit=32):
    total_bit = 0
    assert len(quantize_clusters) == len(quantizable_layer_size)
    for n_clusters, l_size in zip(quantize_clusters, quantizable_layer_size):
        if type(n_clusters) == list:  # if provided bit setting for each sub-layer weight
            assert type(l_size) == list
            assert len(n_clusters) == len(l_size)
            for c, s in zip(n_clusters, l_size):
                total_bit += s * np.log2(c) + c * float_bit  # also count size of the centroids
        else:
            total_size = np.sum(l_size)
            total_bit += total_size * np.ceil(np.log2(n_clusters)) + n_clusters * float_bit  # also count size of the centroids
    return total_bit

def get_huffmaned_weight_size(centroid_label_dict, quantizable_layer_size, quantize_clusters, float_bit=32):
    total_bit = 0
    assert len(centroid_label_dict) == len(quantizable_layer_size)
    assert len(quantize_clusters) == len(quantizable_layer_size)
    for cl_dict, n_clusters, l_size in zip(list(centroid_label_dict.values()), quantize_clusters, quantizable_layer_size):
        if type(n_clusters) == list:  # if provided bit setting for each sub-layer weight
            assert type(l_size) == list
            assert len(n_clusters) == len(l_size)
            assert type(cl_dict) == list
            assert len(cl_dict) == len(l_size)
            for d, c, s in zip(cl_dict, n_clusters, l_size):
                ent = get_entropy(d[0][1].cpu().numpy().flatten())
                total_bit += s * ent + c * float_bit  # also count size of the centroids
        else:
            total_size = np.sum(l_size)
            ent = get_entropy(cl_dict[0][1].cpu().numpy().flatten())
            total_bit += total_size * ent + n_clusters * float_bit  # also count size of the centroids
    return total_bit

def get_entropy(labels):
    positive_labels = labels[labels > -0.5]
    rate = len(positive_labels) * 1.0 / len(labels)
    counts = np.bincount(positive_labels)
    freq = counts*1.0/np.sum(counts) 
    return sp.stats.entropy(freq, base=2)

def get_original_weight_size(quantizable_layer_size, float_bit=32):
    # assume that the original float is 32bit
    total_bit = 0
    for s in quantizable_layer_size:
        if type(s) == list:
            s = sum(s)
        total_bit += s * float_bit
    return total_bit


def seperate_choice(current_policy, quantizable_ind, activation_layer_ind):
    w_choice = []
    a_choice = []
    all_ind = quantizable_ind + activation_layer_ind
    all_ind.sort()
    for i, ind in enumerate(all_ind):
        if ind in quantizable_ind:
            w_choice.append(current_policy[i])
        else:
            a_choice.append(current_policy[i])
    return w_choice, a_choice


# inject activation quantization op
def inject_act_quantize_op(net, n_bits, quantizable_type=(nn.Conv2d, nn.Linear), idx_sampler=None):
    '''
    Inject quantization op to activation
    :param net: model to inject
    :param n_bits: number of bits to quantize
    :param quantizable_type: 
    :param idx_sampler: sample the quantizable idx
    :return: 
    '''
    m_list = list(net.modules())
    quantizable_idx = []
    for i_m, m in enumerate(m_list):
        if type(m) in quantizable_type:
            quantizable_idx.append(i_m)
    if idx_sampler:
        quantizable_idx = idx_sampler(quantizable_idx)
    for i_m in quantizable_idx:
        if i_m == quantizable_idx[0]:  # do not prune the raw input
            continue
        print('Injecting layer {}'.format(i_m))
        m = m_list[i_m]

        def new_forward(m):
            def lambda_forward(x):
                import torch.nn.functional as F
                ori_x = x
                x = F.relu6(x)
                n_centroid = 2 ** n_bits
                x = (x - 0) / (6 - 0)  # normalize to 0 - 1
                x = torch.round(x * n_centroid * 1.)
                x = x * 1. / n_centroid  # re-normalize to 0-1
                x = x * (6 - 0) + 0
                # STE
                x = ori_x + x.detach() - ori_x.detach()
                return m.old_forward(x)

            return lambda_forward

        m.old_forward = m.forward
        m.forward = new_forward(m)
