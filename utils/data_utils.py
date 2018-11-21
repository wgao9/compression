import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import os


def get_dataset(dset_name, batch_size, n_worker, cifar_root='../../data', for_inception=False, normalize=True):
    cifar_tran_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    cifar_tran_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]


    print('==> Preparing data..')
    if dset_name == 'cifar10':
        if not normalize:
            cifar_tran_train = cifar_tran_train[:-1]
            cifar_tran_test = cifar_tran_test[:-1]
        transform_train = transforms.Compose(cifar_tran_train)
        transform_test = transforms.Compose(cifar_tran_test)
        trainset = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=n_worker, pin_memory=True)
        testset = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
    elif dset_name == 'cifar100':
        if not normalize:
            cifar_tran_train = cifar_tran_train[:-1]
            cifar_tran_test = cifar_tran_test[:-1]
        transform_train = transforms.Compose(cifar_tran_train)
        transform_test = transforms.Compose(cifar_tran_test)
        trainset = torchvision.datasets.CIFAR100(root=cifar_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=n_worker, pin_memory=True)
        testset = torchvision.datasets.CIFAR100(root=cifar_root, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 100
    elif dset_name == 'imagenet':
        possible_imgnet_paths = ['/data/dataset/imagenet', '/data1/ilsvrc/', '/ssd/dataset/imagenet/']
        for p in possible_imgnet_paths:
            if os.path.exists(p):
                imagenet_base = p
                break
        traindir = os.path.join(imagenet_base, 'train')
        valdir = os.path.join(imagenet_base, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),  # TODO: find a good scale ratio
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if False:  # args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=n_worker, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)
        n_class = 1000

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class


def get_split_train_dataset(dset_name, batch_size, n_worker, val_size, train_size=None, random_seed=1,
                            cifar_root='../../data', use_real_val=False, for_inception=False, shuffle=True):
    '''
    split the train set into train / val for rl search
    '''
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:  # every time we use the same order for
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('==> Preparing data..')
    if dset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # the 2 sets are actually exactly the same thing here
        trainset = torchvision.datasets.CIFAR100(root=cifar_root, train=True, download=True, transform=transform_train)
        if use_real_val:
            valset = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_test)
            n_val = len(valset)
            indices = list(range(n_val))
            import numpy as np
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            assert val_size < n_val
            _, val_idx = indices[val_size:], indices[:val_size]

            train_idx = list(range(len(trainset)))  # all trainset
        else:
            valset = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_test)
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            import numpy as np
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        if train_size:
            train_idx = train_idx[:train_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
    elif dset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # the 2 sets are actually exactly the same thing here, except for the transform
        trainset = torchvision.datasets.CIFAR100(root=cifar_root, train=True, download=True, transform=transform_train)
        if use_real_val:
            valset = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_test)
            n_val = len(valset)
            indices = list(range(n_val))
            import numpy as np
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            assert val_size < n_val
            _, val_idx = indices[val_size:], indices[:val_size]

            train_idx = list(range(len(trainset)))  # all trainset
        else:
            valset = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform_test)
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            import numpy as np
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        if train_size:
            train_idx = train_idx[:train_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 100
    elif dset_name == 'imagenet':
        imagenet_base = None
        possible_imgnet_paths = ['/data/dataset/imagenet', '/data1/ilsvrc/', '/ssd/dataset/imagenet/']
        for p in possible_imgnet_paths:
            if os.path.exists(p):
                imagenet_base = p
                break
        if imagenet_base is None:
            raise FileNotFoundError
        traindir = os.path.join(imagenet_base, 'train')
        valdir = os.path.join(imagenet_base, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        if use_real_val:
            valset = datasets.ImageFolder(valdir, test_transform)
            n_val = len(valset)
            indices = list(range(n_val))
            import numpy as np
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            assert val_size < n_val
            _, val_idx = indices[val_size:], indices[:val_size]

            train_idx = list(range(len(trainset)))  # all trainset

        else:
            valset = datasets.ImageFolder(traindir, test_transform)
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            import numpy as np
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        if train_size:
            train_idx = train_idx[:train_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)

        n_class = 1000
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
