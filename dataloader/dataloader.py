from __future__ import print_function
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
from PIL import Image
import random


def get_transform(img_size=32):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    return transform_train, transform_test


def get_transform_svhn(img_size=32):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform_train, transform_test

def data_loader(dataset='cifar10', root=None, train_batch=128, test_batch=100, transform=None):
    if dataset == 'cifar10':
        if root is None:
            root = './../../datasets/cifar10'
        if transform is None:
            transform = get_transform(img_size=32)

        # trainset = torchvision.datasets.CIFAR10(root=root,
        trainset = CIFAR10(root=root,
                            train=True,
                            download=True,
                            transform=transform[0])
        testset = torchvision.datasets.CIFAR10(root=root,
                            train=False,
                            download=True,
                            transform=transform[1])
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif dataset == 'cifar100':
        if root is None:
            root = './../../datasets/cifar100'
        if transform is None:
            transform = get_transform(img_size=32)

        # trainset = torchvision.datasets.CIFAR100(root=root,
        trainset = CIFAR100(root=root,
                            train=True,
                            download=True,
                            transform=transform[0])
        testset = torchvision.datasets.CIFAR100(root=root,
                                            train=False,
                                            download=True,
                                            transform=transform[1])

    elif dataset == 'svhn':
        if root is None:
            root = './../../datasets/svhn'
        if transform is None:
            transform = get_transform_svhn(img_size=32)
        trainset = SVHN(root=root,
                                                split='train',
                                                download=True,
                                                transform=transform[0])
        testset = torchvision.datasets.SVHN(root=root,
                                                split='test',
                                                download=True,
                                                transform=transform[1])

    elif dataset == 'mnist':
        if root is None:
            root = './../../datasets/mnist'
        if transform is None:
            transform = get_transform(img_size=28)

        trainset = torchvision.datasets.MNIST(root=root,
                                                train=True,
                                                download=True,
                                                transform=transform[0])
        testset = torchvision.datasets.MNIST(root=root,
                                                train=False,
                                                download=True,
                                                transform=transform[1])

    elif dataset == 'tiny':
        from .crd_tinyImage import load_TinyImageNet
        trainset, testset = load_TinyImageNet(batch_size=train_batch,
                                                test_size=test_batch,
                                                size=64, resize=64)

    else:
        pass

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=train_batch,
                                            shuffle=True,
                                            num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=test_batch,
                                            shuffle=False,
                                            num_workers=2)
    return trainloader, testloader


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # selset_label = [0,1,8,9]
        # while target in selset_label:
        #     # ????????????????????????
        #     p = random.random()
        #     if p > 0.1:
        #         if index < 49999:
        #             index += 1
        #             img, target = self.data[index], self.targets[index]
        #         else:
        #             index -= 1000
        #             img, target = self.data[index], self.targets[index]
        #     else:
        #         break
 
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, index
        else:
            return img, target
    
    @property
    def num_classes(self):
        return 10


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 100


class SVHN(datasets.SVHN):
    def __init__(self, root, split='train',
                 transform=None, target_transform=None,download=False):
        super().__init__(root=root, split=split, download=download,
                         transform=transform, target_transform=target_transform)

        num_classes = 10
        if self.split == 'train':
            num_samples = len(self.data)
            label = self.labels
        else:
            num_samples = len(self.data)
            label = self.labels

    def __getitem__(self, index):
        if self.split == 'train':
            img, target = self.data[index], self.labels[index]
        else:
            img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.transpose(1,2,0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
      