"""
Data transformation utilities.
"""

import torch
from torchvision import transforms as T
from typing import Tuple, Optional


# Standard normalization statistics
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CIFAR_STATS = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
MNIST_STATS = ([0.1307], [0.3081])

# Dataset groups
MNIST_DATASETS = ['MNIST', 'KMNIST', 'FashionMNIST', 'EMNIST']


def get_transform(pad, crop, stats, flip):
    """Get data transforms for image datasets.
    
    Args:
        pad: Padding size
        crop: Crop size
        stats: Normalization statistics (mean, std)
        flip: Whether to use random horizontal flip
        
    Returns:
        Tuple of (train_transform, valid_transform, multiplier)
    """
    tfm = [
        T.Pad(pad, padding_mode="reflect"),
        T.RandomCrop(crop),
    ]

    multiplier = 4

    if flip:
        tfm += [T.RandomHorizontalFlip(0.5)]
        multiplier *= 2

    base = [T.ToTensor(), T.Normalize(*stats)]

    return (
        T.Compose(base + tfm),
        T.Compose(base),
        multiplier
    )


def get_stats(dataset_name):
    """Get normalization statistics for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (mean, std)
    """
    if dataset_name in MNIST_DATASETS:
        return MNIST_STATS
    elif dataset_name in ['CIFAR10', 'CIFAR100']:
        return CIFAR_STATS
    else:
        return IMAGENET_STATS


def get_transforms(dataset_name, train=True):
    """Get transforms for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        train: Whether to get training transforms
        
    Returns:
        Transform composition
    """
    if dataset_name == 'MNIST':
        if train:
            return T.Compose([
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))
            ])
    
    elif dataset_name in ['CIFAR10', 'CIFAR100']:
        if train:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, 4),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    else:
        # Default ImageNet-style transforms
        if train:
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class DataAugmentation:
    """Data augmentation utilities."""
    
    @staticmethod
    def cifar_augmentation():
        """CIFAR-10/100 augmentation."""
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, 4),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def mnist_augmentation():
        """MNIST augmentation."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
    
    @staticmethod
    def imagenet_augmentation():
        """ImageNet-style augmentation."""
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
