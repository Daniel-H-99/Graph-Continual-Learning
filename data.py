import os
import copy
import numpy as np
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset


DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs):
    return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn


def class_split(dataset, num_tasks, num_samples_per_task=-1, multihead=False, **kwargs):
    task_loaders = []
    if not hasattr(dataset, 'classes'):
        dataset.classes = list(range(10))
    task_splits = np.array_split(range(len(dataset.classes)), num_tasks)
    if not hasattr(dataset, 'targets'):
        dataset.targets = dataset.labels
    targets = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else dataset.targets

    for task_id, class_split in enumerate(task_splits):
        indices = [idx for idx, target in enumerate(targets) if target in class_split]
        if num_samples_per_task > 0:
            indices = random.sample(indices, num_samples_per_task)
        if kwargs["batch_size"] < 0:
            kwargs["batch_size"] = len(indices)
        task_dataset = copy.deepcopy(dataset)
        if multihead:
            task_dataset.targets = np.array(task_dataset.targets) % (len(dataset.classes) // num_tasks)
        task_loader = DataLoader(Subset(task_dataset, indices), **kwargs)
        task_loaders.append(task_loader)
    return task_loaders


@register_dataset("rotated_mnist")
def load_rotated_mnist(data, num_samples_per_task=1000, batch_size=10, image_size=28, num_workers=0):
    train_datasets, valid_datasets = torch.load(os.path.join(data, "mnist", "rotated", "mnist_rotations.pt"))
    train_loaders, valid_loaders = [], []
    for task_id, train_data, train_target in train_datasets:
        if num_samples_per_task > 0:
            indices = torch.randperm(len(train_data))[0:num_samples_per_task]
            train_data, train_target = train_data[indices], train_target[indices]
        train_dataset = TensorDataset(train_data, train_target)
        train_loaders.append(DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers))

    for task_id, valid_data, valid_target in valid_datasets:
        valid_dataset = TensorDataset(valid_data, valid_target)
        valid_loaders.append(DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_data), num_workers=num_workers))
    return train_loaders, valid_loaders


@register_dataset("permuted_mnist")
def load_permuted_mnist(data, num_samples_per_task=1000, batch_size=10, image_size=28, num_workers=0):
    train_datasets, valid_datasets = torch.load(os.path.join(data, "mnist", "permuted", "mnist_permutations.pt"))
    train_loaders, valid_loaders = [], []
    for task_id, train_data, train_target in train_datasets:
        if num_samples_per_task > 0:
            indices = torch.randperm(len(train_data))[0:num_samples_per_task]
            train_data, train_target = train_data[indices], train_target[indices]
        train_dataset = TensorDataset(train_data, train_target)
        train_loaders.append(DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers))

    for task_id, valid_data, valid_target in valid_datasets:
        valid_dataset = TensorDataset(valid_data, valid_target)
        valid_loaders.append(DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_data), num_workers=num_workers))
    return train_loaders, valid_loaders


@register_dataset("split_cifar10")
def load_split_cifar10(data, num_samples_per_task=-1, num_tasks=5, batch_size=10, num_workers=2):
    def get_transform(training=True):
        transform = []
        transform.append(transforms.RandomHorizontalFlip()) if training else None
        transform.append(transforms.ToTensor())
        return transforms.Compose(transform)

    data = os.path.join(data, "cifar10")
    train_dataset = datasets.CIFAR10(data, train=True, download=not os.path.exists(data), transform=get_transform(training=True))
    train_loaders = class_split(train_dataset, num_tasks, num_samples_per_task, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    valid_dataset = datasets.CIFAR10(data, train=False, download=not os.path.exists(data), transform=get_transform(training=False))
    valid_loaders = class_split(valid_dataset, num_tasks, num_samples_per_task=-1, shuffle=False, batch_size=-1, num_workers=num_workers)
    return train_loaders, valid_loaders


@register_dataset("split_svhn")
def load_split_svhn(data, num_samples_per_task=-1, num_tasks=5, batch_size=10, num_workers=2):
    def get_transform(training=True):
        transform = []
        transform.append(transforms.RandomHorizontalFlip()) if training else None
        transform.append(transforms.ToTensor())
        return transforms.Compose(transform)

    data = os.path.join(data, "svhn")
    train_dataset = datasets.SVHN(data, split='train', download=not os.path.exists(data), transform=get_transform(training=True))
    train_loaders = class_split(train_dataset, num_tasks, num_samples_per_task, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    valid_dataset = datasets.SVHN(data, split='test', download=not os.path.exists(data), transform=get_transform(training=False))
    valid_loaders = class_split(valid_dataset, num_tasks, num_samples_per_task=-1, shuffle=False, batch_size=-1, num_workers=num_workers)
    return train_loaders, valid_loaders
