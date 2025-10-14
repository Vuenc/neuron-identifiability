from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from ..core.registry import register

# Optional imports for GNN functionality
try:
    import torch_geometric.transforms as T
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    print("Warning: torch_geometric or ogb not available. GNN functionality will be limited.")
    T = None
    PygNodePropPredDataset = None


# Optional imports for FFCV functionality
try:
    import ffcv
    import ffcv.fields.decoders
    import ffcv.pipeline.operation
    import ffcv.transforms
except ImportError:
    print("Warning: FFCV not installed, FFCV dataloaders will not be available.")
    ffcv = None


# Mean values for the datasets are computed on the training set (train_dataset, _, _ = create_train_val_test_split(train_val_dataset, val_split=0.1, test_split=0.0, seed=42))
CIFAR10_MEAN = (0.4917, 0.4823, 0.4467)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5068, 0.4863, 0.4408)
CIFAR100_STD = (0.2672, 0.2564, 0.2760)
MNIST_MEAN = (0.1307)
MNIST_STD = (0.3081)


@register('dataset', 'mnist')
def create_mnist_dataset(data_dir='./data', train=True, transform=None):
    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
    
    return torchvision.datasets.MNIST(root=data_dir, train=train, transform=transform, download=True)


@register('dataset', 'cifar10')
def create_cifar10_dataset(data_dir='./data', train=True, transform=None):
    if transform is None:
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
            ])
    
    return torchvision.datasets.CIFAR10(root=data_dir, train=train, transform=transform, download=True)


def create_ffcv_dataloader_cifar(data_path, mean: Tuple[float, float, float], std: Tuple[float, float, float], train=True, batch_size=32, device="cuda:0", num_workers=-1):
    assert ffcv # type: ignore
    mean_u8: Tuple[int, int, int] = tuple([x * 255 for x in mean]) # type: ignore
    std_u8: Tuple[int, int, int] = tuple(([x * 255 for x in std])) # type: ignore
    label_pipeline: List[ffcv.pipeline.operation.Operation] = [ffcv.fields.decoders.IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.ToDevice(torch.device(device)), ffcv.transforms.Squeeze()]
    image_pipeline: List[ffcv.pipeline.operation.Operation] = [ffcv.fields.decoders.SimpleRGBImageDecoder()]
    if train:
        image_pipeline.extend([
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.RandomTranslate(padding=4), # this is equivalent to RandomCrop(32, 4) since the input images are 32x32
        ])
    image_pipeline.extend([
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(torch.device(device), non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.Convert(torch.float32),
        torchvision.transforms.Normalize(mean=mean_u8, std=std_u8) # type: ignore
    ])

    ordering = ffcv.loader.OrderOption.RANDOM if train else ffcv.loader.OrderOption.SEQUENTIAL

    return ffcv.loader.Loader(data_path, batch_size=batch_size, num_workers=num_workers,
                            order=ordering, drop_last=train, os_cache=True,
                            pipelines={'image': image_pipeline, 'label': label_pipeline})


def create_ffcv_dataloader_cifar10(data_path="ffcv/cifar10_train.beton", train=True, batch_size=128, device="cuda:0", num_workers=-1):
    return create_ffcv_dataloader_cifar(data_path, mean=CIFAR10_MEAN, std=CIFAR10_STD, train=train, batch_size=batch_size, device=device, num_workers=num_workers)


def create_ffcv_dataloader_cifar100(data_path="ffcv/cifar10_train.beton", train=True, batch_size=128, device="cuda:0", num_workers=-1):
    return create_ffcv_dataloader_cifar(data_path, mean=CIFAR100_MEAN, std=CIFAR100_STD, train=train, batch_size=batch_size, device=device, num_workers=num_workers)


@register('dataset', 'cifar100')
def create_cifar100_dataset(data_dir='./data', train=True, transform=None):
    if transform is None:
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
            ])
    
    return torchvision.datasets.CIFAR100(root=data_dir, train=train, transform=transform, download=True)


@register('dataset', 'arxiv')
def create_arxiv_dataset(data_dir='./data'):
    if PygNodePropPredDataset is None or T is None:
        raise ImportError("torch_geometric and ogb are required for ArXiv dataset")
    
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', 
        root=data_dir,
        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    )
    return dataset


def create_dataset(dataset_name, data_dir='./data', train=True, transform=None, **kwargs):
    from ..core.registry import build_component
    
    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        return build_component('dataset', dataset_name, data_dir=data_dir, train=train, transform=transform)
    elif dataset_name == 'arxiv':
        return build_component('dataset', dataset_name, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, 
                     pin_memory=True, drop_last=False, **kwargs):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )

def create_train_val_test_split(dataset, val_split=0.1, test_split=0.1, seed=42):

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset
