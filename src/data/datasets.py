import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from ..core.registry import register
from .transforms import get_transform

# Optional imports for GNN functionality
try:
    import torch_geometric.transforms as T
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    print("Warning: torch_geometric or ogb not available. GNN functionality will be limited.")
    T = None
    PygNodePropPredDataset = None


@register('dataset', 'mnist')
def create_mnist_dataset(data_dir='./data', train=True, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    return datasets.MNIST(root=data_dir, train=train, transform=transform, download=True)


@register('dataset', 'cifar10')
def create_cifar10_dataset(data_dir='./data', train=True, transform=None):
    if transform is None:
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    return datasets.CIFAR10(root=data_dir, train=train, transform=transform, download=True)


@register('dataset', 'cifar100')
def create_cifar100_dataset(data_dir='./data', train=True, transform=None):
    if transform is None:
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    return datasets.CIFAR100(root=data_dir, train=train, transform=transform, download=True)


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


class LMCDataLoader:
    
    def __init__(self, dataset_name, data_dir='./data', batch_size=32, val_split=0.1, seed=42):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        
        self.train_dataset = create_dataset(dataset_name, data_dir, train=True)
        self.test_dataset = create_dataset(dataset_name, data_dir, train=False)
        
        self.train_dataset, self.val_dataset = create_train_val_test_split(
            self.train_dataset, val_split=val_split, test_split=0.0, seed=seed
        )[0:2]
        
        self.train_loader = create_dataloader(self.train_dataset, batch_size, shuffle=True)
        self.val_loader = create_dataloader(self.val_dataset, batch_size, shuffle=False)
        self.test_loader = create_dataloader(self.test_dataset, batch_size, shuffle=False)
    
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


class GNNDataLoader:
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        
        self.dataset = create_dataset('arxiv', data_dir)
        self.data = self.dataset[0]
        self.split_idx = self.dataset.get_idx_split()
    
    def get_data(self):
        return self.data, self.split_idx
