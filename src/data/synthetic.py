from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from ..core.registry import register
import random


@dataclass
class SyntheticCoherenceData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "train_x": self.train_x,
            "train_y": self.train_y,
            "test_x": self.test_x,
            "test_y": self.test_y,
        }


def synthetic_coherence_ambient_dim(copies: int) -> int:
    return 2 * copies


def synthetic_coherence_teacher_output(latent: torch.Tensor) -> torch.Tensor:
    return torch.relu(latent[:, 0] + latent[:, 1]).unsqueeze(1)


def make_exact_copy_embedding(copies: int, active_copies: int) -> torch.Tensor:
    basis = torch.zeros(synthetic_coherence_ambient_dim(copies), 2)
    scale = 1.0 / np.sqrt(active_copies)
    for idx in range(active_copies):
        basis[2 * idx + 0, 0] = scale
        basis[2 * idx + 1, 1] = scale
    return basis


def make_random_tight_embedding(
    copies: int,
    active_copies: int,
    perm: torch.Tensor,
    angles: torch.Tensor,
    anchor_aligned_endpoint: bool,
) -> torch.Tensor:
    basis = torch.zeros(synthetic_coherence_ambient_dim(copies), 2)
    if active_copies == 1 and anchor_aligned_endpoint:
        basis[int(perm[0].item()), 0] = 1.0
        basis[int(perm[1].item()), 1] = 1.0
        return basis

    scale = 1.0 / np.sqrt(active_copies)
    for idx in range(active_copies):
        theta = float(angles[idx].item())
        row_a = scale * torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32)
        row_b = scale * torch.tensor([-np.sin(theta), np.cos(theta)], dtype=torch.float32)
        basis[int(perm[2 * idx].item())] = row_a
        basis[int(perm[2 * idx + 1].item())] = row_b
    return basis


def make_synthetic_coherence_data(
    train_latent: torch.Tensor,
    test_latent: torch.Tensor,
    basis: torch.Tensor,
) -> SyntheticCoherenceData:
    return SyntheticCoherenceData(
        train_x=train_latent @ basis.T,
        train_y=synthetic_coherence_teacher_output(train_latent),
        test_x=test_latent @ basis.T,
        test_y=synthetic_coherence_teacher_output(test_latent),
    )

class GaussianSubspaceDataset(torch.utils.data.Dataset):
    """
    A synthetic dataset that generates Gaussian clusters for a variable number of classes 
    in a strictly low-dimensional intrinsic space, and projects them into a 
    high-dimensional ambient space using a random orthogonal basis.
    """
    def __init__(self, structure_seed: int, sampling_seed=None, num_samples=10000, num_classes=5, 
                 d_intrinsic=10, d_ambient=1000, class_separation=3.0):
        assert d_ambient >= d_intrinsic, "Ambient dimension d must be >= intrinsic dimension k."
        
        structure_generator = torch.Generator()
        structure_generator.manual_seed(structure_seed)
        sampling_generator = torch.Generator()
        sampling_generator.manual_seed(sampling_seed if sampling_seed is not None else random.randint(0, 2**32 - 1))
            
        # 1. Define cluster centers (means) in the low-dimensional space
        # We multiply by class_sep to push the Gaussian centers apart. 
        # Lower class_sep = harder classification task.
        means = torch.randn(num_classes, d_intrinsic, generator=structure_generator) * class_separation
        
        # 2. Generate random labels evenly distributed across num_classes
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=sampling_generator)
        
        # 3. Generate the latent low-dimensional data (Z)
        # Fetch the exact mean for each sample's label, then add standard Gaussian noise
        Z_means = means[self.labels]
        random_matrix_noise = torch.randn(num_classes, d_intrinsic, d_intrinsic, generator=structure_generator)
        P_noise, _ = torch.linalg.qr(random_matrix_noise)
        Z_noise = (
            (torch.randn(num_samples, d_intrinsic, generator=sampling_generator) * torch.randn(1, d_intrinsic, generator=structure_generator)))
        Z_noise = torch.einsum("nd,cDd->cnD", Z_noise, P_noise)
        Z = Z_means + Z_noise[self.labels, torch.arange(num_samples)]  # Shape: (num_samples, d_intrinsic)

        # 4. Create the orthogonal projection matrix (P)
        # Generate a random D x d matrix and use QR to get strictly orthogonal columns
        random_matrix = torch.randn(d_ambient, d_intrinsic, generator=structure_generator)
        P, _ = torch.linalg.qr(random_matrix)  # P shape: (D_ambient, d_intrinsic)
        
        # 5. Project the data into the massive ambient space
        # Z is (N, d) multiplied by P.T (d, D) -> resulting in (N, D)
        self.data = Z @ P.T
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@register('dataset', 'gaussian-subspace-dataset')
def create_gaussian_subspace_dataset(cfg_dataset, train=True):
    config = cfg_dataset.gaussian_subspace_dataset
    return GaussianSubspaceDataset(
        structure_seed=config.structure_seed,
        sampling_seed=(config.sampling_seed_train if train else config.sampling_seed_test),
        num_samples=config.num_samples_train if train else config.num_samples_test,
        num_classes=config.num_classes,
        d_intrinsic=config.d_intrinsic,
        d_ambient=config.d_ambient,
        class_separation=config.class_separation,
    )

class MultiLabelSubspaceDataset(torch.utils.data.Dataset):
    """
    Generates K independent binary labels embedded in a D-dimensional space.
    Instantiate this ONCE, then split it later.
    """
    def __init__(self, num_samples=10000, num_binary_labels=50, d_ambient=1000, data_seed=None, projection_seed=None):
        assert d_ambient >= num_binary_labels, "Ambient dimension must be >= K."

        data_generator = torch.Generator()
        data_generator.manual_seed(data_seed if data_seed is not None else random.randint(0, 2**32 - 1))

        # 1. Generate latent data Z and labels
        Z = torch.randn(num_samples, num_binary_labels, generator=data_generator)
        self.labels = (Z > 0).float()
        

        projection_generator = torch.Generator()
        projection_generator.manual_seed(projection_seed if projection_seed is not None else random.randint(0, 2**32 - 1))
        # 2. Generate Projection Matrix P
        random_matrix = torch.randn(d_ambient, num_binary_labels, generator=projection_generator)
        self.P, _ = torch.linalg.qr(random_matrix)
            
        # 3. Embed into ambient space
        self.data = Z @ self.P.T
        
    def __len__(self): 
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@register('dataset', 'multilabel-subspace-dataset')
def create_multilabel_subspace_dataset(cfg_dataset, train=True):
    config = cfg_dataset.multilabel_subspace_dataset
    return MultiLabelSubspaceDataset(
        config.num_samples_train if train else config.num_samples_test,
        config.num_binary_labels,
        config.d_ambient,
        (config.data_seed_train if train else config.data_seed_test),
        config.projection_seed
    )


class ParitySubspaceDataset(torch.utils.data.Dataset):
    """
    Generates a continuous parity (XOR) problem embedded in a D-dimensional space.
    Instantiate this ONCE, then split it later.
    """
    def __init__(self, num_samples=10000, d_intrinsic=4, d_ambient=200, modulo_base=2, data_seed=None, projection_seed=None):
        assert d_ambient >= d_intrinsic, "Ambient dimension must be >= intrinsic."

        data_generator = torch.Generator()
        data_generator.manual_seed(data_seed if data_seed is not None else random.randint(0, 2**32 - 1))
        
        # 1. Generate latent data Z (Uniform [-1, 1])
        Z = torch.empty(num_samples, d_intrinsic).uniform_(-1.0, 1.0, generator=data_generator)
        
        # 2. The Parity Label
        neg_count = (Z < 0).sum(dim=1)
        self.labels = (neg_count % modulo_base).long()
        
        # 3. Generate Projection Matrix P
        projection_generator = torch.Generator()
        projection_generator.manual_seed(projection_seed if projection_seed is not None else random.randint(0, 2**32 - 1))
        random_matrix: torch.Tensor = torch.randn(d_ambient, d_intrinsic, generator=projection_generator)
        self.P, _ = torch.linalg.qr(random_matrix)
            
        # 4. Embed into ambient space
        self.data = Z @ self.P.T
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):# -> tuple[Any, Tensor]:
        return self.data[idx], self.labels[idx]

@register('dataset', 'parity-subspace-dataset')
def create_parity_subspace_dataset(cfg_dataset, train=True):
    config = cfg_dataset.parity_subspace_dataset
    return ParitySubspaceDataset(
        config.num_samples_train if train else config.num_samples_test,
        config.d_intrinsic,
        config.d_ambient,
        config.modulo_base,
        (config.data_seed_train if train else config.data_seed_test),
        config.projection_seed
    )
