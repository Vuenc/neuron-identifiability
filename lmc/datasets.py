import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence
import torch
import torchvision

class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, device):
        """
        Args:
            image_path (str): Path to .npy file containing image data (N, C, H, W).
            label_path (str): Path to .npy file containing labels (N,)
            device (str or torch.device): Device to move tensors to.
        """
        self.images = torch.tensor(np.load(image_path), dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.load(label_path), dtype=torch.long, device=device)

        self.device = device

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

class MNISTRandomLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, label_seed, device):
        """
        Args:
            image_path (str): Path to .npy file containing image data (N, C, H, W).
            label_path (str): Path to .npy file containing labels (N,)
            device (str or torch.device): Device to move tensors to.
        """
        self.train_dataset = get_mnist_dataset()
        
        self.rng = torch.Generator()
        self.rng.manual_seed(label_seed)
        self.labels = torch.randint(0, 10, size=(len(self.train_dataset),), generator=self.rng, dtype=torch.long)
        self.device = device

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        image = self.train_dataset[idx][0]
        label = self.labels[idx]
        return image, label

def get_mnist_dataset() -> torch.utils.data.Dataset:
    normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ]), download=True)
    dataset, _ = torch.utils.data.random_split(
        dataset,
        [int(len(dataset)*0.9), int(len(dataset)*0.1)],
        generator=torch.Generator().manual_seed(42)
    )
    return dataset

def generate_gaussian_noise_dataset():
    NUM_IMAGES = 50000
    rng = RandomState(MT19937(SeedSequence(4145114)))

    data = rng.randn(NUM_IMAGES, 1, 28, 28)
    labels = rng.randint(low=0, high=10, size=(NUM_IMAGES,)).astype(np.long)

    np.save(f"data/GaussianNoise/noise-{NUM_IMAGES}-images.npy", data)
    np.save(f"data/GaussianNoise/noise-{NUM_IMAGES}-labels.npy", labels)


dataset_factories = {
    "mnist": lambda _: get_mnist_dataset(),
    "mnistrandom": lambda device: list(MNISTRandomLabelsDataset(138914, device)),
    "gaussiannoise": lambda device: GaussianNoiseDataset("data/GaussianNoise/noise-50000-images.npy", "data/GaussianNoise/noise-50000-labels.npy", device)
}

if __name__ == "__main__":
    print("Generating Gaussian noise dataset")
    generate_gaussian_noise_dataset()
