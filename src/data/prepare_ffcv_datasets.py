import torchvision.datasets
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import os
import argparse
import sys

from .datasets import create_train_val_test_split

def prepare_cifar10(out_paths=["ffcv/cifar10_train.beton", "ffcv/cifar10_val.beton", "ffcv/cifar10_test.beton"],
                    val_split=0.1, train_val_split_seed=42):
    train_dataset = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10('/tmp', train=False, download=True)

    train_dataset, val_dataset, _ = create_train_val_test_split(train_dataset, val_split=val_split, test_split=0.0, seed=train_val_split_seed)

    for dataset, out_path in zip([train_dataset, val_dataset, test_dataset], out_paths):
        writer = DatasetWriter(out_path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(dataset)

def prepare_cifar100(out_paths=["ffcv/cifar100_train.beton", "ffcv/cifar100_val.beton", "ffcv/cifar100_test.beton"],
                    val_split=0.1, train_val_split_seed=42):
    train_dataset = torchvision.datasets.CIFAR100('/tmp', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR100('/tmp', train=False, download=True)

    train_dataset, val_dataset, _ = create_train_val_test_split(train_dataset, val_split=val_split, test_split=0.0, seed=train_val_split_seed)

    for dataset, out_path in zip([train_dataset, val_dataset, test_dataset], out_paths):
        writer = DatasetWriter(out_path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(dataset)

if __name__ == "__main__":
    methods = {
        "--cifar10": prepare_cifar10,
        "--cifar100": prepare_cifar100
    }
    USAGE_STRING = f"Usage: In repository root, run `python -m src.data.prepare_ffcv_datasets {' '.join(methods.keys())}`."
    parser = argparse.ArgumentParser(prog="prepare_ffcv_datasets.py", description=f"Download datasets and preprocess for FFCV into .beton files. {USAGE_STRING}")
    for dataset_key in methods:
        parser.add_argument(dataset_key, action="store_true")

    args = parser.parse_args()
    if not any(args.__dict__[dataset_key.removeprefix("--")] for dataset_key in methods):
        print(f"No dataset specified! {USAGE_STRING}")
        sys.exit()

    os.makedirs("ffcv", exist_ok=True)
    for dataset_key, method in methods.items():
        if args.__dict__[dataset_key.removeprefix("--")]:
            print("Preparing dataset:", dataset_key.removeprefix("--"))
            method()