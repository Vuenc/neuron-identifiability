#!/usr/bin/env python3
"""
Evaluate LMC interpolation between saved model checkpoints.
"""

import argparse
import torch
from pathlib import Path
from omegaconf import OmegaConf

from src.models.mlp import create_mlp
from src.models.resnet import create_resnet
from src.data.datasets import create_dataset, create_dataloader, create_train_val_test_split
from src.utils.interpolation import evaluate_checkpoint_interpolation, save_interpolation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LMC interpolation between checkpoints")
    parser.add_argument("checkpoint_dir", type=str, help="Directory containing model checkpoints")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoints", nargs="+", help="Specific checkpoint files to use")
    parser.add_argument("--interpolation-type", choices=['grid', 'midpoint', 'auto'], default='auto',
                       help="Type of interpolation to perform")
    parser.add_argument("--steps", type=int, default=25, help="Number of interpolation steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        # Try to find config in checkpoint directory
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            config = OmegaConf.load(config_path)
        else:
            raise FileNotFoundError(f"No config file found. Please specify --config")
    
    # Find checkpoint files
    if args.checkpoints:
        checkpoint_paths = [checkpoint_dir / cp for cp in args.checkpoints]
    else:
        # Find all model checkpoints
        checkpoint_paths = sorted(list(checkpoint_dir.glob("model_*.pt")))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints: {[cp.name for cp in checkpoint_paths]}")
    
    # Determine interpolation type
    interpolation_type = args.interpolation_type
    if interpolation_type == 'auto':
        if len(checkpoint_paths) == 2:
            interpolation_type = 'grid'
        else:
            interpolation_type = 'midpoint'
    
    print(f"Using interpolation type: {interpolation_type}")
    
    # Create datasets
    if config.dataset.name in ['mnist', 'cifar10', 'cifar100']:
        train_dataset = create_dataset(config.dataset.name, config.dataset.data_dir, train=True)
        test_dataset = create_dataset(config.dataset.name, config.dataset.data_dir, train=False)
        
        train_dataset, val_dataset = create_train_val_test_split(
            train_dataset, val_split=config.dataset.val_split, test_split=0.0, seed=config.seed
        )[0:2]
        
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=config.dataset.batch_size, 
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory
        )
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=config.dataset.batch_size, 
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory
        )
        test_loader = create_dataloader(
            test_dataset, 
            batch_size=config.dataset.batch_size, 
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.name}")
    
    # Prepare model creation function
    def create_model():
        if config.model.name == 'mlp_mnist':
            return create_mlp(
                symmetry=config.model.symmetry,
                input_dim=config.model.input_dim,
                hidden_dim=config.model.hidden_dim,
                output_dim=config.model.output_dim,
                num_layers=config.model.num_layers,
                mask_params=config.model.mask_params if config.model.symmetry == 1 else None,
                norm=config.model.norm
            )
        elif config.model.name == 'resnet_cifar':
            return create_resnet(
                symmetry=config.model.symmetry,
                depth=config.model.depth,
                w=config.model.w,
                mask_params=config.model.mask_params if config.model.symmetry == 1 else None,
                num_classes=config.model.num_classes
            )
        else:
            raise ValueError(f"Unknown model: {config.model.name}")
    
    # Perform interpolation
    print("Starting interpolation evaluation...")
    results = evaluate_checkpoint_interpolation(
        checkpoint_paths=[str(cp) for cp in checkpoint_paths],
        model_class=create_model,
        model_kwargs={},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        interpolation_type=interpolation_type,
        steps=args.steps,
        device=args.device
    )
    
    # Print results
    if interpolation_type == 'grid':
        print(f"\nGrid Interpolation Results:")
        print(f"  Distance: {results['distance']:.4f}")
        print(f"  Normalized distance: {results['normalized_distance']:.6f}")
        print(f"  Barrier height: {results['barrier_height']:.4f}")
        print(f"  Linearity: {results['linearity']}")
        print(f"  Best test accuracy: {max(results['test_accuracy']):.4f}%")
        print(f"  Worst test accuracy: {min(results['test_accuracy']):.4f}%")
    else:
        print(f"\nMidpoint Evaluation Results:")
        print(f"  Train accuracy: {results['train_accuracy']:.4f}%")
        print(f"  Val accuracy: {results['val_accuracy']:.4f}%")
        print(f"  Test accuracy: {results['test_accuracy']:.4f}%")
        print(f"  Train loss: {results['train_loss']:.4f}")
        print(f"  Val loss: {results['val_loss']:.4f}")
        print(f"  Test loss: {results['test_loss']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir
    save_interpolation_results(results, output_dir, f"checkpoint_interpolation_{interpolation_type}.pt")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
