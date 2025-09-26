#!/usr/bin/env python3
"""
Main training script using Hydra configuration management.
Supports both single and multi-model training.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
import os
import traceback
import sys
from pathlib import Path

from src.core.registry import build_component
from src.core.trainer import Trainer
from src.data.datasets import create_dataset, create_dataloader, create_train_val_test_split
from src.models.mlp import create_mlp
from src.models.resnet import create_resnet
from src.models.gnn import create_gnn
from src.optimizers.optimizers import create_optimizer
from src.optimizers.schedulers import create_scheduler
from src.utils.mask_utils import save_masks, load_masks
from src.utils.interpolation import (
    interpolate_models_comprehensive, 
    evaluate_midpoint_models,
    evaluate_checkpoint_interpolation,
    save_interpolation_results
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_model(cfg: DictConfig, output_dir: Path, model_idx: int = None) -> dict:
    """Train a single model and return results."""
    
    # Create datasets
    if cfg.dataset.name in ['mnist', 'cifar10', 'cifar100']:
        train_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir, train=True)
        test_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir, train=False)
        
        # Create train/val split
        train_dataset, val_dataset = create_train_val_test_split(
            train_dataset, val_split=cfg.dataset.val_split, test_split=0.0, seed=cfg.seed
        )[0:2]
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=cfg.dataset.batch_size, 
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory
        )
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=cfg.dataset.batch_size, 
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory
        )
        test_loader = create_dataloader(
            test_dataset, 
            batch_size=cfg.dataset.batch_size, 
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory
        )
        
        # Create model
        if cfg.model.name == 'mlp_mnist':
            model = create_mlp(
                symmetry=cfg.model.symmetry,
                input_dim=cfg.model.input_dim,
                hidden_dim=cfg.model.hidden_dim,
                output_dim=cfg.model.output_dim,
                num_layers=cfg.model.num_layers,
                mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                norm=cfg.model.norm
            )
        elif cfg.model.name == 'resnet_cifar':
            model = create_resnet(
                symmetry=cfg.model.symmetry,
                depth=cfg.model.depth,
                w=cfg.model.w,
                mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                num_classes=cfg.model.num_classes
            )
        else:
            raise ValueError(f"Unknown model: {cfg.model.name}")
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
        scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
            logging={
                **cfg.logging,
                'name': f"{cfg.experiment_name}_model_{model_idx + 1}" if model_idx is not None else cfg.experiment_name
            },
            print_summary=(model_idx == 0) if model_idx is not None else True
        )
        
        # Train
        results = trainer.train(
            num_epochs=cfg.training.num_epochs,
            val_every=cfg.training.val_every,
            save_every=cfg.training.save_every,
            save_path=str(output_dir) if cfg.training.save_path is None else cfg.training.save_path,
            early_stopping=cfg.training.early_stopping,
            save_grad_every=cfg.training.save_grad_every,
            save_params_every=cfg.training.save_params_every
        )
        
        return results, trainer
    
    elif cfg.dataset.name == 'arxiv':
        # GNN training
        try:
            from ogb.nodeproppred import Evaluator
        except ImportError:
            raise ImportError("ogb is required for ArXiv dataset. Install with: pip install ogb")
        
        # Create dataset
        dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
        data = dataset[0]
        data = data.to(cfg.device)
        split_idx = dataset.get_idx_split()
        
        # Create model
        C_lst = None
        if cfg.model.model_type in ['asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn']:
            import math
            C_lst = [0.01 * torch.randn(cfg.model.hidden_channels, cfg.model.hidden_channels) / 
                    math.sqrt(cfg.model.hidden_channels) for _ in range(cfg.model.num_layers)]
        
        model = create_gnn(
            model_type=cfg.model.model_type,
            in_channels=data.num_features,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            C_lst=C_lst
        )
        
        # Create optimizer
        optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
        
        # Create scheduler
        scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
        
        # GNN-specific training loop
        evaluator = Evaluator(name='ogbn-arxiv')
        
        def train_gnn_epoch():
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.adj_t)[split_idx['train']]
            loss = torch.nn.functional.nll_loss(out, data.y.squeeze(1)[split_idx['train']])
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            return loss.item()
        
        def test_gnn():
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.adj_t)
                train_acc = evaluator.eval({
                    'y_true': data.y[split_idx['train']],
                    'y_pred': out[split_idx['train']].argmax(dim=-1, keepdim=True),
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': data.y[split_idx['valid']],
                    'y_pred': out[split_idx['valid']].argmax(dim=-1, keepdim=True),
                })['acc']
                test_acc = evaluator.eval({
                    'y_true': data.y[split_idx['test']],
                    'y_pred': out[split_idx['test']].argmax(dim=-1, keepdim=True),
                })['acc']
            return train_acc, val_acc, test_acc
        
        # Training loop
        best_val_acc = 0
        for epoch in range(cfg.training.num_epochs):
            loss = train_gnn_epoch()
            train_acc, val_acc, test_acc = test_gnn()
            
            if epoch % cfg.training.log_steps == 0:
                print(f'Epoch {epoch+1:03d}: Loss: {loss:.4f}, '
                      f'Train: {100*train_acc:.2f}%, Val: {100*val_acc:.2f}%, '
                      f'Test: {100*test_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        
        print(f'Best validation accuracy: {100*best_val_acc:.2f}%')
        
        # Return results in expected format
        results = {
            'final_test_metrics': {
                'test_accuracy': test_acc,
                'test_loss': 0.0  # Not computed for GNN
            }
        }
        
        return results, None
    
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set seed
    set_seed(cfg.seed)
    
    # Create output directory
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    print(f"Starting experiment: {cfg.experiment_name}")
    print(f"Output directory: {output_dir}")
    
    # Check if this is a multi-model experiment
    num_models = cfg.get('num_models', 1)
    use_fixed_masks = cfg.get('use_fixed_masks', False)
    
    if num_models > 1:
        print(f"Multi-model experiment: {num_models} models")
        print(f"Use fixed masks: {use_fixed_masks}")
        
        # Store checksums for verification
        mask_checksums = []
        param_checksums = []
        
        # Generate fixed masks if needed
        if use_fixed_masks:
            print("Generating fixed masks...")
            # Create a dummy model to generate masks
            if cfg.model.name == 'mlp_mnist':
                dummy_model = create_mlp(
                    symmetry=cfg.model.symmetry,
                    input_dim=cfg.model.input_dim,
                    hidden_dim=cfg.model.hidden_dim,
                    output_dim=cfg.model.output_dim,
                    num_layers=cfg.model.num_layers,
                    mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                    norm=cfg.model.norm
                )
            elif cfg.model.name == 'resnet_cifar':
                dummy_model = create_resnet(
                    symmetry=cfg.model.symmetry,
                    depth=cfg.model.depth,
                    w=cfg.model.w,
                    mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                    num_classes=cfg.model.num_classes
                )
            else:
                raise ValueError(f"Unknown model: {cfg.model.name}")
            
            save_masks(dummy_model, output_dir)
            print(f"Generated and saved masks to {output_dir / 'fixed_masks'}")
        
        # Train multiple models
        model_results = []
        for model_idx in range(num_models):
            print(f"\n=== Training Model {model_idx + 1}/{num_models} ===")
            
            # Load fixed masks if using them
            fixed_masks = None
            if use_fixed_masks:
                try:
                    fixed_masks = load_masks(output_dir)
                    print(f"Loaded fixed masks for model {model_idx + 1}")
                except FileNotFoundError:
                    print("Warning: Fixed masks not found, using random masks")
            
            # Create model with fixed masks if available
            if cfg.model.name == 'mlp_mnist':
                model = create_mlp(
                    symmetry=cfg.model.symmetry,
                    input_dim=cfg.model.input_dim,
                    hidden_dim=cfg.model.hidden_dim,
                    output_dim=cfg.model.output_dim,
                    num_layers=cfg.model.num_layers,
                    mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                    norm=cfg.model.norm,
                    fixed_masks=fixed_masks
                )
            elif cfg.model.name == 'resnet_cifar':
                model = create_resnet(
                    symmetry=cfg.model.symmetry,
                    depth=cfg.model.depth,
                    w=cfg.model.w,
                    mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                    num_classes=cfg.model.num_classes,
                    fixed_masks=fixed_masks
                )
            else:
                raise ValueError(f"Unknown model: {cfg.model.name}")
            
            # Create datasets (reuse for all models)
            if model_idx == 0:
                train_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir, train=True)
                test_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir, train=False)
                
                train_dataset, val_dataset = create_train_val_test_split(
                    train_dataset, val_split=cfg.dataset.val_split, test_split=0.0, seed=cfg.seed
                )[0:2]
                
                train_loader = create_dataloader(
                    train_dataset, 
                    batch_size=cfg.dataset.batch_size, 
                    shuffle=True,
                    num_workers=cfg.dataset.num_workers,
                    pin_memory=cfg.dataset.pin_memory
                )
                val_loader = create_dataloader(
                    val_dataset, 
                    batch_size=cfg.dataset.batch_size, 
                    shuffle=False,
                    num_workers=cfg.dataset.num_workers,
                    pin_memory=cfg.dataset.pin_memory
                )
                test_loader = create_dataloader(
                    test_dataset, 
                    batch_size=cfg.dataset.batch_size, 
                    shuffle=False,
                    num_workers=cfg.dataset.num_workers,
                    pin_memory=cfg.dataset.pin_memory
                )
            
            # Create optimizer and scheduler
            optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
            scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=cfg.device,
                logging={
                    **cfg.logging,
                    'name': f"{cfg.experiment_name}_model_{model_idx + 1}"
                },
                print_summary=(model_idx == 0)  # Only print summary for first model
            )
            
            # Collect parameter checksum at initialization
            if hasattr(trainer, '_compute_param_checksum'):
                param_checksum = trainer._compute_param_checksum()
                param_checksums.append(param_checksum)
                print(f"Model {model_idx + 1} parameter checksum: {param_checksum}")
            
            # Override analysis settings if enabled
            save_grad_every = getattr(cfg, 'save_grad_every', cfg.training.save_grad_every)
            save_params_every = getattr(cfg, 'save_params_every', cfg.training.save_params_every)
            
            # Train
            results = trainer.train(
                num_epochs=cfg.training.num_epochs,
                val_every=cfg.training.val_every,
                save_every=cfg.training.save_every,
                save_path=str(output_dir) if cfg.training.save_path is None else cfg.training.save_path,
                early_stopping=cfg.training.early_stopping,
                save_grad_every=save_grad_every,
                save_params_every=save_params_every
            )
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, output_dir / f"model_{model_idx + 1}.pt")
            
            model_results.append(results)
            
            # Collect mask checksum for verification
            if hasattr(trainer, '_compute_mask_checksum'):
                mask_checksum = trainer._compute_mask_checksum()
                mask_checksums.append(mask_checksum)
                print(f"Model {model_idx + 1} mask checksum: {mask_checksum}")
            
            print(f"Model {model_idx + 1} completed. Final test accuracy: {results['final_test_metrics']['test_accuracy']:.4f}")
        
        # Perform LMC interpolation if enabled
        if cfg.interpolation.enabled and num_models >= 2:
            print("\nPerforming LMC interpolation test...")
            
            # Prepare model creation function
            def create_model_for_interpolation():
                if cfg.model.name == 'mlp_mnist':
                    return create_mlp(
                        symmetry=cfg.model.symmetry,
                        input_dim=cfg.model.input_dim,
                        hidden_dim=cfg.model.hidden_dim,
                        output_dim=cfg.model.output_dim,
                        num_layers=cfg.model.num_layers,
                        mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                        norm=cfg.model.norm,
                        fixed_masks=fixed_masks
                    )
                elif cfg.model.name == 'resnet_cifar':
                    return create_resnet(
                        symmetry=cfg.model.symmetry,
                        depth=cfg.model.depth,
                        w=cfg.model.w,
                        mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                        num_classes=cfg.model.num_classes,
                        fixed_masks=fixed_masks
                    )
                else:
                    raise ValueError(f"Unknown model: {cfg.model.name}")
            
            # Determine interpolation type based on config and number of models
            interpolation_type = cfg.interpolation.type
            
            if interpolation_type == 'auto':
                if num_models == 2:
                    interpolation_type = 'grid'
                else:
                    interpolation_type = 'midpoint'
            
            # Validate interpolation type
            if interpolation_type == 'grid' and num_models != 2:
                print(f"Warning: Grid interpolation requires exactly 2 models, but {num_models} provided. Using midpoint instead.")
                interpolation_type = 'midpoint'
            elif interpolation_type == 'midpoint' and num_models < 2:
                print(f"Warning: Midpoint evaluation requires at least 2 models, but {num_models} provided. Skipping LMC.")
                return
            
            if interpolation_type == 'grid':
                # Grid interpolation between 2 models
                print("Performing grid interpolation between 2 models...")
                
                # Load the two models
                model1_state = torch.load(output_dir / "model_1.pt", map_location=cfg.device)
                model2_state = torch.load(output_dir / "model_2.pt", map_location=cfg.device)
                
                # Create fresh models
                model1 = create_model_for_interpolation()
                model2 = create_model_for_interpolation()
                
                model1.load_state_dict(model1_state['model_state_dict'])
                model2.load_state_dict(model2_state['model_state_dict'])
                model1.to(cfg.device)
                model2.to(cfg.device)
                
                # Perform comprehensive interpolation
                interpolation_results = interpolate_models_comprehensive(
                    model1, model2, train_loader, val_loader, test_loader,
                    steps=cfg.interpolation.steps,
                    device=cfg.device
                )
                
                print(f"Grid interpolation completed successfully!")
                
            else:  # midpoint
                # Midpoint evaluation for multiple models
                print(f"Performing midpoint evaluation for {num_models} models...")
                
                # Load all models
                models = []
                for i in range(num_models):
                    model_state = torch.load(output_dir / f"model_{i+1}.pt", map_location=cfg.device)
                    model = create_model_for_interpolation()
                    model.load_state_dict(model_state['model_state_dict'])
                    model.to(cfg.device)
                    models.append(model)
                
                # Perform midpoint evaluation
                interpolation_results = evaluate_midpoint_models(
                    models, train_loader, val_loader, test_loader, cfg.device
                )
                
                print(f"Midpoint evaluation completed:")
                print(f"  Train accuracy: {interpolation_results['train_accuracy']:.4f}%")
                print(f"  Val accuracy: {interpolation_results['val_accuracy']:.4f}%")
                print(f"  Test accuracy: {interpolation_results['test_accuracy']:.4f}%")
                print(f"  Train loss: {interpolation_results['train_loss']:.4f}")
                print(f"  Val loss: {interpolation_results['val_loss']:.4f}")
                print(f"  Test loss: {interpolation_results['test_loss']:.4f}")
            
            # Save interpolation results
            save_interpolation_results(interpolation_results, output_dir)
        
        # Verify checksums
        if len(mask_checksums) > 1:
            mask_all_same = all(checksum == mask_checksums[0] for checksum in mask_checksums)
            print(f"\nMask consistency check:")
            print(f"   All models have identical masks: {'YES' if mask_all_same else 'NO'}")
            if not mask_all_same:
                print(f"   Mask checksums: {mask_checksums}")
        
        if len(param_checksums) > 1:
            param_all_different = len(set(param_checksums)) == len(param_checksums)
            print(f"\nParameter initialization check:")
            print(f"   All models have different parameters: {'YES' if param_all_different else 'NO'}")
            if not param_all_different:
                print(f"   Parameter checksums: {param_checksums}")
            else:
                print(f"   Parameter checksums: {param_checksums}")
        
        print(f"\nMulti-model experiment completed. Results saved to {output_dir}")
    
    else:
        # Single model training
        print("Single model training")
        results, trainer = train_single_model(cfg, output_dir)
        
        # LMC interpolation not available for single model training
        if cfg.interpolation.enabled:
            print("Warning: LMC interpolation requires at least 2 models. Skipping LMC for single model training.")
            print("Use num_models=2 or more to enable LMC interpolation.")
        
        print(f"Experiment completed. Results saved to {output_dir}")


if __name__ == "__main__":
    # Enable debug mode if requested
    import os
    if os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes'):
        from debug_mode import setup_debug_mode
        setup_debug_mode()
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔍 FULL STACK TRACE:")
        traceback.print_exc()
        sys.exit(1)