#!/usr/bin/env python3
"""
Main training script using Hydra configuration management.
"""

from __future__ import annotations
import concurrent.futures
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
import os
import traceback
import sys
from pathlib import Path

from src.core.trainer import Trainer
from src.core.gnn_trainer import GNNTrainer
from src.data.datasets import (
    create_dataset,
    create_dataloader,
    create_train_val_test_split,
    create_ffcv_dataloader_cifar10,
    create_ffcv_dataloader_cifar100
)
from src.models.mlp import create_mlp
from src.models.resnet import create_resnet
from src.models.gnn import create_gnn
from src.optimizers.optimizers import create_optimizer
from src.optimizers.schedulers import create_scheduler
from src.utils.interpolation import (
    interpolate_models, 
    evaluate_midpoint_models,
    save_interpolation_results
)
from src.utils.gnn_interpolation import interpolate_gnn_models
from src.utils.cosine_similarity import (
    compute_cossim_per_layer,
    compute_cossim_aggregate,
)
from src.utils.functional_similarity import (
    compute_functional_similarity,
    compute_functional_similarity_gnn,
    compute_functional_similarity_aggregate,
)
from src.utils.ntk import (
    compute_ntk_from_dataloader,
    compute_ntk_gnn,
    compute_ntk_stats,
    compute_cka,
    compute_linearization_agreement,
    sample_data_points,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(cfg: DictConfig, mask_seed, device=None):
    if cfg.model.name == 'mlp_mnist':
        return create_mlp(
            symmetry=cfg.model.symmetry,
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.output_dim,
            num_layers=cfg.model.num_layers,
            mask_params=cfg.model.mask_params if cfg.model.symmetry in [1, 3] or cfg.model.get('asym_init_only', False) else None,
            norm=cfg.model.norm,
            elementwise_affine=cfg.model.get('elementwise_affine', True),
            activation=cfg.model.get('activation', None),
            mask_seed=mask_seed,
            n_mul=cfg.model.get('n_mul', 1.0),
            asym_init_only=cfg.model.get('asym_init_only', False),
        )
    elif cfg.model.name == 'resnet_cifar':
        return create_resnet(
            symmetry=cfg.model.symmetry,
            depth=cfg.model.depth,
            w=cfg.model.w,
            mask_params=cfg.model.mask_params if cfg.model.symmetry in [1, 3] or cfg.model.get('asym_init_only', False) else None,
            num_classes=cfg.model.num_classes,
            mask_seed=mask_seed,
            n_mul=cfg.model.get('n_mul', 1.0),
            asym_init_only=cfg.model.get('asym_init_only', False),
        )
    elif cfg.model.name == 'gnn_arxiv':
        return create_gnn(
            symmetry=cfg.model.symmetry,
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            mask_params=cfg.model.mask_params if cfg.model.symmetry in [1, 3] else None,
            model_type=cfg.model.model_type,
            mask_seed=mask_seed,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def setup_data_loaders(cfg: DictConfig):
    if cfg.dataset.enable_ffcv:
        # Check if FFCV is actually available
        try:
            import ffcv
        except ImportError:
            print("Warning: FFCV not available, falling back to standard data loading.")
            cfg.dataset.enable_ffcv = False
        
    if cfg.dataset.enable_ffcv:
        if cfg.dataset.name not in ['cifar10', 'cifar100']:
            raise ValueError(f"FFCV data loading is not supported for dataset {cfg.dataset.name}")
        print("Using FFCV for data loading. Works best for large batch sizes (>= 256).")
        create_ffcv_dataloader = {
            "cifar10": create_ffcv_dataloader_cifar10,
            "cifar100": create_ffcv_dataloader_cifar100
        }[cfg.dataset.name]
        train_path, val_path, test_path = {
            "cifar10": ["ffcv/cifar10_train.beton", "ffcv/cifar10_val.beton", "ffcv/cifar10_test.beton"],
            "cifar100": ["ffcv/cifar100_train.beton", "ffcv/cifar100_val.beton", "ffcv/cifar100_test.beton"]
        }[cfg.dataset.name]
        train_loader = create_ffcv_dataloader(
            data_path=train_path,
            train=True,
            batch_size=cfg.dataset.batch_size,
            device=cfg.device,
            num_workers=cfg.dataset.num_workers,
        )
        val_loader = create_ffcv_dataloader(
            data_path=val_path,
            train=False,
            batch_size=cfg.dataset.batch_size,
            device=cfg.device,
            num_workers=cfg.dataset.num_workers,
        )
        test_loader = create_ffcv_dataloader(
            data_path=test_path,
            train=False,
            batch_size=cfg.dataset.batch_size,
            device=cfg.device,
            num_workers=cfg.dataset.num_workers,
        )
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
        }
    elif cfg.dataset.name in ['mnist', 'cifar10', 'cifar100', 'mnist-coherence', 'gaussian-subspace-dataset', 'multilabel-subspace-dataset', 'parity-subspace-dataset']:
        train_dataset = create_dataset(cfg_dataset=cfg.dataset, train=True)
        test_dataset = create_dataset(cfg_dataset=cfg.dataset, train=False)
        train_dataset, val_dataset = create_train_val_test_split(
            train_dataset, val_split=cfg.dataset.val_split, test_split=0.0, seed=42
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
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }        
    elif cfg.dataset.name == 'arxiv':
        dataset = create_dataset(cfg_dataset=cfg.dataset)
        data = dataset[0]
        data = data.to(cfg.device)
        split_idx = dataset.get_idx_split()
        return {
            'dataset': dataset,
            'data': data,
            'split_idx': split_idx,
            'train_loader': None,
            'val_loader': None,
            'test_loader': None
        }
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def setup_wandb(cfg: DictConfig):
    if not cfg.logging.get('use_wandb', False):
        return None

    try:
        import wandb
        wandb_config = {
            'project': cfg.logging.get('project', 'asymmetric-networks'),
            'name': cfg.logging.get('name', f"{cfg.experiment_name}_multi_model"),
            'config': dict(cfg),
        }
        
        optional_params = ['entity', 'group', 'job_type', 'tags', 'notes', 'resume', 'reinit', 'mode']
        for param in optional_params:
            if cfg.logging.get(param) is not None:
                wandb_config[param] = cfg.logging[param]
        
        run = wandb.init(**wandb_config)
        
        if run.sweep_id is not None:
            cfg.experiment_name += f"__{run.sweep_id}"
        cfg.experiment_name += f"__{run.id}"
        
        print("Initialized wandb")
        return wandb
    except ImportError:
        print("Warning: wandb not available")
        return None


def train_one(cfg: DictConfig, output_dir: Path, init_seed: int, optimization_seed: int, mask_seed: int, model_index: int, runs_in_separate_process: bool=True):
    if runs_in_separate_process:
        global print
        _print = __builtins__.print
        def print_with_model_id(*args, **kwargs):
            _print(f"[Model {model_index+1}]:", *args, **kwargs)
        print = print_with_model_id

    num_models = cfg.get('num_models', 1)

    print(f"=== Model {model_index + 1}/{num_models} ===")

    set_seed(init_seed)
    print(f"Init seed: {init_seed}")

    model = create_model(cfg, mask_seed, device=cfg.device)

    # Set optimizer seed before creating optimizer
    set_seed(optimization_seed)
    print(f"Opt seed: {optimization_seed}")
    data_info = setup_data_loaders(cfg)

    optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
    scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None

    trainer_class = GNNTrainer if cfg.model.name == 'gnn_arxiv' else Trainer
    trainer = trainer_class(
        model=model,
        data=data_info,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.device,
        model_prefix=f'model_{model_index + 1}',
        shared_wandb=True,
        logging=cfg.logging,
        print_summary=(model_index == 0)
    )

    save_grad_every = getattr(cfg, 'save_grad_every', cfg.training.save_grad_every)
    save_params_every = getattr(cfg, 'save_params_every', cfg.training.save_params_every)

    results = trainer.train(
        num_epochs=cfg.training.num_epochs,
        val_every=cfg.training.val_every,
        save_every=cfg.training.save_every,
        save_path=str(output_dir) if cfg.training.save_path is None else cfg.training.save_path,
        early_stopping=None,
        save_grad_every=save_grad_every,
        save_params_every=save_params_every,
        model_idx=model_index
    )
    return results


def train_multi(cfg: DictConfig, init_seeds: list, optimization_seeds: list, mask_seed: int):
    num_models = cfg.get('num_models', 1)
    print(f"Training {num_models} models")
    
    wandb = setup_wandb(cfg)
    
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    print(f"Output dir: {output_dir}")

    # Training (in parallel)
    max_parallel_processes = min(num_models, cfg.max_parallel_processes)
    if max_parallel_processes > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_processes) as executor:
            model_results = list(executor.map(
                train_one,
                *zip(*[(cfg, output_dir, init_seeds[model_index], optimization_seeds[model_index], mask_seed, model_index) for model_index in range(num_models)])
            ))
    else:
        model_results = list(map(
                train_one,
                *zip(*[(cfg, output_dir, init_seeds[model_index], optimization_seeds[model_index], mask_seed, model_index) for model_index in range(num_models)])
            ))

    evaluate_all(cfg, output_dir, mask_seed)

    if wandb:
        try:
            wandb.finish()
            print("Finished wandb")
        except ImportError:
            print("Warning: wandb not available")

    print(f"\nMulti-model completed. Results saved to {output_dir}")
    return model_results

def evaluate_all(cfg: DictConfig, output_dir: Path, mask_seed: int):
    num_models = cfg.get('num_models', 1)
    cossim_enabled = (cfg.training.cosine_similarity.enabled and 
                     num_models == 2 and 
                     cfg.training.cosine_similarity.save_every is not None)

    func_sim_enabled = (cfg.training.functional_similarity.enabled and 
                       num_models == 2 and 
                       cfg.training.functional_similarity.save_every is not None)
    ntk_enabled = (cfg.training.get('ntk_tracking', {}).get('enabled', False) and
                   cfg.training.get('ntk_tracking', {}).get('save_every') is not None)

    if cossim_enabled and num_models == 2:
        print("Cossim enabled")
        cossim_analysis(cfg, output_dir)

    data_info = None
    if func_sim_enabled and num_models == 2:
        data_info = data_info or setup_data_loaders(cfg)
        print("Funcsim enabled")
        functional_similarity_analysis(cfg, output_dir, data_info, mask_seed=mask_seed)

    if ntk_enabled:
        data_info = data_info or setup_data_loaders(cfg)
        print("NTK tracking enabled")
        ntk_tracking_analysis(cfg, output_dir, data_info)

    if cfg.interpolation.enabled and num_models >= 2:
        data_info = data_info or setup_data_loaders(cfg)
        save_every = cfg.interpolation.get('save_every')
        epochs_to_save = [None] if save_every is None else range(0, cfg.training.num_epochs + 1, save_every)
        for epoch in epochs_to_save:
            interpolation_analysis(cfg, output_dir, data_info, epoch=epoch, mask_seed=mask_seed)
    

def cossim_analysis(cfg: DictConfig, output_dir: Path):
    print("\nComputing cossim...")
    import gc
    
    # First, collect the list of epochs that have checkpoints (without loading them)
    available_epochs = []
    for epoch in range(0, cfg.training.num_epochs + 1):
        if cfg.training.save_every is None or epoch % cfg.training.save_every == 0:
            checkpoint_path1 = output_dir / f"checkpoint_epoch_{epoch}_model_1.pt"
            checkpoint_path2 = output_dir / f"checkpoint_epoch_{epoch}_model_2.pt"
            if checkpoint_path1.exists() and checkpoint_path2.exists():
                available_epochs.append(epoch)
            else:
                if epoch > 0:  # Allow missing epoch 0, but break if later epochs are missing
                    break
    
    if len(available_epochs) == 0:
        print("Warning: No checkpoints found")
        return
    
    print(f"Found {len(available_epochs)} checkpoints")
    all_epoch_results = []
    
    # Use rolling window: only keep current and previous checkpoint in memory
    prev_checkpoint1 = None
    prev_checkpoint2 = None
    
    for i, epoch in enumerate(available_epochs):
        # Load current checkpoints
        try:
            current_checkpoint1 = torch.load(
                output_dir / f"checkpoint_epoch_{epoch}_model_1.pt", 
                map_location='cpu'
            )
            current_checkpoint2 = torch.load(
                output_dir / f"checkpoint_epoch_{epoch}_model_2.pt", 
                map_location='cpu'
            )
        except FileNotFoundError:
            print(f"Warning: Checkpoint epoch {epoch} not found, skipping")
            continue
        
        # For first epoch (i=0, epoch=0), store as previous checkpoint for next iteration
        # We'll compute epoch 0->1 comparison in the next iteration (i=1)
        if i == 0:
            prev_checkpoint1 = current_checkpoint1
            prev_checkpoint2 = current_checkpoint2
            # Clean up current references (they're now in prev_*)
            current_checkpoint1 = None
            current_checkpoint2 = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue  # Continue to next iteration where we'll compute epoch 0->1
        
        # Compute updates using current and previous checkpoints
        prev_params1 = prev_checkpoint1['model_state_dict']
        prev_params2 = prev_checkpoint2['model_state_dict']
        
        current_params1 = current_checkpoint1['model_state_dict']
        current_params2 = current_checkpoint2['model_state_dict']

        updates1 = {}
        updates2 = {}
        for name in current_params1:
            if name in prev_params1:
                updates1[name] = current_params1[name] - prev_params1[name]
        for name in current_params2:
            if name in prev_params2:
                updates2[name] = current_params2[name] - prev_params2[name]

        per_layer_similarities = compute_cossim_per_layer(updates1, updates2, current_checkpoint1['trainable'])
        aggregate_similarity = compute_cossim_aggregate(updates1, updates2, current_checkpoint1['trainable'])
        
        epoch_results = {
            'epoch': epoch,
            'per_layer_similarities': per_layer_similarities,
            'aggregate_similarity': aggregate_similarity,
        }
        
        all_epoch_results.append(epoch_results)

        print(f"Epoch {epoch} cossim: {aggregate_similarity:.4f}")
        print(f"Epoch {epoch} per layer: {per_layer_similarities}")
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'cossim_aggregate': aggregate_similarity})

                    for param_name, similarity in per_layer_similarities.items():
                        wandb.log({
                            f'cossim_{param_name}': similarity
                        })
            except ImportError:
                print("Warning: wandb not available")
        
        # Clear updates dictionaries (they may contain large tensors)
        del updates1, updates2
        del current_params1, current_params2, prev_params1, prev_params2
        
        # Move to next iteration: current becomes previous
        prev_checkpoint1 = current_checkpoint1
        prev_checkpoint2 = current_checkpoint2
        current_checkpoint1 = None
        current_checkpoint2 = None
        
        # Force garbage collection and clear GPU cache to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save({
        'all_epoch_results': all_epoch_results,
        'num_epochs': len(all_epoch_results)
    }, output_dir / "cossim_all_epochs.pt")
    
    print(f"\nSaved cossim results to {output_dir}/cossim_all_epochs.pt")


def functional_similarity_analysis(cfg: DictConfig, output_dir: Path, data_info: dict, mask_seed: int):
    """Perform functional similarity analysis between two models."""
    print("\nComputing functional similarity...")
    import gc
    
    # First, collect the list of epochs that have checkpoints (without loading them)
    available_epochs = []
    for epoch in range(0, cfg.training.num_epochs + 1):
        if cfg.training.save_every is None or epoch % cfg.training.save_every == 0:
            checkpoint_path1 = output_dir / f"checkpoint_epoch_{epoch}_model_1.pt"
            checkpoint_path2 = output_dir / f"checkpoint_epoch_{epoch}_model_2.pt"
            if checkpoint_path1.exists() and checkpoint_path2.exists():
                available_epochs.append(epoch)
            else:
                if epoch > 0:  # Allow missing epoch 0, but break if later epochs are missing
                    break
    
    if len(available_epochs) == 0:
        print("Warning: No checkpoints found")
        return
    
    print(f"Found {len(available_epochs)} checkpoints")
    all_epoch_results = []
    
    # Determine device for models
    if cfg.model.name == 'gnn_arxiv':
        device = data_info['data'].x.device
    else:
        device = cfg.device
    
    # Process each checkpoint one at a time to avoid memory issues
    for epoch in available_epochs:
        # Load checkpoints for this epoch only
        try:
            checkpoint1 = torch.load(
                output_dir / f"checkpoint_epoch_{epoch}_model_1.pt", 
                map_location='cpu'
            )
            checkpoint2 = torch.load(
                output_dir / f"checkpoint_epoch_{epoch}_model_2.pt", 
                map_location='cpu'
            )
        except FileNotFoundError:
            print(f"Warning: Checkpoint epoch {epoch} not found, skipping")
            continue
        
        # Create models for evaluation
        model1 = create_model(cfg, mask_seed, device='cpu')
        model2 = create_model(cfg, mask_seed, device='cpu')
        
        # Load model states
        model1.load_state_dict(checkpoint1['model_state_dict'])
        model2.load_state_dict(checkpoint2['model_state_dict'])
        
        # Move models to appropriate device
        model1 = model1.to(device)
        model2 = model2.to(device)
        
        # Compute functional similarity
        if cfg.model.name == 'gnn_arxiv':
            # GNN case - use split indices
            results = compute_functional_similarity_gnn(
                model1, model2, 
                data_info['data'], 
                data_info['split_idx'], 
                device=device
            )
        else:
            # Standard case - use data loaders
            results = {}
            for split in cfg.training.functional_similarity.splits:
                # Map split names to data_info keys
                split_key = f'{split}_loader' if f'{split}_loader' in data_info else split
                if split_key in data_info and data_info[split_key] is not None:
                    split_results = compute_functional_similarity(
                        model1, model2, 
                        data_info[split_key], 
                        device=device
                    )
                    # Add split prefix to results
                    for key, value in split_results.items():
                        results[f'{split}_{key}'] = value
        
        # Compute aggregate similarity
        aggregate_similarity = compute_functional_similarity_aggregate(results)
        
        epoch_results = {
            'epoch': epoch,
            'split_results': results,
            'aggregate_similarity': aggregate_similarity,
        }
        
        all_epoch_results.append(epoch_results)
        
        # Format individual split similarities
        train_sim = results.get('train_funcsim', 0.0)
        # Handle both 'val' and 'valid' split names
        val_sim = results.get('val_funcsim', results.get('valid_funcsim', 0.0))
        test_sim = results.get('test_funcsim', 0.0)
        
        print(f"Epoch {epoch} funcsim: {aggregate_similarity:.4f} (Train: {train_sim:.4f}, Val: {val_sim:.4f}, Test: {test_sim:.4f})")
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'funcsim_aggregate': aggregate_similarity})
                    for key, value in results.items():
                        if key.endswith('_funcsim'):
                            wandb.log({key: value})
            except ImportError:
                print("Warning: wandb not available")
        
        # Clean up models and checkpoints for this epoch
        del model1, model2
        del checkpoint1, checkpoint2
        del results
        
        # Force garbage collection and clear GPU cache to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save({
        'all_epoch_results': all_epoch_results,
        'num_epochs': len(all_epoch_results)
    }, output_dir / "funcsim_all_epochs.pt")
    
    print(f"\nSaved functional similarity results to {output_dir}/funcsim_all_epochs.pt")


def ntk_tracking_analysis(cfg: DictConfig, output_dir: Path, data_info: dict):
    """Perform NTK tracking analysis by computing NTK matrix at different epochs and tracking changes."""
    print("\nComputing NTK tracking...")
    import gc
    import numpy as np
    
    ntk_config = cfg.training.get('ntk_tracking', {})
    num_samples = ntk_config.get('num_samples', 100)
    class_balanced = ntk_config.get('class_balanced', True)
    split = ntk_config.get('split', 'train')
    compute_ntk_stats = ntk_config.get('compute_ntk_stats', True)
    compute_linearization = ntk_config.get('compute_linearization', True)
    
    # First, collect the list of epochs that have checkpoints (without loading them)
    num_models = cfg.get('num_models', 1)
    available_epochs = []
    for epoch in range(0, cfg.training.num_epochs + 1):
        if cfg.training.save_every is None or epoch % cfg.training.save_every == 0:
            # Check if checkpoint exists (use model 1 for single model, or check all models)
            checkpoint_exists = True
            for model_idx in range(1, num_models + 1):
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}_model_{model_idx}.pt"
                if not checkpoint_path.exists():
                    checkpoint_exists = False
                    break
            
            if checkpoint_exists:
                available_epochs.append(epoch)
            else:
                if epoch > 0:  # Allow missing epoch 0, but break if later epochs are missing
                    break
    
    if len(available_epochs) == 0:
        print("Warning: No checkpoints found")
        return
    
    print(f"Found {len(available_epochs)} checkpoints")
    all_epoch_results = []
    
    # Determine device for models
    if cfg.model.name == 'gnn_arxiv':
        device = data_info['data'].x.device
    else:
        device = cfg.device
    
    # Get data loader for the specified split (prepare once for reuse)
    if cfg.model.name == 'gnn_arxiv':
        # For GNN, we'll use split indices
        if split not in data_info.get('split_idx', {}):
            print(f"Warning: Split '{split}' not found in split_idx, using 'train'")
            split = 'train'
        # Prepare indices once (will reuse same indices across epochs)
        split_idx = data_info['split_idx'][split]
        if len(split_idx) == 0:
            print(f"Warning: Split '{split}' has no indices")
            return
        
        # Sample indices if needed
        if num_samples < len(split_idx):
            labels = data_info['data'].y[split_idx].cpu()
            unique_classes = torch.unique(labels).tolist()
            
            from collections import defaultdict
            indices_by_class = defaultdict(list)
            for i, label in enumerate(labels):
                indices_by_class[label.item()].append(i)
            
            samples_per_class = num_samples // len(unique_classes)
            remainder = num_samples % len(unique_classes)
            
            generator = torch.Generator()
            generator.manual_seed(42)
            
            selected_local_indices = []
            for i, class_idx in enumerate(unique_classes):
                class_indices = indices_by_class[class_idx]
                num_class_samples = samples_per_class + (1 if i < remainder else 0)
                num_class_samples = min(num_class_samples, len(class_indices))
                
                perm = torch.randperm(len(class_indices), generator=generator)
                selected_class_indices = [class_indices[perm[j].item()] for j in range(num_class_samples)]
                selected_local_indices.extend(selected_class_indices)
            
            perm = torch.randperm(len(selected_local_indices), generator=generator)
            selected_local_indices = [selected_local_indices[perm[i].item()] for i in range(len(selected_local_indices))]
            
            fixed_indices = split_idx[selected_local_indices[:num_samples]]
        else:
            fixed_indices = split_idx
    else:
        # For standard models, use data loaders
        split_key = f'{split}_loader' if f'{split}_loader' in data_info else 'train_loader'
        if split_key not in data_info or data_info[split_key] is None:
            print(f"Warning: Split '{split}' loader not found, using 'train' loader")
            split_key = 'train_loader'
        data_loader = data_info[split_key]
    
    # Store initialization NTK (epoch 0) for comparison
    init_ntk_matrix = None
    
    # For NTK tracking, we typically use model 1 (or just one model)
    model_idx = 1
    
    # First, compute and store initialization NTK (epoch 0)
    if 0 not in available_epochs:
        print("Warning: Initialization checkpoint (epoch 0) not found, cannot compare to initialization")
        return
    
    try:
        init_checkpoint = torch.load(
            output_dir / f"checkpoint_epoch_0_model_{model_idx}.pt", 
            map_location='cpu'
        )
        
        init_model = create_model(cfg, device='cpu')
        init_model.load_state_dict(init_checkpoint['model_state_dict'])
        init_model = init_model.to(device)
        
        # Store init model state dict on CPU for linearization agreement (if needed)
        init_model_state_dict = None
        selected_data = None
        init_ntk_matrix = None
        
        if compute_linearization:
            init_model_state_dict = init_checkpoint['model_state_dict'].copy()
        
        # Compute initialization NTK and get selected data points (if needed)
        if compute_ntk_stats:
            if cfg.model.name == 'gnn_arxiv':
                init_ntk_matrix, _ = compute_ntk_gnn(
                    init_model, 
                    data_info['data'], 
                    fixed_indices,
                    device=device
                )
                init_ntk_matrix = init_ntk_matrix.cpu()
            else:
                init_ntk_matrix, _, selected_data = compute_ntk_from_dataloader(
                    init_model,
                    data_loader,
                    num_samples=num_samples,
                    class_balanced=class_balanced,
                    seed=42,
                    device=device,
                    return_selected_data=compute_linearization
                )
                init_ntk_matrix = init_ntk_matrix.cpu()
                if selected_data is not None:
                    selected_data = selected_data.cpu()
        
        # Get selected data points for linearization (without computing NTK)
        if compute_linearization and not compute_ntk_stats:
            if cfg.model.name == 'gnn_arxiv':
                # For GNN, store the fixed indices as "selected data"
                # (we'll need to handle this differently in linearization agreement)
                pass  # TODO: handle GNN case
            else:
                selected_data = sample_data_points(
                    data_loader,
                    num_samples=num_samples,
                    class_balanced=class_balanced,
                    seed=42
                )
                selected_data = selected_data.cpu()
        
        # Move init model back to CPU for storage
        init_model = init_model.cpu()
        
        if compute_ntk_stats:
            print(f"Computed initialization NTK (epoch 0)")
        if compute_linearization:
            print(f"Prepared data for linearization agreement computation")
    except Exception as e:
        print(f"Error computing initialization NTK: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Store previous epoch NTK for step-to-step comparison (only if computing NTK stats)
    prev_ntk_matrix = None
    
    # Now compute NTK for each epoch and compare to both initialization and previous step
    for epoch in available_epochs:
        # Skip epoch 0 since we already computed it and comparing to itself is not informative
        if epoch == 0:
            if compute_ntk_stats and init_ntk_matrix is not None:
                # Store epoch 0 as previous for step-to-step comparison
                prev_ntk_matrix = init_ntk_matrix.clone()
            continue
        
        # Load checkpoint for this epoch
        try:
            checkpoint = torch.load(
                output_dir / f"checkpoint_epoch_{epoch}_model_{model_idx}.pt", 
                map_location='cpu'
            )
        except FileNotFoundError:
            print(f"Warning: Checkpoint epoch {epoch} not found, skipping")
            continue
        
        # Create model for this epoch (will be cleaned up after)
        model = create_model(cfg, device='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Initialize variables for results
        ntk_matrix_cpu = None
        stats = None
        ntk_diff_init_frobenius = None
        ntk_cosine_sim_init = None
        ntk_relative_change_init = None
        ntk_cka_init = None
        ntk_diff_prev_frobenius = None
        ntk_cosine_sim_prev = None
        ntk_relative_change_prev = None
        ntk_cka_prev = None
        
        # Compute NTK statistics (if enabled)
        if compute_ntk_stats:
            try:
                if cfg.model.name == 'gnn_arxiv':
                    ntk_matrix, stats = compute_ntk_gnn(
                        model, 
                        data_info['data'], 
                        fixed_indices,
                        device=device
                    )
                else:
                    ntk_matrix, stats, _ = compute_ntk_from_dataloader(
                        model,
                        data_loader,
                        num_samples=num_samples,
                        class_balanced=class_balanced,
                        seed=42,
                        device=device
                    )
                
                # Move to CPU for comparison
                ntk_matrix_cpu = ntk_matrix.cpu()
                
                # === Comparison to Initialization ===
                if init_ntk_matrix is not None:
                    ntk_diff_init = ntk_matrix_cpu - init_ntk_matrix  # (N, N)
                    ntk_diff_init_np = ntk_diff_init.numpy()
                    ntk_diff_init_frobenius = float(np.linalg.norm(ntk_diff_init_np, 'fro'))
                    
                    # Cosine similarity vs initialization
                    ntk_flat_init = init_ntk_matrix.flatten().numpy()
                    ntk_flat_curr = ntk_matrix_cpu.flatten().numpy()
                    norm_init = np.linalg.norm(ntk_flat_init)
                    norm_curr = np.linalg.norm(ntk_flat_curr)
                    
                    if norm_init > 0 and norm_curr > 0:
                        ntk_cosine_sim_init = float(np.dot(ntk_flat_init, ntk_flat_curr) / (norm_init * norm_curr))
                    else:
                        ntk_cosine_sim_init = 0.0
                    
                    # Relative change vs initialization
                    if norm_init > 0:
                        ntk_relative_change_init = ntk_diff_init_frobenius / norm_init
                    else:
                        ntk_relative_change_init = 0.0
                    
                    # CKA vs initialization
                    ntk_cka_init = compute_cka(init_ntk_matrix, ntk_matrix_cpu)
                
                # === Comparison to Previous Step ===
                if prev_ntk_matrix is not None:
                    ntk_diff_prev = ntk_matrix_cpu - prev_ntk_matrix  # (N, N)
                    ntk_diff_prev_np = ntk_diff_prev.numpy()
                    ntk_diff_prev_frobenius = float(np.linalg.norm(ntk_diff_prev_np, 'fro'))
                    
                    # Cosine similarity vs previous step
                    ntk_flat_prev = prev_ntk_matrix.flatten().numpy()
                    norm_prev = np.linalg.norm(ntk_flat_prev)
                    
                    if norm_prev > 0 and norm_curr > 0:
                        ntk_cosine_sim_prev = float(np.dot(ntk_flat_prev, ntk_flat_curr) / (norm_prev * norm_curr))
                    else:
                        ntk_cosine_sim_prev = 0.0
                    
                    # Relative change vs previous step
                    if norm_prev > 0:
                        ntk_relative_change_prev = ntk_diff_prev_frobenius / norm_prev
                    else:
                        ntk_relative_change_prev = 0.0
                    
                    # CKA vs previous step
                    ntk_cka_prev = compute_cka(prev_ntk_matrix, ntk_matrix_cpu)
                else:
                    # No previous step available (shouldn't happen after epoch 0)
                    ntk_diff_prev_frobenius = 0.0
                    ntk_cosine_sim_prev = 1.0
                    ntk_relative_change_prev = 0.0
                    ntk_cka_prev = 1.0
                
            except Exception as e:
                print(f"Error computing NTK for epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                # Clean up before continuing
                del model
                del checkpoint
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # === Linearization Agreement ===
        # Compute agreement between f_theta and its linearization around initialization
        linearization_agreement = None
        if compute_linearization and selected_data is not None and init_model_state_dict is not None and cfg.model.name != 'gnn_arxiv':
            try:
                # Recreate init_model from state dict
                init_model_for_lin = create_model(cfg, device='cpu')
                init_model_for_lin.load_state_dict(init_model_state_dict)
                
                # Compute linearization agreement
                linearization_agreement = compute_linearization_agreement(
                    init_model_for_lin,
                    model,
                    selected_data.to(device),
                    device=device
                )
                
                # Clean up
                del init_model_for_lin
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: Failed to compute linearization agreement for epoch {epoch}: {e}")
                linearization_agreement = None
        
        # Update previous NTK for next iteration (only if computing NTK stats)
        if compute_ntk_stats and ntk_matrix_cpu is not None:
            prev_ntk_matrix = ntk_matrix_cpu.clone()
        
        epoch_results = {
            'epoch': epoch,
            'ntk_stats': stats,
            # Comparison to initialization
            'ntk_diff_frobenius_init': ntk_diff_init_frobenius,
            'ntk_cosine_sim_init': ntk_cosine_sim_init,
            'ntk_relative_change_init': ntk_relative_change_init,
            'ntk_cka_init': ntk_cka_init,
            # Comparison to previous step
            'ntk_diff_frobenius_prev': ntk_diff_prev_frobenius,
            'ntk_cosine_sim_prev': ntk_cosine_sim_prev,
            'ntk_relative_change_prev': ntk_relative_change_prev,
            'ntk_cka_prev': ntk_cka_prev,
            # Linearization agreement
            'linearization_agreement': linearization_agreement,
            'num_samples': ntk_matrix_cpu.shape[0] if ntk_matrix_cpu is not None else num_samples,
        }
        
        # Optionally save NTK matrix (can be large, so we skip by default)
        if compute_ntk_stats and ntk_config.get('save_matrix', False) and ntk_matrix_cpu is not None:
            epoch_results['ntk_matrix'] = ntk_matrix_cpu
        
        all_epoch_results.append(epoch_results)
        
        # Print results
        if compute_ntk_stats and stats is not None:
            print(f"Epoch {epoch} NTK - Trace: {stats['trace']:.4f}, "
                  f"Max eig: {stats['max_eigenvalue']:.4f}, "
                  f"Min eig: {stats['min_eigenvalue']:.4f}, "
                  f"Condition: {stats['condition_number']:.4f}")
            if ntk_diff_init_frobenius is not None:
                print(f"Epoch {epoch} NTK vs init - Frobenius diff: {ntk_diff_init_frobenius:.4f}, "
                      f"Cosine sim: {ntk_cosine_sim_init:.4f}, "
                      f"Relative change: {ntk_relative_change_init:.4f}, "
                      f"CKA: {ntk_cka_init:.4f}")
            if ntk_diff_prev_frobenius is not None:
                print(f"Epoch {epoch} NTK vs prev - Frobenius diff: {ntk_diff_prev_frobenius:.4f}, "
                      f"Cosine sim: {ntk_cosine_sim_prev:.4f}, "
                      f"Relative change: {ntk_relative_change_prev:.4f}, "
                      f"CKA: {ntk_cka_prev:.4f}")
        if linearization_agreement is not None:
            print(f"Epoch {epoch} Linearization agreement - MSE: {linearization_agreement['mse']:.6f}, "
                  f"Relative error: {linearization_agreement['relative_error']:.6f}, "
                  f"Cosine sim: {linearization_agreement['cosine_sim']:.4f}")
            if 'accuracy_agreement' in linearization_agreement:
                print(f"  Accuracy agreement: {linearization_agreement['accuracy_agreement']:.4f}")
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb_log = {}
                    for key, value in stats.items():
                        wandb_log[f'ntk_{key}'] = value
                    # Comparison to initialization
                    wandb_log['ntk_diff_frobenius_init'] = ntk_diff_init_frobenius
                    wandb_log['ntk_cosine_sim_init'] = ntk_cosine_sim_init
                    wandb_log['ntk_relative_change_init'] = ntk_relative_change_init
                    wandb_log['ntk_cka_init'] = ntk_cka_init
                    # Comparison to previous step
                    wandb_log['ntk_diff_frobenius_prev'] = ntk_diff_prev_frobenius
                    wandb_log['ntk_cosine_sim_prev'] = ntk_cosine_sim_prev
                    wandb_log['ntk_relative_change_prev'] = ntk_relative_change_prev
                    wandb_log['ntk_cka_prev'] = ntk_cka_prev
                # Linearization agreement metrics
                if linearization_agreement is not None:
                    for key, value in linearization_agreement.items():
                        wandb_log[f'linearization_{key}'] = value
                wandb_log['epoch'] = epoch
                wandb.log(wandb_log)
            except ImportError:
                print("Warning: wandb not available")
        
        # Clean up for this epoch
        if compute_ntk_stats:
            if 'ntk_diff_init' in locals() and ntk_diff_init is not None:
                del ntk_diff_init
            if 'ntk_diff_prev' in locals() and ntk_diff_prev is not None:
                del ntk_diff_prev
            if ntk_matrix_cpu is not None and not ntk_config.get('save_matrix', False):
                del ntk_matrix_cpu
            if stats is not None:
                del stats
        del model
        del checkpoint
        
        # Force garbage collection and clear GPU cache to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    torch.save({
        'all_epoch_results': all_epoch_results,
        'num_epochs': len(all_epoch_results),
        'num_samples': num_samples,
        'class_balanced': class_balanced,
        'split': split,
    }, output_dir / "ntk_tracking_all_epochs.pt")
    
    print(f"\nSaved NTK tracking results to {output_dir}/ntk_tracking_all_epochs.pt")


def interpolation_analysis(cfg: DictConfig, output_dir: Path, data_info: dict, mask_seed: int, epoch: int | None = None):
    """Perform LMC interpolation analysis for a specific epoch or final models."""
    num_models = cfg.get('num_models', 1)
    interpolation_type = cfg.interpolation.type

    if interpolation_type == 'midpoint' and num_models < 2:
        print(f"Warning: Midpoint needs 2+ models, got {num_models}. Skipping LMC.")
        return None

    def create_model_for_interpolation():
        return create_model(cfg, mask_seed=mask_seed, device=cfg.device)

    target_epoch = epoch if epoch is not None else cfg.training.num_epochs

    interpolation_results = None
    if interpolation_type == 'grid':
        print(f"Grid interpolation between all pairs of models{(f' at epoch {epoch}'    ) if epoch is not None else ''}...")

        interpolation_table = None
        PER_STEP_METRICS_KEYS = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'test_accuracy', 'test_loss']
        SPLITS = ["train", "val", "test"]
        interpolation_results = {}
        try:
            import wandb
            if wandb.run is not None:
                interpolation_table = wandb.Table(columns=[
                    "interpolation_lambda", "model1_index", "model2_index", "model_pair",
                    *[f"interpolation_{key}" for key in PER_STEP_METRICS_KEYS]
                ])
        except ImportError:
            pass

        for model1_index in range(1, num_models+1):
            model1_state = torch.load(output_dir / f"checkpoint_epoch_{target_epoch}_model_{model1_index}.pt", map_location=cfg.device)
            for model2_index in range(model1_index+1, num_models+1):
                model2_state = torch.load(output_dir / f"checkpoint_epoch_{target_epoch}_model_{model2_index}.pt", map_location=cfg.device)

                # We could reuse the model object, but recreate it to be on the safe side
                model = create_model_for_interpolation().to(cfg.device)
                if cfg.model.name == 'gnn_arxiv':
                    interpolation_results_pair = interpolate_gnn_models(
                        model, model1_state['model_state_dict'], model2_state['model_state_dict'],
                        data_info['data'], data_info['split_idx'],
                        steps=cfg.interpolation.steps,
                        device=cfg.device,
                        rewarm=True,
                    )
                else:
                    interpolation_results_pair = interpolate_models(
                        model, model1_state['model_state_dict'], model2_state['model_state_dict'],
                        data_info['train_loader'], data_info['val_loader'], data_info['test_loader'],
                        steps=cfg.interpolation.steps,
                        device=cfg.device,
                    )
                interpolation_results[(model1_index, model2_index)] = interpolation_results_pair

                print(f"Grid interpolation{(f' at epoch {epoch}'    ) if epoch is not None else ''} (model {model1_index} vs. model {model2_index}) completed!")
                # Print summary
                print(f'  Model distance: {interpolation_results_pair["distance"]:.4f}')
                print(f'  Normalized distance: {interpolation_results_pair["normalized_distance"]:.6f}')
                print(f"  Best accuracy (train/val/test): {'/'.join([f'{max(interpolation_results_pair[f'{split}_accuracy']):.2f}%' for split in SPLITS])}")
                print(f"  Worst accuracy (train/val/test): {'/'.join([f'{min(interpolation_results_pair[f'{split}_accuracy']):.2f}%' for split in SPLITS])}")
                print(f"  Barrier height (train/val/test): {'/'.join([f'{interpolation_results_pair[f'{split}_barrier_height']:.2f}%' for split in SPLITS])}")
                print(f"  Linearity: (train/val/test): {'/'.join([f'{interpolation_results_pair[f'{split}_linearity']}' for split in SPLITS])}")

                if cfg.logging.get('use_wandb', False):
                    try:
                        import wandb
                        if wandb.run is not None:
                            # Log the metrics that are computed per interpolation step
                            assert interpolation_table is not None
                            for i, interpolation_factor in enumerate(interpolation_results_pair["lambdas"]):
                                wandb.log({
                                    'interpolation_lambda': interpolation_factor,
                                    'model1_index': model1_index, 'model2_index': model2_index,
                                    'epoch': epoch, # can be None
                                    **{f'interpolation_{key}': interpolation_results_pair[key][i] if key in interpolation_results_pair else float('nan') for key in PER_STEP_METRICS_KEYS}
                                })
                                interpolation_table.add_data(
                                    interpolation_factor, model1_index, model2_index, f"{model1_index}_{model2_index}",
                                    *[interpolation_results_pair[key][i] for key in PER_STEP_METRICS_KEYS]
                                )

                            wandb.log({
                                'interpolation_type': 'grid',
                                'model1_index': model1_index, 'model2_index': model2_index,
                                **{f'interpolation_best_{split}_accuracy': max(interpolation_results_pair[f'{split}_accuracy']) for split in SPLITS},
                                **{f'interpolation_worst_{split}_accuracy': min(interpolation_results_pair[f'{split}_accuracy']) for split in SPLITS},
                                'epoch': epoch, # can be None
                                **{f'interpolation_{split}_barrier_height': interpolation_results_pair[f'{split}_barrier_height'] for split in SPLITS},
                                'distance': interpolation_results_pair['distance']
                            })
                    except ImportError:
                        print("Warning: wandb not available")
        if interpolation_table is not None:
            import wandb
            # Log custom plot to visualize interpolation of different metrics over model pairs nicely
            wandb.log({
                f"interpolation_plot_{key}": wandb.plot.line(interpolation_table, x="interpolation_lambda", y=f"interpolation_{key}", stroke="model_pair", title=f"Interpolation {key} plot", split_table=True)
                for key in PER_STEP_METRICS_KEYS
            })
        
    else:
        print(f"Midpoint interpolation between all pairs of models{(f' at epoch {epoch}'    ) if epoch is not None else ''}...")
        
        models = []
        for i in range(num_models):
            model_state = torch.load(output_dir / f"checkpoint_epoch_{target_epoch}_model_{i+1}.pt", map_location=cfg.device)
            model = create_model_for_interpolation()
            model.load_state_dict(model_state['model_state_dict'])
            model.to(cfg.device)
            models.append(model)
        
        interpolation_results = evaluate_midpoint_models(
            models, data_info['train_loader'], data_info['val_loader'], data_info['test_loader'], cfg.device,
            use_wandb=False
        )
        
        if epoch is not None:
            print(f"Midpoint at epoch {epoch} completed:")
        else:
            print(f"Midpoint completed:")
        print(f"  Train acc: {interpolation_results['train_accuracy']:.4f}%")
        print(f"  Val acc: {interpolation_results['val_accuracy']:.4f}%")
        print(f"  Test acc: {interpolation_results['test_accuracy']:.4f}%")
        print(f"  Train loss: {interpolation_results['train_loss']:.4f}")
        print(f"  Val loss: {interpolation_results['val_loss']:.4f}")
        print(f"  Test loss: {interpolation_results['test_loss']:.4f}")
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    log_data = {
                        'interpolation_type': 'midpoint',
                        'interpolation_train_accuracy': interpolation_results['train_accuracy'],
                        'interpolation_val_accuracy': interpolation_results['val_accuracy'],
                        'interpolation_test_accuracy': interpolation_results['test_accuracy'],
                        'interpolation_train_loss': interpolation_results['train_loss'],
                        'interpolation_val_loss': interpolation_results['val_loss'],
                        'interpolation_test_loss': interpolation_results['test_loss'],
                        'interpolation_num_models': interpolation_results['num_models'],
                    }
                    
                    if epoch is not None:
                        epoch_log_data = {}
                        for key, value in log_data.items():
                            epoch_log_data[f'epoch_{epoch}_{key}'] = value
                        wandb.log(epoch_log_data)
                    else:
                        wandb.log(log_data)
            except ImportError:
                print("Warning: wandb not available")
    
    assert interpolation_results is not None
    if epoch is None:
        save_interpolation_results(interpolation_results, output_dir)
    
    return interpolation_results

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Global seed (determines init_seed and optimization_seed if these are unset)
    if cfg.get('seed', None) is not None:
        set_seed(cfg.seed)
    
    print("Config:")
    print(cfg)

    init_seeds_raw = cfg.get('init_seed', None)
    if init_seeds_raw is None:
        init_seeds = [random.randint(0, 2**32 - 1) for _ in range(cfg.num_models)]
    elif hasattr(init_seeds_raw, '__iter__'):
        init_seeds = [int(x) for x in init_seeds_raw]
    else:
        init_seeds = [int(init_seeds_raw)] * cfg.num_models
    
    optimization_seeds_raw = cfg.get('optimization_seed', None)
    if optimization_seeds_raw is None:
        optimization_seeds = [random.randint(0, 2**32 - 1) for _ in range(cfg.num_models)]
    elif hasattr(optimization_seeds_raw, '__iter__'):
        optimization_seeds = [int(x) for x in optimization_seeds_raw]
    else:
        optimization_seeds = [int(optimization_seeds_raw)] * cfg.num_models

    if len(optimization_seeds) != cfg.num_models:
        raise ValueError(f"Invalid optimization_seed: {len(optimization_seeds)} values specified, expected either one value or num_models={cfg.num_models} values.")
    if len(init_seeds) != cfg.num_models:
        raise ValueError(f"Invalid init_seed: {len(init_seeds)} values specified, expected either one value or num_models={cfg.num_models} values.")

    mask_seed = cfg.mask_seed
    if mask_seed is None:
        mask_seed = random.randint(0, 2**32 - 1)
    
    print(f"Global seed: {cfg.seed}")
    print(f"Init seed(s): {init_seeds}")
    print(f"Opt seed(s): {optimization_seeds}")
    print(f"Mask seed: {mask_seed}")
    
    print(f"Starting: {cfg.experiment_name}")
    
    num_models = cfg.get('num_models', 1)
    cfg.num_models = num_models
    
    if num_models == 1 and cfg.interpolation.enabled:
            print("Warning: LMC needs 2+ models. Skipping LMC.")
    
    train_multi(cfg, init_seeds, optimization_seeds, mask_seed=mask_seed)
    print(f"Completed.")


if __name__ == "__main__":
    import os
    if os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes'):
        torch.autograd.set_detect_anomaly(True)
        import traceback
        import sys
        
        def excepthook(type, value, tb):
            print(f"\nERROR: {value}")
            print("\nFULL STACK TRACE:")
            traceback.print_exception(type, value, tb)
            sys.exit(1)
        
        sys.excepthook = excepthook
    
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFULL STACK TRACE:")
        traceback.print_exc()
        sys.exit(1)
