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
from src.core.gnn_trainer import GNNTrainer
from src.data.datasets import create_dataset, create_dataloader, create_train_val_test_split
from src.models.mlp import create_mlp
from src.models.resnet import create_resnet
from src.models.gnn import create_gnn
from src.optimizers.optimizers import create_optimizer
from src.optimizers.schedulers import create_scheduler
from src.utils.mask_utils import save_masks, load_masks
from src.utils.interpolation import (
    interpolate_models, 
    evaluate_midpoint_models,
    evaluate_checkpoint_interpolation,
    save_interpolation_results
)
from src.utils.gnn_interpolation import interpolate_gnn_models
from src.utils.cosine_similarity import (
    compute_cosine_similarity_analysis,
    compute_cosine_similarity_per_layer,
    compute_cosine_similarity_aggregate,
    save_cosine_similarity_results
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_init_seed(init_seed):
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)

def set_mask_seed(mask_seed):
    torch.manual_seed(mask_seed)
    torch.cuda.manual_seed_all(mask_seed)


def train_single_model(cfg: DictConfig, output_dir: Path, model_idx: int = None) -> dict:
    
    if cfg.dataset.name in ['mnist', 'cifar10', 'cifar100']:
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
        
        if cfg.model.name == 'mlp_mnist':
            model = create_mlp(
                symmetry=cfg.model.symmetry,
                input_dim=cfg.model.input_dim,
                hidden_dim=cfg.model.hidden_dim,
                output_dim=cfg.model.output_dim,
                num_layers=cfg.model.num_layers,
                mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                norm=cfg.model.norm,
                elementwise_affine=cfg.model.get('elementwise_affine', True),
                activation=cfg.model.get('activation', None)
            )
        elif cfg.model.name == 'resnet_cifar':
            model = create_resnet(
                symmetry=cfg.model.symmetry,
                depth=cfg.model.depth,
                w=cfg.model.w,
                mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                num_classes=cfg.model.num_classes
            )
        elif cfg.model.name == 'gnn_arxiv':
            # GNN training
            dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
            data = dataset[0]
            data = data.to(cfg.device)
            split_idx = dataset.get_idx_split()
            
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
            ).to(cfg.device)
        else:
            raise ValueError(f"Unknown model: {cfg.model.name}")
        
        optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
        scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
        
        if cfg.model.name == 'gnn_arxiv':
            # GNN trainer for single-graph datasets
            trainer = GNNTrainer(
                model=model,
                data=data,
                split_idx=split_idx,
                optimizer=optimizer,
                scheduler=scheduler,
                device=cfg.device,
                logger=None,
                save_every=cfg.training.save_every,
                output_dir=output_dir
            )
        else:
            # Standard trainer for batch-based datasets
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
        
        results = trainer.train(
            num_epochs=cfg.training.num_epochs,
            val_every=cfg.training.val_every,
            save_every=cfg.training.save_every,
            save_path=str(output_dir) if cfg.training.save_path is None else cfg.training.save_path,
            early_stopping=cfg.training.early_stopping,
            model_idx=model_idx,
            save_grad_every=cfg.training.save_grad_every,
            save_params_every=cfg.training.save_params_every
        )
        
        return results, trainer
    
    elif cfg.dataset.name == 'arxiv':

        try:
            from ogb.nodeproppred import Evaluator
        except ImportError:
            raise ImportError("ogb is required for ArXiv dataset.")
        
        dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
        data = dataset[0]
        data = data.to(cfg.device)
        split_idx = dataset.get_idx_split()
        
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
        ).to(cfg.device)
        
        optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
        scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
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
        
        model_num = (model_idx + 1) if model_idx is not None else 1
        torch.save({
            'epoch': 0,
            'step': 0,
            'model_state_dict': model.state_dict(),
            'val_accuracy': 0.0
        }, output_dir / f"checkpoint_epoch_0_model_{model_num}.pt")
        print(f"Saved initialization checkpoint to {output_dir}/checkpoint_epoch_0_model_{model_num}.pt")
        
        best_val_acc = 0
        step = 0
        for epoch in range(cfg.training.num_epochs):
            loss = train_gnn_epoch()
            train_acc, val_acc, test_acc = test_gnn()
            step += 1
            
            if epoch % cfg.training.log_steps == 0:
                print(f'Epoch {epoch+1:03d}: Loss: {loss:.4f}, '
                      f'Train: {100*train_acc:.2f}%, Val: {100*val_acc:.2f}%, '
                      f'Test: {100*test_acc:.2f}%')
            
            if cfg.training.save_every is not None and (epoch + 1) % cfg.training.save_every == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc
                }, output_dir / f"checkpoint_epoch_{epoch+1}_model_{model_num}.pt")
                print(f"Saved checkpoint to {output_dir}/checkpoint_epoch_{epoch+1}_model_{model_num}.pt")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc
                }, output_dir / "checkpoint_best.pt")
        
        print(f'Best validation accuracy: {100*best_val_acc:.2f}%')
        
        results = {
            'final_test_metrics': {
                'test_accuracy': test_acc,
                'test_loss': 0.0
            }
        }
        
        return results, None
    
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    set_seed(cfg.seed)
    init_seeds_raw = cfg.get('init_seed', cfg.seed)
    mask_seeds_raw = cfg.get('mask_seed', cfg.seed)
    
    if init_seeds_raw == 'infer_from_mask' and cfg.use_fixed_masks:
        init_seeds = [int(mask_seeds_raw)+i+1 for i in range(cfg.num_models)]
    elif isinstance(init_seeds_raw, (list, tuple)) or hasattr(init_seeds_raw, '__iter__'):
        init_seeds = [int(x) for x in init_seeds_raw]
    else:
        init_seeds = [int(init_seeds_raw)]
    
    if isinstance(mask_seeds_raw, (list, tuple)) or hasattr(mask_seeds_raw, '__iter__'):
        mask_seeds = [int(x) for x in mask_seeds_raw]
    else:
        mask_seeds = [int(mask_seeds_raw)]
    
    print(f"Global seed: {cfg.seed}")
    print(f"Initialization seeds: {init_seeds}")
    print(f"Mask seeds: {mask_seeds}")
    
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    print(f"Starting experiment: {cfg.experiment_name}")
    print(f"Output directory: {output_dir}")
    
    num_models = cfg.get('num_models', 1)
    use_fixed_masks = cfg.get('use_fixed_masks', False)
    
    if num_models > 1:
        print(f"Multi-model experiment: {num_models} models")
        print(f"Use fixed masks: {use_fixed_masks}")
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                wandb_config = {
                    'project': cfg.logging.get('project', 'asymmetric-networks'),
                    'name': cfg.logging.get('name', f"{cfg.experiment_name}_multi_model"),
                    'config': dict(cfg),
                }
                
                if cfg.logging.get('entity'):
                    wandb_config['entity'] = cfg.logging['entity']
                if cfg.logging.get('group'):
                    wandb_config['group'] = cfg.logging['group']
                if cfg.logging.get('job_type'):
                    wandb_config['job_type'] = cfg.logging['job_type']
                if cfg.logging.get('tags'):
                    wandb_config['tags'] = cfg.logging['tags']
                if cfg.logging.get('notes'):
                    wandb_config['notes'] = cfg.logging['notes']
                if cfg.logging.get('resume') is not None:
                    wandb_config['resume'] = cfg.logging['resume']
                if cfg.logging.get('reinit'):
                    wandb_config['reinit'] = cfg.logging['reinit']
                if cfg.logging.get('mode'):
                    wandb_config['mode'] = cfg.logging['mode']
                
                wandb.init(**wandb_config)
                print("Initialized wandb for multi-model experiment")
            except ImportError:
                print("Warning: wandb not available for logging")
        
        mask_checksums = []
        param_checksums = []
        
        if use_fixed_masks:
            print("Generating fixed masks...")
            set_mask_seed(mask_seeds[0])
            
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
            elif cfg.model.name == 'gnn_arxiv':
                
                # GNN mask generation
                dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
                data = dataset[0]
                data = data.to(cfg.device)
                split_idx = dataset.get_idx_split()
                
                C_lst = None
                if cfg.model.model_type in ['asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn']:
                    import math
                    C_lst = [0.01 * torch.randn(cfg.model.hidden_channels, cfg.model.hidden_channels) / 
                            math.sqrt(cfg.model.hidden_channels) for _ in range(cfg.model.num_layers)]
                
                dummy_model = create_gnn(
                    model_type=cfg.model.model_type,
                    in_channels=data.num_features,
                    hidden_channels=cfg.model.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=cfg.model.num_layers,
                    dropout=cfg.model.dropout,
                    C_lst=C_lst
                ).to(cfg.device)
            else:
                raise ValueError(f"Unknown model: {cfg.model.name}")
            
            save_masks(dummy_model, output_dir)
            print(f"Generated and saved masks to {output_dir / 'fixed_masks'}")
        
        model_results = []
        
        cosine_sim_enabled = (cfg.training.cosine_similarity.enabled and 
                             num_models == 2 and 
                             cfg.training.cosine_similarity.save_every is not None)
        
        if cosine_sim_enabled:
            print("Cosine similarity analysis enabled for 2-model training")
        
        for model_idx in range(num_models):
            print(f"\n=== Training Model {model_idx + 1}/{num_models} ===")
            
            fixed_masks = None
            if use_fixed_masks:
                try:
                    fixed_masks = load_masks(output_dir)
                    print(f"Loaded fixed masks for model {model_idx + 1}")
                except FileNotFoundError:
                    print("Warning: Fixed masks not found, using random masks")
            
            model_init_seed = init_seeds[model_idx % len(init_seeds)]
            model_mask_seed = mask_seeds[model_idx % len(mask_seeds)]
            
            set_seed(cfg.seed)
            set_init_seed(model_init_seed)
            
            print(f"Model {model_idx + 1} initialization seed: {model_init_seed}")
            print(f"Model {model_idx + 1} mask seed: {model_mask_seed}")
            
            if cfg.model.name == 'mlp_mnist':
                model = create_mlp(
                    symmetry=cfg.model.symmetry,
                    input_dim=cfg.model.input_dim,
                    hidden_dim=cfg.model.hidden_dim,
                    output_dim=cfg.model.output_dim,
                    num_layers=cfg.model.num_layers,
                    mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
                    norm=cfg.model.norm,
                    fixed_masks=fixed_masks,
                    elementwise_affine=cfg.model.get('elementwise_affine', True),
                    activation=cfg.model.get('activation', None)
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
            elif cfg.model.name == 'gnn_arxiv':

                dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
                data = dataset[0]
                data = data.to(cfg.device)
                split_idx = dataset.get_idx_split()
                
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
                ).to(cfg.device)
            else:
                raise ValueError(f"Unknown model: {cfg.model.name}")
            
            if model_idx == 0:
                if cfg.model.name == 'gnn_arxiv':
                    train_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
                    test_dataset = None
                else:
                    train_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir, train=True)
                    test_dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir, train=False)
                
                if cfg.model.name == 'gnn_arxiv':
                    val_dataset = None
                else:
                    train_dataset, val_dataset = create_train_val_test_split(
                        train_dataset, val_split=cfg.dataset.val_split, test_split=0.0, seed=cfg.seed
                    )[0:2]
                
                if cfg.model.name == 'gnn_arxiv':
                    train_loader = None
                    val_loader = None
                else:
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
                if cfg.model.name == 'gnn_arxiv':
                    test_loader = None
                else:
                    test_loader = create_dataloader(
                        test_dataset, 
                        batch_size=cfg.dataset.batch_size, 
                        shuffle=False,
                        num_workers=cfg.dataset.num_workers,
                        pin_memory=cfg.dataset.pin_memory
                    )
            
            optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
            scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
            
            if cfg.model.name == 'gnn_arxiv':
                trainer = GNNTrainer(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=cfg.device,
                    data=data,
                    split_idx=split_idx,
                    logger=None,
                    save_every=cfg.training.save_every,
                    output_dir=output_dir
                )
            else:
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=cfg.device,
                    logging=cfg.logging,
                    print_summary=(model_idx == 0),
                    model_prefix=f'model_{model_idx + 1}',
                    shared_wandb=True
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
                model_idx=model_idx
            )         
            
            model_results.append(results)
        
        if cosine_sim_enabled and num_models == 2:
            print("\nComputing cosine similarity analysis from checkpoints...")
            
            model1_checkpoints = []
            model2_checkpoints = []
            
            for epoch in range(0, cfg.training.num_epochs + 1):
                if cfg.training.save_every is None or epoch % cfg.training.save_every == 0:
                    try:
                        checkpoint1 = torch.load(output_dir / f"checkpoint_epoch_{epoch}_model_1.pt", map_location='cpu')
                        checkpoint2 = torch.load(output_dir / f"checkpoint_epoch_{epoch}_model_2.pt", map_location='cpu')
                        model1_checkpoints.append(checkpoint1)
                        model2_checkpoints.append(checkpoint2)
                    except FileNotFoundError:
                        print(f"Warning: Checkpoint for epoch {epoch} not found")
                        break
            
            if len(model1_checkpoints) > 0:
                
                print(f"Found {len(model1_checkpoints)} epoch checkpoints")
                all_epoch_results = []
                
                for i, (checkpoint1, checkpoint2) in enumerate(zip(model1_checkpoints, model2_checkpoints)):
                    epoch = i + 1
                    
                    prev_checkpoint1 = model1_checkpoints[i-1]
                    prev_checkpoint2 = model2_checkpoints[i-1]
                    prev_params1 = prev_checkpoint1['model_state_dict']
                    prev_params2 = prev_checkpoint2['model_state_dict']
                    
                    current_params1 = checkpoint1['model_state_dict']
                    current_params2 = checkpoint2['model_state_dict']

                    updates1 = {}
                    updates2 = {}
                    for name in current_params1:
                        if name in prev_params1:
                            updates1[name] = current_params1[name] - prev_params1[name]
                    for name in current_params2:
                        if name in prev_params2:
                            updates2[name] = current_params2[name] - prev_params2[name]

                    per_layer_similarities = compute_cosine_similarity_per_layer(updates1, updates2, checkpoint1['trainable'])
                    aggregate_similarity = compute_cosine_similarity_aggregate(updates1, updates2, checkpoint1['trainable'])
                    
                    epoch_results = {
                        'epoch': epoch,
                        'per_layer_similarities': per_layer_similarities,
                        'aggregate_similarity': aggregate_similarity,
                    }
                    
                    all_epoch_results.append(epoch_results)

                    print(f"Epoch {epoch} aggregate cossim: {aggregate_similarity:.4f}")
                    print(f"Epoch {epoch} per layer similarities: {per_layer_similarities}")
                    
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
                            print("Warning: wandb not available for logging")

                torch.save({
                    'all_epoch_results': all_epoch_results,
                    'num_epochs': len(all_epoch_results)
                }, output_dir / "cossim_all_epochs.pt")
                
                print(f"\nSaved cosine similarity results for all epochs to {output_dir}/cossim_all_epochs.pt")

            else:
                print("Warning: No checkpoints found for cosine similarity analysis")
        
        if cfg.interpolation.enabled and num_models >= 2:
            print("\nPerforming LMC interpolation test...")
            
            if cfg.logging.get('use_wandb', False):
                try:
                    import wandb
                except ImportError:
                    print("Warning: wandb not available for logging")
            
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
                        fixed_masks=fixed_masks,
                        elementwise_affine=cfg.model.elementwise_affine,
                        activation=cfg.model.activation
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
                elif cfg.model.name == 'gnn_arxiv':

                    dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
                    data = dataset[0]
                    data = data.to(cfg.device)
                    split_idx = dataset.get_idx_split()
                    
                    C_lst = None
                    if cfg.model.model_type in ['asym_gelu_gnn', 'asym_swiglu_gnn', 'asym_w_gnn']:
                        import math
                        C_lst = [0.01 * torch.randn(cfg.model.hidden_channels, cfg.model.hidden_channels) / 
                                math.sqrt(cfg.model.hidden_channels) for _ in range(cfg.model.num_layers)]
                    
                    return create_gnn(
                        model_type=cfg.model.model_type,
                        in_channels=data.num_features,
                        hidden_channels=cfg.model.hidden_channels,
                        out_channels=dataset.num_classes,
                        num_layers=cfg.model.num_layers,
                        dropout=cfg.model.dropout,
                        C_lst=C_lst
                    ).to(cfg.device)
                else:
                    raise ValueError(f"Unknown model: {cfg.model.name}")
            
            interpolation_type = cfg.interpolation.type
            
            if interpolation_type == 'auto':
                if num_models == 2:
                    interpolation_type = 'grid'
                else:
                    interpolation_type = 'midpoint'
            
            if interpolation_type == 'grid' and num_models != 2:
                print(f"Warning: Grid interpolation requires exactly 2 models, but {num_models} provided. Using midpoint instead.")
                interpolation_type = 'midpoint'
            elif interpolation_type == 'midpoint' and num_models < 2:
                print(f"Warning: Midpoint evaluation requires at least 2 models, but {num_models} provided. Skipping LMC.")
                return
            
            if interpolation_type == 'grid':

                print("Performing grid interpolation between 2 models...")

                model1_state = torch.load(output_dir / f"checkpoint_epoch_{cfg.training.num_epochs}_model_1.pt", map_location=cfg.device)
                model2_state = torch.load(output_dir / f"checkpoint_epoch_{cfg.training.num_epochs}_model_2.pt", map_location=cfg.device)

                model1 = create_model_for_interpolation()
                model2 = create_model_for_interpolation()
                model1.load_state_dict(model1_state['model_state_dict'])
                model2.load_state_dict(model2_state['model_state_dict'])
                model1.to(cfg.device)
                model2.to(cfg.device)

                if cfg.model.name == 'gnn_arxiv':
                    interpolation_results = interpolate_gnn_models(
                        model1, model2, data, split_idx,
                        steps=cfg.interpolation.steps,
                        device=cfg.device,
                        use_wandb=cfg.logging.get('use_wandb', False)
                    )
                else:
                    interpolation_results = interpolate_models(
                        model1, model2, train_loader, val_loader, test_loader,
                        steps=cfg.interpolation.steps,
                        device=cfg.device,
                        use_wandb=cfg.logging.get('use_wandb', False)
                    )
                
                print(f"Grid interpolation completed successfully!")
                
                if cfg.logging.get('use_wandb', False):
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                'interpolation_completed': True,
                                'interpolation_type': 'grid',
                                'interpolation_best_test_accuracy': max(interpolation_results['test_accuracy']),
                                'interpolation_worst_test_accuracy': min(interpolation_results['test_accuracy']),
                                'interpolation_barrier_height': interpolation_results['barrier_height'],
                            })
                    except ImportError:
                        print("Warning: wandb not available for logging")
                
            else:
                print(f"Performing midpoint evaluation for {num_models} models...")
                
                models = []
                for i in range(num_models):
                    model_state = torch.load(output_dir / f"checkpoint_epoch_{cfg.training.num_epochs}_model_{i+1}.pt", map_location=cfg.device)
                    model = create_model_for_interpolation()
                    model.load_state_dict(model_state['model_state_dict'])
                    model.to(cfg.device)
                    models.append(model)
                
                interpolation_results = evaluate_midpoint_models(
                    models, train_loader, val_loader, test_loader, cfg.device,
                    use_wandb=cfg.logging.get('use_wandb', False)
                )
                
                print(f"Midpoint evaluation completed:")
                print(f"  Train accuracy: {interpolation_results['train_accuracy']:.4f}%")
                print(f"  Val accuracy: {interpolation_results['val_accuracy']:.4f}%")
                print(f"  Test accuracy: {interpolation_results['test_accuracy']:.4f}%")
                print(f"  Train loss: {interpolation_results['train_loss']:.4f}")
                print(f"  Val loss: {interpolation_results['val_loss']:.4f}")
                print(f"  Test loss: {interpolation_results['test_loss']:.4f}")
                
                if cfg.logging.get('use_wandb', False):
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                'interpolation_completed': True,
                                'interpolation_type': 'midpoint',
                                'interpolation_test_accuracy': interpolation_results['test_accuracy'],
                                'interpolation_test_loss': interpolation_results['test_loss'],
                                'interpolation_num_models': interpolation_results['num_models'],
                            })
                    except ImportError:
                        print("Warning: wandb not available for logging")
            
            save_interpolation_results(interpolation_results, output_dir)
        
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
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                wandb.finish()
                print("Finished wandb run for multi-model experiment")
            except ImportError:
                print("Warning: wandb not available for logging")
    
    else:
        print("Single model training")
        set_init_seed(init_seeds[0])
        results, trainer = train_single_model(cfg, output_dir)
        
        if cfg.interpolation.enabled:
            print("Warning: LMC interpolation requires at least 2 models. Skipping LMC for single model training.")
            print("Use num_models=2 or more to enable LMC interpolation.")
        
        print(f"Experiment completed. Results saved to {output_dir}")


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