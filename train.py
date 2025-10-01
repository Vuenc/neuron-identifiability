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
    save_interpolation_results
)
from src.utils.gnn_interpolation import interpolate_gnn_models
from src.utils.cosine_similarity import (
    compute_cossim_per_layer,
    compute_cossim_aggregate,
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


def create_model(cfg: DictConfig, fixed_masks=None, device=None):
    if cfg.model.name == 'mlp_mnist':
        return create_mlp(
            symmetry=cfg.model.symmetry,
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.output_dim,
            num_layers=cfg.model.num_layers,
            mask_params=cfg.model.mask_params if cfg.model.symmetry == 1 else None,
            norm=cfg.model.norm,
            elementwise_affine=cfg.model.get('elementwise_affine', True),
            activation=cfg.model.get('activation', None),
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
    elif cfg.model.name == 'gnn_arxiv':
        # TODO: why don't we have fixed_masks here?
        return create_gnn(
            model_type=cfg.model.model_type,
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def setup_data(cfg: DictConfig):
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
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }
    elif cfg.dataset.name == 'arxiv':
        dataset = create_dataset(cfg.dataset.name, cfg.dataset.data_dir)
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
        
        wandb.init(**wandb_config)
        print("Initialized wandb")
        return wandb
    except ImportError:
        print("Warning: wandb not available")
        return None


def generate_masks(cfg: DictConfig, output_dir: Path):
    print("Generating fixed masks...")
    set_mask_seed(cfg.get('mask_seed', cfg.seed))
    
    dummy_model = create_model(cfg, device=cfg.device)
    save_masks(dummy_model, output_dir)
    print(f"Saved masks to {output_dir / 'fixed_masks'}")


def train_multi(cfg: DictConfig, output_dir: Path, init_seeds: list, mask_seeds: list):
    num_models = cfg.get('num_models', 1)
    use_fixed_masks = cfg.get('use_fixed_masks', False)
    
    print(f"Multi-model: {num_models} models")
    print(f"Fixed masks: {use_fixed_masks}")
    
    wandb = setup_wandb(cfg)
    
    if use_fixed_masks:
        generate_masks(cfg, output_dir)
    
    data_info = setup_data(cfg)
    
    cossim_enabled = (cfg.training.cosine_similarity.enabled and 
                     num_models == 2 and 
                     cfg.training.cosine_similarity.save_every is not None)
    
    if cossim_enabled:
        print("Cossim enabled")
    
    model_results = []
    
    for model_idx in range(num_models):
        print(f"\n=== Model {model_idx + 1}/{num_models} ===")
        
        fixed_masks = None
        if use_fixed_masks:
            try:
                fixed_masks = load_masks(output_dir)
                print(f"Loaded masks for model {model_idx + 1}")
            except FileNotFoundError:
                print("Warning: Masks not found")
        
        model_init_seed = init_seeds[model_idx % len(init_seeds)]
        model_mask_seed = mask_seeds[model_idx % len(mask_seeds)]
        
        set_seed(cfg.seed)
        set_init_seed(model_init_seed)
        
        print(f"Init seed: {model_init_seed}, mask seed: {model_mask_seed}")
        
        model = create_model(cfg, fixed_masks=fixed_masks, device=cfg.device)
        
        optimizer = create_optimizer(cfg.optimizer.name, model, **cfg.optimizer)
        scheduler = create_scheduler(cfg.scheduler.name, optimizer, **cfg.scheduler) if cfg.scheduler.name != 'none' else None
        
        trainer_class = GNNTrainer if cfg.model.name == 'gnn_arxiv' else Trainer
        trainer = trainer_class(
            model=model,
            data=data_info,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
            model_prefix=f'model_{model_idx + 1}',
            shared_wandb=True,
            logging=cfg.logging,
            print_summary=(model_idx == 0)
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
    
    if cossim_enabled and num_models == 2:
        cossim_analysis(cfg, output_dir)
    
    if cfg.interpolation.enabled and num_models >= 2:
        if cfg.interpolation.get('save_every') is not None:
            interpolation_interval_analysis(cfg, output_dir, data_info)
        else:
            interpolation_analysis(cfg, output_dir, data_info)
    
    if wandb:
        try:
            wandb.finish()
            print("Finished wandb")
        except ImportError:
            print("Warning: wandb not available")
    
    print(f"\nMulti-model completed. Results saved to {output_dir}")
    return model_results


def cossim_analysis(cfg: DictConfig, output_dir: Path):
    print("\nComputing cossim...")
    
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
                print(f"Warning: Checkpoint epoch {epoch} not found")
                break
    
    if len(model1_checkpoints) > 0:
        print(f"Found {len(model1_checkpoints)} checkpoints")
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

            per_layer_similarities = compute_cossim_per_layer(updates1, updates2, checkpoint1['trainable'])
            aggregate_similarity = compute_cossim_aggregate(updates1, updates2, checkpoint1['trainable'])
            
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

        torch.save({
            'all_epoch_results': all_epoch_results,
            'num_epochs': len(all_epoch_results)
        }, output_dir / "cossim_all_epochs.pt")
        
        print(f"\nSaved cossim results to {output_dir}/cossim_all_epochs.pt")
    else:
        print("Warning: No checkpoints found")


def interpolation_analysis(cfg: DictConfig, output_dir: Path, data_info: dict, epoch: int = None):
    """Perform LMC interpolation analysis for a specific epoch or final models."""
    if epoch is not None:
        print(f"\nPerforming LMC interpolation at epoch {epoch}...")
    else:
        print("\nPerforming LMC interpolation...")
    
    num_models = cfg.get('num_models', 1)
    interpolation_type = cfg.interpolation.type
    
    if interpolation_type == 'grid' and num_models != 2:
        print(f"Warning: Grid needs 2 models, got {num_models}. Using midpoint.")
        interpolation_type = 'midpoint'
    elif interpolation_type == 'midpoint' and num_models < 2:
        print(f"Warning: Midpoint needs 2+ models, got {num_models}. Skipping LMC.")
        return None
    
    def create_model_for_interpolation():
        return create_model(cfg, device=cfg.device)
    
    target_epoch = epoch if epoch is not None else cfg.training.num_epochs
    
    if interpolation_type == 'grid':
        if epoch is not None:
            print(f"Grid interpolation between 2 models at epoch {epoch}...")
        else:
            print("Grid interpolation between 2 models...")

        model1_state = torch.load(output_dir / f"checkpoint_epoch_{target_epoch}_model_1.pt", map_location=cfg.device)
        model2_state = torch.load(output_dir / f"checkpoint_epoch_{target_epoch}_model_2.pt", map_location=cfg.device)

        model1 = create_model_for_interpolation()
        model2 = create_model_for_interpolation()
        model1.load_state_dict(model1_state['model_state_dict'])
        model2.load_state_dict(model2_state['model_state_dict'])
        model1.to(cfg.device)
        model2.to(cfg.device)

        if cfg.model.name == 'gnn_arxiv':
            interpolation_results = interpolate_gnn_models(
                model1, model2, data_info['data'], data_info['split_idx'],
                steps=cfg.interpolation.steps,
                device=cfg.device,
                use_wandb=False
            )
        else:
            interpolation_results = interpolate_models(
                model1, model2, data_info['train_loader'], data_info['val_loader'], data_info['test_loader'],
                steps=cfg.interpolation.steps,
                device=cfg.device,
                use_wandb=False
            )
        
        if epoch is not None:
            print(f"Grid interpolation at epoch {epoch} completed!")
        else:
            print(f"Grid interpolation completed!")
        
        if cfg.logging.get('use_wandb', False):
            try:
                import wandb
                if wandb.run is not None:
                    log_data = {
                        'interpolation_type': 'grid',
                        'interpolation_best_test_accuracy': max(interpolation_results['test_accuracy']),
                        'interpolation_worst_test_accuracy': min(interpolation_results['test_accuracy']),
                        'interpolation_barrier_height': interpolation_results['barrier_height'],
                    }
                    
                    for i, (lam, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss) in enumerate(zip(
                        interpolation_results['lambdas'],
                        interpolation_results['train_accuracy'],
                        interpolation_results['val_accuracy'],
                        interpolation_results['test_accuracy'],
                        interpolation_results['train_loss'],
                        interpolation_results['val_loss'],
                        interpolation_results['test_loss']
                    )):
                        step_data = {
                            'interpolation_lambda': lam,
                            'interpolation_train_accuracy': train_acc,
                            'interpolation_val_accuracy': val_acc,
                            'interpolation_test_accuracy': test_acc,
                            'interpolation_train_loss': train_loss,
                            'interpolation_val_loss': val_loss,
                            'interpolation_test_loss': test_loss,
                        }
                        
                        if epoch is not None:
                            epoch_step_data = {}
                            for key, value in step_data.items():
                                epoch_step_data[f'epoch_{epoch}_{key}'] = value
                            wandb.log(epoch_step_data)
                        else:
                            wandb.log(step_data)
                    
                    if epoch is not None:
                        epoch_log_data = {}
                        for key, value in log_data.items():
                            epoch_log_data[f'epoch_{epoch}_{key}'] = value
                        wandb.log(epoch_log_data)
                    else:
                        wandb.log(log_data)
            except ImportError:
                print("Warning: wandb not available")
        
    else:
        if epoch is not None:
            print(f"Midpoint evaluation for {num_models} models at epoch {epoch}...")
        else:
            print(f"Midpoint evaluation for {num_models} models...")
        
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
    
    # Save results if this is the final analysis
    if epoch is None:
        save_interpolation_results(interpolation_results, output_dir)
    
    return interpolation_results


def interpolation_interval_analysis(cfg: DictConfig, output_dir: Path, data_info: dict):
    """Perform interval-based interpolation analysis similar to cosine similarity."""
    print("\nComputing interval-based interpolation analysis...")
    
    save_every = cfg.interpolation.get('save_every', None)
    
    if save_every is None:
        print("Warning: Interpolation save_every not configured")
        return
    
    print(f"Analyzing every {save_every} epochs")
    
    all_epoch_results = []
    
    for epoch in range(0, cfg.training.num_epochs + 1):
        if epoch % save_every == 0:
            print(f"Running interpolation analysis for epoch {epoch}...")
            results = interpolation_analysis(cfg, output_dir, data_info, epoch=epoch)
            
            if results is not None:
                epoch_results = {
                    'epoch': epoch,
                    'interpolation_results': results
                }
                all_epoch_results.append(epoch_results)
                
                # Log summary metrics
                if 'test_accuracy' in results:
                    if isinstance(results['test_accuracy'], list):
                        # Grid interpolation - show best and worst
                        best_acc = max(results['test_accuracy'])
                        worst_acc = min(results['test_accuracy'])
                        print(f"Epoch {epoch} interpolation completed - Test acc: {best_acc:.4f}% (best), {worst_acc:.4f}% (worst)")
                    else:
                        # Midpoint interpolation - single value
                        print(f"Epoch {epoch} interpolation completed - Test acc: {results['test_accuracy']:.4f}%")
                elif 'barrier_height' in results:
                    print(f"Epoch {epoch} interpolation completed - Barrier height: {results['barrier_height']:.4f}")
                else:
                    print(f"Epoch {epoch} interpolation completed")
            else:
                print(f"Warning: Interpolation analysis failed for epoch {epoch}")
    
    if len(all_epoch_results) > 0:
        torch.save({
            'all_epoch_results': all_epoch_results,
            'num_epochs': len(all_epoch_results),
            'save_every': save_every
        }, output_dir / "interpolation_all_epochs.pt")
        
        print(f"\nSaved interval interpolation results to {output_dir}/interpolation_all_epochs.pt")
        print(f"Completed interpolation analysis for {len(all_epoch_results)} epochs")
    else:
        print("Warning: No interpolation results found")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    init_seeds_raw = cfg.get('init_seed', cfg.seed)
    mask_seeds_raw = cfg.get('mask_seed', cfg.seed)
    
    if init_seeds_raw == 'infer_from_mask' and cfg.use_fixed_masks:
        init_seeds = [int(mask_seeds_raw)+i+1 for i in range(cfg.num_models)]
    elif isinstance(init_seeds_raw, (list, tuple)):
        init_seeds = [int(x) for x in init_seeds_raw]
    else:
        init_seeds = [int(init_seeds_raw)]
    
    if isinstance(mask_seeds_raw, (list, tuple)):
        mask_seeds = [int(x) for x in mask_seeds_raw]
    else:
        mask_seeds = [int(mask_seeds_raw)]
    
    print(f"Global seed: {cfg.seed}")
    print(f"Init seed(s): {init_seeds}")
    print(f"Mask seed(s): {mask_seeds}")
    
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    print(f"Starting: {cfg.experiment_name}")
    print(f"Output: {output_dir}")
    
    num_models = cfg.get('num_models', 1)
    cfg.num_models = num_models
    
    if num_models == 1:
        print("Single model training")
        set_init_seed(init_seeds[0])
        if cfg.interpolation.enabled:
            print("Warning: LMC needs 2+ models. Skipping LMC.")
    else:
        print(f"Multi-model training: {num_models} models")
    
    train_multi(cfg, output_dir, init_seeds, mask_seeds)
    print(f"Completed. Results saved to {output_dir}")


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
