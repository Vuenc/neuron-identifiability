import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Callable
import wandb


class Trainer:
    
    def __init__(self, 
                 model: nn.Module,
                 data: Dict[str, Any],
                 optimizer: optim.Optimizer,
                 scheduler: Optional[Any] = None,
                 device: str = 'cuda',
                 loss_fn: Optional[Callable] = None,
                 metrics: Optional[Dict[str, Callable]] = None,
                 logging: Optional[Dict[str, Any]] = None,
                 print_summary: bool = True,
                 model_prefix: str = '',
                 shared_wandb: bool = False):
        self.model = model
        self.train_loader = data.get('train_loader')
        self.val_loader = data.get('val_loader')
        self.test_loader = data.get('test_loader')
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics = metrics or {}
        self.logging = logging or {}
        self.print_summary = print_summary
        self.model_prefix = model_prefix
        self.shared_wandb = shared_wandb
        
        self.model.to(self.device)
        
        if self.print_summary:
            self._print_model_summary()
            self._print_mask_checksum()
            self._print_param_checksum()
        else:
            self._print_mask_checksum()
            self._print_param_checksum()
        
        if self.logging.get('use_wandb', False) and not self.shared_wandb:
            wandb_config = {
                'project': self.logging.get('project', 'asymmetric-networks'),
                'name': self.logging.get('name', None),
                'config': self.logging.get('config', {}),
            }
            
            if self.logging.get('entity'):
                wandb_config['entity'] = self.logging['entity']
            if self.logging.get('group'):
                wandb_config['group'] = self.logging['group']
            if self.logging.get('job_type'):
                wandb_config['job_type'] = self.logging['job_type']
            if self.logging.get('tags'):
                wandb_config['tags'] = self.logging['tags']
            if self.logging.get('notes'):
                wandb_config['notes'] = self.logging['notes']
            if self.logging.get('resume') is not None:
                wandb_config['resume'] = self.logging['resume']
            if self.logging.get('reinit'):
                wandb_config['reinit'] = self.logging['reinit']
            if self.logging.get('mode'):
                wandb_config['mode'] = self.logging['mode']
            
            wandb.init(**wandb_config)
    
    def _print_model_summary(self):
        """Print model summary and parameter counts."""
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

        if hasattr(self.model, 'count_unused_params'):
            masked_params = self.model.count_unused_params()
            trainable_params = total_params - masked_params
            sparsity = masked_params / total_params * 100
            print(f"Masked parameters: {int(masked_params):,}")
            print(f"Trainable parameters: {int(trainable_params):,}")
            print(f"Sparsity: {sparsity:.1f}%")
        else:
            print(f"Trainable parameters: {total_params:,}")
            print(f"Sparsity: 0.0%")

        print("\nModel structure:")
        print("-" * 40)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    print(f"{name:30} | {param_count:>8,} params")
        
        print("="*60)
        print()
    
    def _compute_mask_checksum(self) -> str:
        import hashlib
        mask_data = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'mask') and module.mask is not None:
                mask_data.append(f"{name}_mask:{module.mask.cpu().numpy().tobytes()}")
                if hasattr(module, 'normal_mask'):
                    mask_data.append(f"{name}_normal_mask:{module.normal_mask.cpu().numpy().tobytes()}")
            
            if hasattr(module, 'C') and module.C is not None:
                mask_data.append(f"{name}_C:{module.C.cpu().numpy().tobytes()}")
        
        combined = "|".join(sorted(mask_data))
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _compute_param_checksum(self) -> str:
        import hashlib
        
        param_data = []
        for name, param in self.model.named_parameters():
            param_data.append(f"{name}:{param.cpu().detach().numpy().tobytes()}")
        
        combined = "|".join(sorted(param_data))
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def _print_mask_checksum(self):
        checksum = self._compute_mask_checksum()
        print(f"Mask checksum: {checksum}")
    
    def _print_param_checksum(self):
        checksum = self._compute_param_checksum()
        print(f"Parameter checksum: {checksum}")
    
    def train_epoch(self) -> Dict[str, float]:
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if self.logging.get('log_batch_metrics', False) and batch_idx % 100 == 0:
                if self.logging.get('use_wandb', False):
                    wandb.log({
                        f'{self.model_prefix}_batch_loss': loss.item(),
                        f'{self.model_prefix}_batch': batch_idx
                    })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate(self, data_loader: DataLoader, prefix: str = '') -> Dict[str, float]:
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        metrics_results = {}
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                for metric_name, metric_fn in self.metrics.items():
                    if metric_name not in metrics_results:
                        metrics_results[metric_name] = []
                    metrics_results[metric_name].append(metric_fn(output, target).item())
        
        for metric_name in metrics_results:
            metrics_results[metric_name] = np.mean(metrics_results[metric_name])
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        
        results = {
            f'{prefix}_loss': avg_loss,
            f'{prefix}_accuracy': accuracy,
            **{f'{prefix}_{k}': v for k, v in metrics_results.items()}
        }
        
        return results
    
    def train(self, 
              num_epochs: int,
              val_every: int = 1,
              save_every: Optional[int] = None,
              save_path: Optional[str] = None,
              early_stopping: Optional[Dict[str, Any]] = None,
              save_grad_every: Optional[int] = None,
              save_params_every: Optional[int] = None,
              model_idx: Optional[int] = None) -> Dict[str, Any]:

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'test_loss': [],
            'test_accuracy': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        global_step = 0
        
        if save_path is not None:
            torch.save({
                'epoch': 0,
                'step': 0,
                'model_state_dict': self.model.state_dict(),
                'trainable': [k for k, v in self.model.named_parameters() \
                              if v.requires_grad],
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_accuracy': 0.0
            }, f"{save_path}/checkpoint_epoch_0_{self.model_prefix if model_idx is None else f'model_{model_idx + 1}'}.pt")
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['train_loss'])
            
            if save_grad_every is not None and epoch % save_grad_every == 0:
                self._save_gradients(save_path, epoch, global_step)
            
            if save_params_every is not None and epoch % save_params_every == 0:
                self._save_parameters(save_path, epoch, global_step)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if epoch % val_every == 0 or epoch == num_epochs - 1:
                val_metrics = self.evaluate(self.val_loader, 'val')
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                if self.logging.get('use_wandb', False):
                    wandb.log({
                        f'{self.model_prefix}_epoch': epoch,
                        f'{self.model_prefix}_train_loss': train_metrics['train_loss'],
                        f'{self.model_prefix}_val_loss': val_metrics['val_loss'],
                        f'{self.model_prefix}_val_accuracy': val_metrics['val_accuracy']
                    })
                
                print(f'Epoch {epoch+1}/{num_epochs}: '
                      f'Train Loss: {train_metrics["train_loss"]:.4f}, '
                      f'Val Loss: {val_metrics["val_loss"]:.4f}, '
                      f'Val Acc: {val_metrics["val_accuracy"]:.4f}')
                
                if early_stopping is not None:
                    if val_metrics['val_accuracy'] > best_val_acc:
                        best_val_acc = val_metrics['val_accuracy']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping.get('patience', 10):
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if save_every is not None and (epoch + 1) % save_every == 0 and \
                    save_path is not None:
                torch.save({
                    'epoch': epoch + 1,
                    'step': global_step,
                    'model_state_dict': self.model.state_dict(),
                    'trainable': [k for k, v in self.model.named_parameters() \
                                  if v.requires_grad],
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics.get('val_accuracy', 0.0)
                }, f"{save_path}/checkpoint_epoch_{epoch+1}_{self.model_prefix if model_idx is None else f'model_{model_idx + 1}'}.pt")
            
            global_step += len(self.train_loader) if self.train_loader is not None else 1
        
        test_metrics = self.evaluate(self.test_loader, 'test')
        history['test_loss'].append(test_metrics['test_loss'])
        history['test_accuracy'].append(test_metrics['test_accuracy'])
        
        if self.logging.get('use_wandb', False):
            wandb.log({
                f'{self.model_prefix}_final_test_loss': test_metrics['test_loss'],
                f'{self.model_prefix}_final_test_accuracy': test_metrics['test_accuracy']
            })
            if not self.shared_wandb:
                wandb.finish()
        
        print(f'Final Test Results: Loss: {test_metrics["test_loss"]:.4f}, '
              f'Accuracy: {test_metrics["test_accuracy"]:.4f}')
        
        return {
            'history': history,
            'final_test_metrics': test_metrics,
            'best_val_accuracy': best_val_acc
        }
    
    def _save_gradients(self, save_path: str, epoch: int, step: int):
        if save_path is None:
            return
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu()
        
        save_file = f"{save_path}/checkpoint_epoch_{epoch}_step_{step}_gradients.pt"
        torch.save({
            'epoch': epoch,
            'step': step,
            'gradients': gradients
        }, save_file)
        print(f"Saved gradients to {save_file}")
    
    def _save_parameters(self, save_path: str, epoch: int, step: int):
        if save_path is None:
            return
        
        parameters = {}
        for name, param in self.model.named_parameters():
            parameters[name] = param.detach().cpu()
        
        save_file = f"{save_path}/checkpoint_epoch_{epoch}_step_{step}_parameters.pt"
        torch.save({
            'epoch': epoch,
            'step': step,
            'parameters': parameters
        }, save_file)
        print(f"Saved parameters to {save_file}")
    
    def interpolate_test(self, 
                        model2: nn.Module, 
                        test_fn: Optional[Callable] = None,
                        steps: int = 10) -> list:

        if test_fn is None:
            test_fn = lambda model: self.evaluate(self.test_loader, '')['accuracy']
        
        original_state = self.model.state_dict()
        model2_state = model2.state_dict()
        
        results = []
        
        for i in range(steps + 1):
            alpha = i / steps
            
            interpolated_state = {}
            for key in original_state:
                interpolated_state[key] = (1 - alpha) * original_state[key] + alpha * model2_state[key]
            
            self.model.load_state_dict(interpolated_state)
            
            result = test_fn(self.model)
            results.append(result)
            
            print(f'Interpolation step {i}/{steps}: {result:.4f}')
        
        self.model.load_state_dict(original_state)
        
        return results
