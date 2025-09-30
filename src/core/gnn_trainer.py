"""
GNN-specific trainer for single-graph datasets like arxiv.
"""

import torch
import torch.nn.functional as F
from .trainer import Trainer
from ogb.nodeproppred import Evaluator


class GNNTrainer(Trainer):
    """Trainer for GNN models on single-graph datasets."""
    
    def __init__(self, model, optimizer, scheduler, device, data, split_idx, 
                 evaluator=None, logger=None, save_every=None, output_dir=None):
        super().__init__(model, None, None, None, optimizer, scheduler, device)
        
        self.data = data
        self.split_idx = split_idx
        self.evaluator = evaluator or Evaluator(name='ogbn-arxiv')
        self.logger = logger
        self.save_every = save_every
        self.output_dir = output_dir
        
        # Move data to device
        self.data = self.data.to(device)
        for key in self.split_idx:
            self.split_idx[key] = self.split_idx[key].to(device)
    
    def train_epoch(self):
        """Train for one epoch on the entire graph."""
        self.model.train()
        
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.adj_t)[self.split_idx['train']]
        loss = F.nll_loss(out, self.data.y.squeeze(1)[self.split_idx['train']])
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {'loss': loss.item()}
    
    def validate(self):
        """Validate on the entire graph."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.adj_t)
            
            # Calculate losses for each split
            train_loss = F.nll_loss(out[self.split_idx['train']], 
                                  self.data.y.squeeze(1)[self.split_idx['train']]).item()
            val_loss = F.nll_loss(out[self.split_idx['valid']], 
                                self.data.y.squeeze(1)[self.split_idx['valid']]).item()
            test_loss = F.nll_loss(out[self.split_idx['test']], 
                                 self.data.y.squeeze(1)[self.split_idx['test']]).item()
            
            # Calculate accuracies
            y_pred = out.argmax(dim=-1, keepdim=True)
            
            train_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['train']],
                'y_pred': y_pred[self.split_idx['train']],
            })['acc']
            
            val_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['valid']],
                'y_pred': y_pred[self.split_idx['valid']],
            })['acc']
            
            test_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['test']],
                'y_pred': y_pred[self.split_idx['test']],
            })['acc']
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        }
    
    def train(self, num_epochs, val_every=1, save_every=None, save_path=None, early_stopping=None, 
              save_grad_every=None, save_params_every=None, model_idx=None):
        """Train the model for specified number of epochs."""
        if save_every is None:
            save_every = self.save_every
        
        best_val_acc = 0.0
        train_losses = []
        val_accs = []
        
        # Save initialization checkpoint (epoch 0)
        if model_idx is not None:
            self._save_checkpoint(0, 0.0, model_idx)  # epoch 0, 0% accuracy
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            val_accs.append(val_metrics['val_acc'])
            
            # Log results
            if self.logger is not None:
                self.logger.log_epoch(epoch + 1, train_metrics, val_metrics)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}, "
                  f"Test Acc: {val_metrics['test_acc']:.4f}")
            
            # Save checkpoint
            if save_every is not None and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1, val_metrics['val_acc'], model_idx)
            
            # Track best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
        
        return {
            'train_losses': train_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'final_metrics': val_metrics
        }
    
    def _save_checkpoint(self, epoch, val_acc, model_idx=None):
        """Save model checkpoint."""
        if self.output_dir is not None:
            import torch
            if model_idx is not None:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_model_{model_idx + 1}.pt"
            else:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'val_accuracy': val_acc,
                'trainable': [k for k, v in self.model.named_parameters() if v.requires_grad]  # Match standard trainer format
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
