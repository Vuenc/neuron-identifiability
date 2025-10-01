"""
GNN-specific trainer for single-graph datasets like arxiv.
"""

import torch
import torch.nn.functional as F
from .trainer import Trainer
from ogb.nodeproppred import Evaluator


class GNNTrainer(Trainer):
    """Trainer for GNN models on single-graph datasets."""
    
    def __init__(self, model, optimizer, scheduler, device, data, 
                 evaluator=None, model_prefix='', shared_wandb=False, logging=None, print_summary=True):
        super().__init__(model, data, optimizer, scheduler, device,
                        model_prefix=model_prefix, shared_wandb=shared_wandb, 
                        logging=logging, print_summary=print_summary)
        
        self.data = data['data']
        self.split_idx = data['split_idx']
        self.evaluator = evaluator or Evaluator(name='ogbn-arxiv')
        
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
        
        return {'train_loss': loss.item()}
    
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
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        }
    
    def evaluate(self, data_loader, prefix=''):
        """Evaluate on the entire graph."""
        return self.validate()
    
    def train(self, num_epochs, val_every=1, save_every=None, save_path=None, early_stopping=None, 
              save_grad_every=None, save_params_every=None, model_idx=None):
        """Train the model for specified number of epochs."""
        return super().train(num_epochs, val_every, save_every, save_path, early_stopping, 
                           save_grad_every, save_params_every, model_idx)
