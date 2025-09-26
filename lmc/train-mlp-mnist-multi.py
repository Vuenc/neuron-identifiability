# -*- coding: utf-8 -*-
import math
import itertools
import copy
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import random
import numpy as np
from models.models_mlp import SigmaMLP, WMLP, MLP
from LMC_utils import *
import argparse
import multiprocessing
import pickle
import datasets
import os
from datetime import datetime

def _model_prefix(model):
    if isinstance(model, MLP):
        return "mlp"
    elif isinstance(model, WMLP):
        return "wmlp"
    elif isinstance(model, SigmaMLP):
        return "sigmlp"

def _snap(output_path, model, index, step, epoch, batch_idx, loss):
    md = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    gd = {n: (p.grad.detach().clone().cpu() if p.grad is not None else None) for n, p in model.named_parameters()}
    prefix = _model_prefix(model)
    torch.save(dict(step=step, epoch=epoch, batch_idx=batch_idx, loss=float(loss),
                    model_state_dict=md, grads_state_dict=gd),
               f"{output_path}/trained_{prefix}_{index}_step-{step}.pt")

def train(model, optimizer, train_loader, device, index, batch_size=32, n_epochs=10, lr_schedule = None, output_path=None, save_every_epoch=0, save_every_step=0):
    assert output_path is not None
    os.makedirs(output_path, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    train_loss_history = np.zeros([n_epochs, 1])
    valid_accuracy_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])
    step = 0
    for epoch in range(n_epochs):

        # Train code from CS189
        model.train()

        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            train_loss += loss.item()
            loss.backward()
            if save_every_step and step % save_every_step == 0:
                _snap(output_path, model, index, step, epoch, batch_idx, loss.item()/data.size(0))
            optimizer.step()
            step += 1
        if lr_schedule:
            lr_schedule.step()
        train_loss_history[epoch] = train_loss / len(train_loader.dataset)

        # Track loss each epoch
        print('Train Epoch: %d  Average loss: %.4f' %
              (epoch + 1,  train_loss_history[epoch]))

        # if epoch % 5 == 0:

        if save_every_epoch > 0 and epoch % save_every_epoch == 0:
            import pickle
            with open(f"{output_path}/trained_{_model_prefix(model)}_{index}_epoch-{epoch}.pickle", 'wb') as handle:
                pickle.dump(dict(model=model, loss=float(train_loss)/len(train_loader.dataset)), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # model.eval()

        # valid_loss = 0
        # correct = 0
        # valid_loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
        # with torch.no_grad():
        #     for data, target in valid_loader:
        #         data = data.to(device)
        #         target = target.to(device)
        #         output = model(data)
        #         valid_loss += valid_loss_fn(output, target).item()
        #         pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
        #         correct += pred.eq(target.view_as(pred)).sum().item()

        # valid_loss_history[epoch] = valid_loss / len(valid_loader.dataset)

        
        # valid_accuracy_history[epoch] = correct / len(valid_loader.dataset)

        # print('Valid set: Average loss: %.4f, Accuracy: %d/%d (%.4f)\n' %
        #       (valid_loss_history[epoch], correct, len(valid_loader.dataset),
        #       100. * valid_accuracy_history[epoch]))

    return model


# def test(model, batch_size=32):

#     model.eval()
#     loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
 
#     test_loss = 0
#     correct = 0
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.to(device)
#             target = target.to(device)
#             output = model(data)
#             test_loss += loss_fn(output, target).item()  # Sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     test_accuracy = correct / len(test_loader.dataset)
    
#     print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
#           (test_loss, correct, len(test_loader.dataset),
#           100. * test_accuracy))
#     return test_loss, test_accuracy

def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one(index, args, seed):
    seed_training(seed)
    device = torch.device('cuda')

    train_dataset = datasets.dataset_factories[args.dataset](device)
    # train_dataset = random_datasets.GaussianNoiseDataset("data/GaussianNoise/noise-50000-images.npy", "data/GaussianNoise/noise-50000-labels.npy", device)
    # train_dataset = [(image.to(device), label.to(device)) for image, label in random_datasets.MNISTRandomLabelsDataset(138914, device)]
    # train_dataset = list(random_datasets.MNISTRandomLabelsDataset(138914, device))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # TODO: Turn on again! Just for reproducibility w/ gradients
        pin_memory = False
    )

    in_dim, out_dim = 784,10
    linear_mask_params_0 = {'mask_constant' : args.lin_c_0, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_0}
    linear_mask_params_1 = {'mask_constant' : args.lin_c_1, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_1}
    linear_mask_params_2 = {'mask_constant' : args.lin_c_2, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_2}
    linear_mask_params_3 = {'mask_constant' : args.lin_c_3, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_3}

    
    mask_params = {
        0 : linear_mask_params_0, 
        1 : linear_mask_params_1, 
        2: linear_mask_params_2,
        3: linear_mask_params_3
    }

    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    model = None
    if args.symmetry == 0:
        model = MLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm=args.norm, act=args.act).to(device)
    elif args.symmetry == 1:
        model = WMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, mask_params, norm=args.norm, act=args.act).to(device)
    elif args.symmetry == 2:
        model = SigmaMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm=args.norm, act=args.act).to(device)
    assert model is not None

    naive_num_params = sum(p.numel() for p in model.parameters())
    num_params = naive_num_params - model.count_unused_params()
    print('Naive param count:', naive_num_params)
    print('Actual param count:', num_params)

    epochs = args.epochs
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    out_dir = os.path.join(os.path.realpath(args.output_path), f"run_{index}")
    os.makedirs(out_dir, exist_ok=True)
    train(model, optimizer, train_loader, device, index, n_epochs = epochs, lr_schedule = None, batch_size=args.batch_size, output_path=out_dir, save_every_epoch=args.save_every_epoch, save_every_step=args.save_every_step)
    import pickle
    with open(f'{out_dir}/trained_{_model_prefix(model)}_{index}.pickle', 'wb') as handle:
        pickle.dump(dict(model=model), handle, protocol=pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('-w', '--weight_decay', default=.03, type=float,
                        metavar='W', help='weight decay (default: 0e-4')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        metavar='W', help='lr (default: 1e-3')
    
    parser.add_argument('--lin_n_1', default = 64 , type=int,
                        metavar='LN1', help='lin c 1 (default: 64)')
    parser.add_argument('--lin_c_1', default=1, type=int,
                        metavar='L', help='lin c 1 (default: 1)')
    
    parser.add_argument('--lin_n_2', default=64, type=int,
                        metavar='LC', help='lin n 2 (default: 64)')
    parser.add_argument('--lin_c_2', default=1/2, type=int,
                        metavar='LC', help='lin c 2 (default: 1/2)')
    parser.add_argument('--lin_n_3', default=256, type=int,
                        metavar='LC', help='lin n 3 (default: 256)')
    parser.add_argument('--lin_c_3', default=1/4, type=int,
                        metavar='LC', help='lin c 3 (default: 1/4)')
    parser.add_argument('--lin_n_0', default=64, type=int,
                        metavar='LC', help='lin n 0 (default: 64)')
    parser.add_argument('--lin_c_0', default=1, type=int,
                        metavar='LC', help='lin c 0 (default: 1)')

    parser.add_argument('--symmetry',  default=1, type=int,
                        metavar='s', help='Symmetry: 0 (Standard) 1 (W) 2 (Sigma)')
    parser.add_argument('--act', default='gelu', type=str, choices=('gelu', 'identity'))
    parser.add_argument('--norm', default=None, type=str, choices=('batch', 'layer', 'batch_linear', 'layer_linear'))
    parser.add_argument('--output_path',  required=True,
                        metavar='o', help='Path of directory where to write output files')
    parser.add_argument("--num", required=True, type=int, metavar="n", help="Number of models to train")
    parser.add_argument("--dataset", type=str, required=True, choices=list(datasets.dataset_factories.keys()))
    parser.add_argument("--save_every_epoch", type=int, required=False, default=1, help="Save every n-th epoch (default: 1). Pass 0 to disable saving.")
    parser.add_argument("--save_every_step", type=int, default=0)
    args = parser.parse_args()

    # resolve base output dir, then append date/time subfolders
    base_out = os.path.realpath(args.output_path)
    if os.path.exists(base_out) and not os.path.isdir(base_out):
        raise ValueError(f"--output_path '{args.output_path}' exists but is not a directory")
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    base_out = os.path.join(base_out, date_str, time_str)
    os.makedirs(base_out, exist_ok=True)
    args.output_path = base_out

    seeds = [random.randint(0, 2**31) for i in range(args.num)]
    print(seeds)
    with multiprocessing.Pool(17) as pool:
        pool.starmap(train_one, [(i, args, seed) for i, seed in enumerate(seeds)])
