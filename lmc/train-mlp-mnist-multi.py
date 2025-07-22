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

def train(model, optimizer, train_loader, device, batch_size=32, n_epochs=10, lr_schedule = None):
    loss_fn = nn.CrossEntropyLoss()
    train_loss_history = np.zeros([n_epochs, 1])
    valid_accuracy_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])
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
            optimizer.step()
        if lr_schedule:
            lr_schedule.step()
        train_loss_history[epoch] = train_loss / len(train_loader.dataset)

        # Track loss each epoch
        print('Train Epoch: %d  Average loss: %.4f' %
              (epoch + 1,  train_loss_history[epoch]))

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


def test(model, batch_size=32):

    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
 
    test_loss = 0
    correct = 0
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    
    print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
          (test_loss, correct, len(test_loader.dataset),
          100. * test_accuracy))
    return test_loss, test_accuracy

def train_one(index, args):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    #USING CS189 CODE
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset,
        [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory = False
    )

    device = torch.device('cuda')
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
        model = MLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm='layer').to(device)
    elif args.symmetry == 1:
        model = WMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, mask_params, norm='layer').to(device)
    elif args.symmetry == 2:
        model = SigmaMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm='layer').to(device)
    assert model is not None

    naive_num_params = sum(p.numel() for p in model.parameters())
    num_params = naive_num_params - model.count_unused_params()
    print('Naive param count:', naive_num_params)
    print('Actual param count:', num_params)

    epochs = args.epochs
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    train(model, optimizer, train_loader, device, n_epochs = epochs, lr_schedule = None, batch_size=args.batch_size)
    import pickle
    with open(f'experiments/interpolations-trained/trained_{'mlp' if args.symmetry == 0 else 'wmlp'}_{index}.pickle', 'wb') as handle:
        pickle.dump(dict(model=model), handle, protocol=pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':
    from torchvision import datasets, transforms
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
    args = parser.parse_args()

    with multiprocessing.Pool(16) as pool:
        pool.starmap(train_one, [(i, args) for i in range(64)])
