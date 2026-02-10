
from __future__ import annotations
import torch
import pathlib
import hydra
import tqdm
import train
from collections import defaultdict
import enum
from typing import DefaultDict, Dict, List, NamedTuple, Tuple
from src.models.mlp import SparseLinear, NoiseLinear
import copy
from src.models.normalization import setup_normalization
from src.models.activation import setup_activation
import json
from src.utils.record_activations import record_activations, RecordInput
import argparse
from checkpoint_directories import checkpoint_directories_by_architecture

type PermutationName = str

def measure_distance_between_functional_classes(run_keys, model_index, epoch, output_path, num_outer_loop_samples: int, num_inner_descent_iterations: int, num_neurons_to_compare: int, data_info=None, device="cuda:0"):
    make_checkpoint_path = lambda model_index, epoch, run_key: f"{checkpoint_directories_by_architecture["mlp"][run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"
    results = {}
    for run_key in run_keys:
        checkpoint_path = make_checkpoint_path(model_index, epoch, run_key)

        with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path).parent)):
            cfg = hydra.compose(config_name="config")
        cfg.dataset.batch_size = 500
        if data_info is None:
            data_info = train.setup_data_loaders(cfg)

        model = train.create_model(cfg, mask_seed=-1).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        results[run_key] = estimate_neuron_pair_distances(model, data_loader=data_info["val_loader"], num_outer_loop_samples=num_outer_loop_samples, num_inner_descent_iterations=num_inner_descent_iterations, num_neurons_to_compare=num_neurons_to_compare, device=device)
        with open(output_path, "w") as f:
            json.dump({"num_outer_loop_samples": num_outer_loop_samples, "num_inner_descent_iterations": num_inner_descent_iterations, "results": results}, f)

class HausdorffDistanceEstimates(NamedTuple):
    distance_per_neuron: torch.Tensor
    weight_scale_1_per_neuron: torch.Tensor
    weight_scale_2_per_neuron: torch.Tensor

def estimate_hausdorff_distance(
        neurons_1_module: torch.nn.Sequential,
        neurons_2_module: torch.nn.Sequential,
        inputs: torch.Tensor,
        num_outer_loop_samples: int=100,
        num_inner_descent_iterations: int=3000,
        weight_decay: float=1e-3
) -> HausdorffDistanceEstimates:
    optim2 = torch.optim.Adam(neurons_2_module[0].parameters(), weight_decay=weight_decay)
    all_min_loss_per_neuron = []
    all_weight_scale_2_at_min_per_neuron = []
    for k in tqdm.tqdm(range(num_outer_loop_samples)):
        neurons_1_module[0].reset_parameters()
        neurons_2_module[0].reset_parameters()
        min_loss_per_neuron = None
        weight_scale_2_at_min_per_neuron = None
        out1 = neurons_1_module.forward(inputs).detach()  # Don't update neuron_1 here
        for i in (progress := tqdm.tqdm(range(num_inner_descent_iterations))):
            optim2.zero_grad()
            out2 = neurons_2_module.forward(inputs)
            loss_per_neuron = ((out1 - out2)**2).mean(dim=0)
            loss_per_neuron.sum().backward()
            optim2.step()
            with torch.no_grad():
                if min_loss_per_neuron is not None and weight_scale_2_at_min_per_neuron is not None:
                    min_loss_per_neuron = torch.minimum(min_loss_per_neuron, loss_per_neuron)
                    is_currently_min = (min_loss_per_neuron == loss_per_neuron)
                    weight_scale_2_at_min_per_neuron[is_currently_min] = neurons_2_module[0].weight[is_currently_min].norm(dim=1)
                else:
                    min_loss_per_neuron = loss_per_neuron.detach()
                    weight_scale_2_at_min_per_neuron = neurons_2_module[0].weight.norm(dim=1)

            # progress.set_description(f"Loss mean: {min_loss_per_neuron.mean().item():.4f}, Weight scales: {neurons_1_module[0].weight.norm(dim=1).mean():.3f}, {neurons_2_module[0].weight.norm(dim=1).mean():.3f}")
        all_min_loss_per_neuron.append(min_loss_per_neuron)
        all_weight_scale_2_at_min_per_neuron.append(weight_scale_2_at_min_per_neuron)
    maximization_result = torch.max(torch.stack(all_min_loss_per_neuron, dim=0), dim=0)
    distance_per_neuron = maximization_result.values
    weight_scale_1_per_neuron = neurons_1_module[0].weight.norm(dim=1)
    weight_scale_2_per_neuron = torch.stack(all_weight_scale_2_at_min_per_neuron, dim=0)[maximization_result.indices, torch.arange(len(maximization_result.indices))]
    return HausdorffDistanceEstimates(distance_per_neuron, weight_scale_1_per_neuron, weight_scale_2_per_neuron)

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

def create_neuron_subset_sparse_linear(other: SparseLinear | torch.nn.Linear, neuron_ids: List[int] | torch.Tensor) -> SparseLinear | torch.nn.Linear:
    with torch.no_grad():
        if isinstance(other, SparseLinear):
            new_lin = SparseLinear(other.weight.shape[1], len(neuron_ids), torch.Generator(), other.mask_constant)
            new_lin.weight[:] = other.weight[neuron_ids, :]
            new_lin.bias[:] = other.bias[neuron_ids]
            new_lin.mask[:] = other.mask[neuron_ids, :]
            new_lin.normal_mask[:] = other.normal_mask[neuron_ids, :]
            new_lin.mask_constant = other.mask_constant
        elif isinstance(other, NoiseLinear):
            new_lin = NoiseLinear(other.weight.shape[1], len(neuron_ids), torch.Generator(), mask_constant=other.mask_constant)
            new_lin.weight[:] = other.weight[neuron_ids, :]
            new_lin.bias[:] = other.bias[neuron_ids]
            new_lin.noise[:] = other.noise[neuron_ids, :]
            new_lin.mask_constant = other.mask_constant
        elif isinstance(other, torch.nn.Linear):
            new_lin = torch.nn.Linear(other.weight.shape[1], len(neuron_ids))
            new_lin.weight[:] = other.weight[neuron_ids, :]
            new_lin.bias[:] = other.bias[neuron_ids]
        else:
            raise ValueError(f"Unsupported type: {type(other)}")
    return new_lin.to(other.weight.device)

# TODO make configurable if norm is used or not, what kind of norm, etc.
def create_neuron_subset_sparse_linear_with_activation_and_norm(other: SparseLinear, neuron_ids: List[int] | torch.Tensor, wmlp, norm) -> torch.nn.Sequential:
    new_lin = create_neuron_subset_sparse_linear(other, neuron_ids)
    
    with torch.no_grad():
        # new_norm = wmlp.norm(hidden_dim=len(neuron_ids))
        # new_norm.weight[:] = norm.weight[neuron_ids]
        # new_norm = torch.nn.BatchNorm1d(len(neuron_ids), )
        new_activation = Lambda(setup_activation(wmlp.activation))

    return torch.nn.Sequential(
        new_lin,
        # new_norm,
        new_activation,
    ).to(new_lin.weight.device)

def estimate_neuron_pair_distances(model, data_loader, num_outer_loop_samples, num_inner_descent_iterations, num_neurons_to_compare, device="cuda:0"):
    [activations_by_layer,] = record_activations([("lins.0", RecordInput), ("lins.1", RecordInput), ("lins.2", RecordInput)], [model], data_loader, device=device)
    activations_by_layer = {layer_name: torch.cat(activations) for (layer_name, _), activations in activations_by_layer.items()}
    named_modules = dict(model.named_modules())
    outputs = []
    for layer_name, input_activations in tqdm.tqdm(activations_by_layer.items()):
        neuron_indices_1, neuron_indices_2 = torch.triu_indices(row=num_neurons_to_compare, col=num_neurons_to_compare, offset=-num_neurons_to_compare)
        neurons_1, neurons_2 = [
            # TODO fix the norm
            create_neuron_subset_sparse_linear_with_activation_and_norm(named_modules[layer_name], neuron_indices, model, norm="TODOREMOVETHISLATER")
            for neuron_indices in [neuron_indices_1, neuron_indices_2]
        ]
        hausdorff_distance_results = estimate_hausdorff_distance(neurons_1, neurons_2, input_activations, num_outer_loop_samples=num_outer_loop_samples, num_inner_descent_iterations=num_inner_descent_iterations)
        outputs.append({
            "layer": layer_name,
            "neuron_hausdorff_distances": {
                f"{id1}, {id2}": [distance, scale_1, scale_2]
                for id1, id2, distance, scale_1, scale_2 in zip(
                    neuron_indices_1.tolist(), neuron_indices_2.tolist(),
                    hausdorff_distance_results.distance_per_neuron.tolist(),
                    hausdorff_distance_results.weight_scale_1_per_neuron.tolist(),
                    hausdorff_distance_results.weight_scale_2_per_neuron.tolist()
                )
            },
        })
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser("measure_distance_between_functional_classes.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--outer-samples", type=int, required=True)
    parser.add_argument("--inner-iterations", type=int, required=True)
    parser.add_argument("--num-neurons", type=int, required=True)
    parser.add_argument("--run-keys", type=str, nargs="+", required=True)
    args = parser.parse_args()
    print(args.run_keys)
    measure_distance_between_functional_classes(args.run_keys, model_index=1, epoch=100, output_path=args.output_file, num_outer_loop_samples=args.outer_samples, num_inner_descent_iterations=args.inner_iterations, num_neurons_to_compare=args.num_neurons)
