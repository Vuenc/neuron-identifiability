from __future__ import annotations
import math
import torch
import pathlib
import hydra
import tqdm
import train
from typing import Dict, List, NamedTuple
from src.models.mlp import SparseLinear, NoiseLinear
from src.models.activation import setup_activation
from src.utils.record_activations import record_activations, RecordInput
import argparse
from checkpoint_directories import checkpoint_directories_by_architecture
import polars as pl

type PermutationName = str

class RealizationCostEstimates(NamedTuple):
    objective_full_per_neuron: torch.Tensor
    objective_mse_per_neuron: torch.Tensor
    objective_weight_regularization_per_neuron: torch.Tensor
    weight_scale_1_per_neuron: torch.Tensor
    weight_scale_2_per_neuron: torch.Tensor

def estimate_realization_costs(
        neurons_1_module: torch.nn.Sequential,
        neurons_2_module: torch.nn.Sequential,
        inputs: torch.Tensor,
        num_inner_descent_iterations: int=3000,
        beta: float=10.
) -> RealizationCostEstimates:
    optim2 = torch.optim.AdamW(neurons_2_module[0].parameters(), weight_decay=0., lr=5e-3)
    out1 = neurons_1_module.forward(inputs).detach()  # Don't update neuron_1 here

    def trainable_weights(module):
        if isinstance(module, SparseLinear):
            return module.weight * (module.mask == 1).to(module.weight.dtype)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, NoiseLinear):
            return module.weight
        raise ValueError("Unknown module type:", type(module))

    loss_per_neuron, objective_mse, objective_weight_regularization = None, None, None

    for _ in (progress := tqdm.tqdm(range(num_inner_descent_iterations), leave=False)):
        optim2.zero_grad()
        out2 = neurons_2_module.forward(inputs)
        objective_mse = 1/beta * (out1 - out2)**2
        objective_weight_regularization = (trainable_weights(neurons_2_module[0]) ** 2).sum(dim=1)
        loss_per_neuron = (objective_mse.mean(dim=0) + objective_weight_regularization)
        loss_per_neuron.sum().backward()
        optim2.step()

        progress.set_description(f"Loss mean: {loss_per_neuron.mean().item():.4f} = {objective_mse.mean().item()*beta:.4f}/{beta} + {objective_weight_regularization.mean().item():.4f}, Weight scales: {trainable_weights(neurons_1_module[0]).norm(dim=1).mean():.3f}, {trainable_weights(neurons_2_module[0]).norm(dim=1).mean():.3f}")

    assert loss_per_neuron is not None and objective_mse is not None and objective_weight_regularization is not None

    return RealizationCostEstimates(
        objective_full_per_neuron=loss_per_neuron.detach(),
        objective_mse_per_neuron=objective_mse.mean(dim=0).detach(),
        objective_weight_regularization_per_neuron=objective_weight_regularization.detach(),
        weight_scale_1_per_neuron=neurons_1_module[0].weight.norm(dim=1),
        weight_scale_2_per_neuron=neurons_2_module[0].weight.norm(dim=1)
    )

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def create_neuron_subset_linear_layer(other: SparseLinear | NoiseLinear | torch.nn.Linear, neuron_ids: List[int] | torch.Tensor) -> SparseLinear | NoiseLinear | torch.nn.Linear:
    """
    Create a Linear/SparseLinear/NoiseLinear from an existing Linear/SparseLinear/NoiseLinear
    that contains exactly the neurons at IDs `neuron_ids`.
    """
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

def create_neuron_subset_linear_layer_with_activation_and_norm(other: torch.nn.Linear | SparseLinear | NoiseLinear, neuron_ids: List[int] | torch.Tensor, wmlp) -> torch.nn.Sequential:
    new_lin = create_neuron_subset_linear_layer(other, neuron_ids)

    with torch.no_grad():
        # new_norm = wmlp.norm(hidden_dim=len(neuron_ids))
        # new_norm.weight[:] = norm.weight[neuron_ids]
        if not isinstance(wmlp.norm, torch.nn.BatchNorm1d):
            raise ValueError(f"Trained model must use batch norm. Found: {type(wmlp.norm)}")
        new_norm = torch.nn.BatchNorm1d(len(neuron_ids), )
        new_norm.eval()
        new_activation = Lambda(setup_activation(wmlp.activation))

    return torch.nn.Sequential(
        new_lin,
        new_norm,
        new_activation,
    ).to(new_lin.weight.device)

def estimate_neuron_pair_distances(
        model,
        data_loader,
        cfg,
        num_inner_descent_iterations: int,
        base_num_neurons: int,
        num_neuron_pairs_to_compare: int,
        data_subsampling_ratio: float,
        betas: List[float],
        device="cuda:0"
) -> List[Dict]:
    activation_recording_points = [(f"lins.{i}", RecordInput) for i in range(cfg.model.num_layers)]
    [activations_by_layer,] = record_activations(activation_recording_points, [model], data_loader, device=device)
    dataset_size = sum(activations.shape[0] for activations in next(iter(activations_by_layer.values())))
    subsampling_generator = torch.Generator()
    subsampling_generator.manual_seed(4215)
    dataset_permutation = torch.randperm(dataset_size, generator=subsampling_generator)[:math.floor(dataset_size * data_subsampling_ratio)]
    activations_by_layer = {layer_name: torch.cat(activations)[dataset_permutation] for (layer_name, _), activations in activations_by_layer.items()}

    named_modules = dict(model.named_modules())
    outputs: List[Dict] = []
    for layer_name, input_activations in tqdm.tqdm(activations_by_layer.items(), leave=False):
        num_neurons = min(base_num_neurons, named_modules[layer_name].weight.shape[0])
        neuron_indices_1, neuron_indices_2 = torch.triu_indices(row=num_neurons, col=num_neurons, offset=-num_neurons)
        subset = torch.randperm(neuron_indices_1.shape[0])[:num_neuron_pairs_to_compare]
        neuron_indices_1, neuron_indices_2 = neuron_indices_1[subset], neuron_indices_2[subset]
        print(subset.shape)

        neurons_1, neurons_2 = [
            create_neuron_subset_linear_layer_with_activation_and_norm(named_modules[layer_name], neuron_indices, model)
            for neuron_indices in [neuron_indices_1, neuron_indices_2]
        ]
        for beta in tqdm.tqdm(betas, leave=False):
            realization_cost_results = estimate_realization_costs(neurons_1, neurons_2, input_activations, num_inner_descent_iterations=num_inner_descent_iterations, beta=beta)
            outputs.append({
                "layer": layer_name,
                "beta": float(beta),
                "neuron_realization_costs": [
                    dict(id1=id1, id2=id2, objective_full=objective_full, objective_mse=objective_mse, objective_weight_regularization=objective_weight_regularization, weight_scale_1=weight_scale_1, weight_scale_2=weight_scale_2)
                    for id1, id2, objective_full, objective_mse, objective_weight_regularization, weight_scale_1, weight_scale_2 in zip(
                        neuron_indices_1.tolist(),
                        neuron_indices_2.tolist(),
                        realization_cost_results.objective_full_per_neuron.tolist(),
                        realization_cost_results.objective_mse_per_neuron.tolist(),
                        realization_cost_results.objective_weight_regularization_per_neuron.tolist(),
                        realization_cost_results.weight_scale_1_per_neuron.tolist(),
                        realization_cost_results.weight_scale_2_per_neuron.tolist()
                    )
                ]
            })
    return outputs

def measure_realization_costs_for_runs(
        run_keys,
        model_index,
        epoch,
        output_path,
        architecture,
        num_inner_descent_iterations: int,
        base_num_neurons: int,
        num_neuron_pairs_to_compare: int,
        data_subsampling_ratio: float,
        betas: List[float],
        data_info=None,
        device="cuda:0"
):
    make_checkpoint_path = lambda model_index, epoch, run_key: f"{checkpoint_directories_by_architecture[architecture][run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"
    results = {}
    data_info_provided = data_info is not None
    for run_key in run_keys or tqdm.tqdm(checkpoint_directories_by_architecture[architecture].keys(), leave=False):
        checkpoint_path = make_checkpoint_path(model_index, epoch, run_key)

        with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path).parent)):
            cfg = hydra.compose(config_name="config")
        cfg.dataset.batch_size = 500
        if not data_info_provided:
            data_info = train.setup_data_loaders(cfg)
        assert data_info is not None

        model = train.create_model(cfg, mask_seed=-1).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
        results[run_key] = estimate_neuron_pair_distances(
            model=model, data_loader=data_info["val_loader"],
            cfg=cfg,
            num_inner_descent_iterations=num_inner_descent_iterations,
            base_num_neurons=base_num_neurons,
            num_neuron_pairs_to_compare=num_neuron_pairs_to_compare,
            data_subsampling_ratio=data_subsampling_ratio,
            betas=betas,
            device=device
        )
        pl.DataFrame({
            "num_inner_descent_iterations": num_inner_descent_iterations,
            "results": results
        }).write_parquet(output_path, compression="zstd")


def main():
    parser = argparse.ArgumentParser("measure_realization_cost_ridge_regression.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--inner-iterations", type=int, required=True)
    parser.add_argument("--num-neurons", type=int, required=True)
    parser.add_argument("--num-neuron-pairs", type=int, required=True)
    parser.add_argument("--run-keys", type=str, nargs="+", required=False)
    parser.add_argument("--betas", type=float, nargs="+", required=False, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--architecture", type=str, default="mlp-batchnorm")
    parser.add_argument("--data-subsampling-ratio", type=float, default=1.0)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    print(args.run_keys)
    measure_realization_costs_for_runs(
        args.run_keys,
        model_index=1,
        epoch=args.epoch,
        output_path=args.output_file,
        architecture=args.architecture,
        num_inner_descent_iterations=args.inner_iterations,
        base_num_neurons=args.num_neurons,
        num_neuron_pairs_to_compare=args.num_neuron_pairs,
        data_subsampling_ratio=args.data_subsampling_ratio,
        betas=args.betas
    )


if __name__ == "__main__":
    main()