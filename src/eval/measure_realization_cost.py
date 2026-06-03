import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List
import torch
import pathlib

from torch.nn import Linear

from src.models.mlp import MLP, WMLP, NoiseLinear, NoiseMLP, SparseLinear
from src.utils.record_activations import RecordInput
import train
from src.utils.rebasin.subspace_coherence import compute_gram_matrices, compute_subspace_coherence, estimate_subspace_basis_at_explained_variance
import src.utils.record_activations
from contextlib import contextmanager
from src.eval.checkpoint_directories import checkpoint_directories_by_architecture
import tqdm
import argparse
import numpy as np
import polars as pl
from src.utils.load_config import load_config

@contextmanager
def suppress_prints(suppress=True):
    """Context manager to suppress all print statements."""
    if not suppress:
        yield
    else:
        original_print = __builtins__.print
        __builtins__.print = lambda *args, **kwargs: None
        try:
            yield original_print
        finally:
            __builtins__.print = original_print


def load_model(checkpoint_path: str, device="cuda:0"):
    cfg = load_config(checkpoint_path)
    model = train.create_model(cfg, mask_seed=-1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    return model

@dataclass
class MahalanobisDistancesResult:
    mahalanobis_distances_squared: np.ndarray
    projected_center_distances: np.ndarray

def compute_mahalanobis_distances(layer: Linear | SparseLinear | NoiseLinear, subspace_basis) -> MahalanobisDistancesResult:
    gram_matrices = compute_gram_matrices(subspace_basis, layer=layer)
    effective_weights = (
        layer.effective_weight() if isinstance(layer, SparseLinear) or isinstance(layer, NoiseLinear)
        else layer.weight if isinstance(layer, Linear)
        else None
    )
    assert effective_weights is not None

    fixed_weights = (
        (1 - layer.mask.detach()) * layer.mask_constant * layer.normal_mask if isinstance(layer, SparseLinear)
        else layer.mask_constant * layer.noise if  isinstance(layer, NoiseLinear)
        else torch.zeros_like(effective_weights) if isinstance(layer, Linear)
        else None
    )
    assert fixed_weights is not None

    # Compute quantities used in the paper:
    # - induced_preactivation_coeffs is $a$ in the paper
    # - projected_centers is $v_i$ in the paper
    # - diffs_from_centers[:, i, j] is $a_j - v_i \in \mathbb{R}^k$ in the paper
    # - gram_matrices_pinv[i] is S_i^\dagger in the paper.
    induced_preactivation_coeffs = subspace_basis.T @ effective_weights.T
    projected_centers = subspace_basis.T @ fixed_weights.T
    diffs_from_centers = induced_preactivation_coeffs[:, None, :] - projected_centers[:, :, None]
    gram_matrices_pinv = torch.linalg.pinv(gram_matrices)

    # Compute the Mahalanobis distances $\sqrt((a_j - v_i)^T S_i^\dagger (a_j - vi_i))$.
    mahalanobis_distances_squared = torch.einsum("kij,ikK,Kij->ij", diffs_from_centers, gram_matrices_pinv, diffs_from_centers)
    projected_center_distances = torch.linalg.norm(
        projected_centers[:, :, None] - projected_centers[:, None, :],
        axis=0
    )

    return MahalanobisDistancesResult(
        mahalanobis_distances_squared=mahalanobis_distances_squared.detach().cpu().numpy(),
        projected_center_distances=projected_center_distances.detach().cpu().numpy(),
    )

def compute_transposition_costs(mahalanobis_distances_squared: np.ndarray) -> np.ndarray:
    return (mahalanobis_distances_squared + mahalanobis_distances_squared.T) - np.diag(mahalanobis_distances_squared) - np.diag(mahalanobis_distances_squared)[:,None]

def compute_realization_cost_results(
        checkpoint_path,
        target_explained_variance_ratio = 0.9
) -> List[Dict]:
    cfg = load_config(checkpoint_path)
    model = load_model(checkpoint_path)
    data_info = train.setup_data_loaders(load_config(checkpoint_path))

    assert isinstance(model, MLP) or isinstance(model, WMLP) or isinstance(model, NoiseMLP)

    activation_recording_points = [(f"lins.{i}", RecordInput) for i in range(cfg.model.num_layers)]
    [recorded_activations,] = src.utils.record_activations.record_activations(
        activation_recording_points, models=[model], data_loader=data_info["train_loader"]
    )

    subspace_basis_and_variance_by_recording_point = {
        recording_point: estimate_subspace_basis_at_explained_variance(
            torch.cat(activations), target_explained_variance_ratio)
        for recording_point, activations in recorded_activations.items()
    }
    named_modules: Dict[str, Any] = dict(model.named_modules()) # type: ignore

    mahalanobis_distances_by_recording_point = {
        recording_point: compute_mahalanobis_distances(named_modules[recording_point[0]], subspace_basis)
        for recording_point, (subspace_basis, _) in subspace_basis_and_variance_by_recording_point.items()
    }

    transposition_costs_by_recording_point = {
        recording_point: compute_transposition_costs(mahalanobis_distances.mahalanobis_distances_squared)
        for recording_point, mahalanobis_distances in mahalanobis_distances_by_recording_point.items()
    }

    subspace_coherences_by_recording_point = {
        recording_point: compute_subspace_coherence(subspace_basis)
        for recording_point, (subspace_basis, _) in subspace_basis_and_variance_by_recording_point.items()   
    }

    results = [{
            "layer": recording_point[0] if recording_point != "" else ("model-input" if recording_point[1] == RecordInput else "model-output"),
            "hook_mode": recording_point[1].name,
            "mahalanobis_distance_squared": mahalanobis_distances_by_recording_point[recording_point].mahalanobis_distances_squared.tolist(),
            "proj_center_distances": mahalanobis_distances_by_recording_point[recording_point].projected_center_distances.tolist(),
            "transposition_cost_squared": transposition_costs_by_recording_point[recording_point].tolist(),
            "subspace_dimension": subspace_basis_and_variance_by_recording_point[recording_point][0].shape[1],
            "full_dimension": subspace_basis_and_variance_by_recording_point[recording_point][0].shape[0],
            "explained_variance_ratio": subspace_basis_and_variance_by_recording_point[recording_point][1],
            "subspace_coherence": subspace_coherences_by_recording_point[recording_point],
        }
        for recording_point in activation_recording_points
    ]
    return results


def main():
    DEFAULT_MODEL_RANGE = [1]
    DEFAULT_EPOCH_RANGE = [100]
    MAX_PARALLEL_PROCESSES = 15

    all_results = []

    parser = argparse.ArgumentParser(prog="measure_realization_cost.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="mlp-batchnorm")
    parser.add_argument("--explained-variance", type=float, default=0.9)
    parser.add_argument("--no-suppress-prints", action="store_true")
    parser.add_argument("--parallel-processes", type=int, default=MAX_PARALLEL_PROCESSES)
    parser.add_argument("--epochs", type=int, nargs="+", required=False)
    parser.add_argument("--model-indices", type=int, nargs="+", required=False)
    args = parser.parse_args()

    epoch_range = args.epochs if args.epochs is not None else DEFAULT_EPOCH_RANGE
    model_range = args.model_indices if args.model_indices is not None else DEFAULT_MODEL_RANGE

    if args.parallel_processes > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_processes)
    else:
        import types
        executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
    
    # checkpoint_paths: List[str] = [f"{path}/checkpoint_epoch_100_model_1.pt" for path in .values()]
    checkpoint_directories = checkpoint_directories_by_architecture[args.architecture]
    checkpoint_path = lambda model_index, epoch, checkpoint_directory: f"{checkpoint_directory}/checkpoint_epoch_{epoch}_model_{model_index}.pt"

    with suppress_prints(suppress=not args.no_suppress_prints):
        all_results = list(tqdm.tqdm((
            {
                "checkpoint_path": (path := checkpoint_path(model_index, epoch, checkpoint_directory)),
                "run_directory": str(pathlib.Path(path).parent),
                "realization_cost_results": future.result(),
                "model_index": model_index,
                "epoch": epoch
            }
            for future, model_index, epoch, checkpoint_directory in (
                (
                    executor.submit(compute_realization_cost_results,
                        **dict(checkpoint_path=checkpoint_path(model_index, epoch, checkpoint_directory), target_explained_variance_ratio=args.explained_variance)
                    ),
                    model_index, epoch, checkpoint_directory
                )
                for model_index in model_range
                for epoch in epoch_range
                for checkpoint_directory in checkpoint_directories.values()
            )
        ), total=len(model_range)*len(epoch_range)*len(checkpoint_directories)))
    pl.DataFrame(all_results).write_parquet(args.output_file, compression="zstd")

if __name__ == "__main__":
    main()
