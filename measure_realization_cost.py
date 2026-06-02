import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List
import hydra
import torch
import pathlib

from torch.nn import Linear

from src.models.mlp import MLP, WMLP, NoiseLinear, NoiseMLP, SparseLinear
from src.utils.record_activations import MODEL_OUTPUT_RECORDING_POINT, ActivationRecordingPoint, RecordInput
import train
import src.utils.rebasin.subspace_coherence
from src.utils.rebasin.subspace_coherence import SubspaceCoherenceResult, compute_gram_matrices, compute_subspace_coherence, estimate_subspace_basis_at_explained_variance
import src.utils.record_activations
from contextlib import contextmanager
from checkpoint_directories import checkpoint_directories_by_architecture
import tqdm
import json
import argparse
import numpy as np
import polars as pl
import random

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


def load_config(directory_or_checkpoint_path: str):
    path = pathlib.Path(directory_or_checkpoint_path)
    config_dir = path.parent if path.is_file() else path
    
    with hydra.initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        return hydra.compose(config_name="config")

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
    # MODEL_RANGE = list(range(1, 17, 1))
    # MODEL_RANGE = [1]
    # EPOCH_RANGE = list(range(0, 101, 5))
    # EPOCH_RANGE = list(range(0, 101, 5))
    # EPOCH_RANGE = [*range(10), *range(10, 101, 5)]
    # EPOCH_RANGE = [100]
    # EPOCH_RANGE = list(range(0, 11, 1))
    MAX_PARALLEL_PROCESSES = 15

    all_results = []

    parser = argparse.ArgumentParser(prog="measure_realization_cost.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="mlp-batchnorm")
    # parser.add_argument("--checkpoint-directories", type=str, nargs="+", required=False)
    # parser.add_argument("--run-key")
    parser.add_argument("--explained-variance", type=float, default=0.9)
    parser.add_argument("--no-suppress-prints", action="store_true")
    parser.add_argument("--parallel-processes", type=int, default=MAX_PARALLEL_PROCESSES)
    args = parser.parse_args()

    # if (args.checkpoint_directories is None) != (args.run_key is None):
    #     raise ValueError("Arguments --checkpoint-directory and --run-key must both be specified if one of them is specified.")
    # if (args.checkpoint_directories is None) == (args.architecture is None):
    #     raise ValueError("Exactly one of the arguments --architecture and --checkpoint-directory must be specified!")

    # if args.architecture:
    #     checkpoint_directories = checkpoint_directories_by_architecture[args.architecture]
    # else:
    #     checkpoint_directories = {args.run_key: args.checkpoint_directories}

    # # Assuming all are using the same dataset
    # with hydra.initialize(version_base=None, config_path=str(pathlib.Path(list(checkpoint_directories.values())[0]))):
    #     _cfg = hydra.compose(config_name="config")
    # _cfg.dataset.batch_size = 2**15

    if args.parallel_processes > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_processes)
    else:
        import types
        executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
    
    checkpoint_paths: List[str] = [f"{path}/checkpoint_epoch_100_model_1.pt" for path in checkpoint_directories_by_architecture[args.architecture].values()]

    with suppress_prints(suppress=not args.no_suppress_prints):
        all_results = list(tqdm.tqdm((
            {
                "checkpoint_path": checkpoint_path,
                "run_directory": str(pathlib.Path(checkpoint_path).parent),
                "realization_cost_results": future.result()
            }
            for future, checkpoint_path in (
                (
                    executor.submit(compute_realization_cost_results,
                        **dict(checkpoint_path=checkpoint_path, target_explained_variance_ratio=args.explained_variance)
                    ),
                    checkpoint_path
                )
                for checkpoint_path in checkpoint_paths
            )
        ), total=len(checkpoint_paths)))
    pl.DataFrame(all_results).write_parquet(args.output_file, compression="zstd")

if __name__ == "__main__":
    main()
