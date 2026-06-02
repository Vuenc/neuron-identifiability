import concurrent.futures
from typing import Dict, List
import hydra
import torch
import pathlib
from src.utils.record_activations import MODEL_OUTPUT_RECORDING_POINT, ActivationRecordingPoint, RecordInput
import train
import src.utils.rebasin.subspace_coherence
from src.utils.rebasin.subspace_coherence import SubspaceCoherenceResult
from contextlib import contextmanager
from checkpoint_directories import checkpoint_directories_by_architecture
import tqdm
import json
import argparse

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

# checkpoint_directories_mlp_nonorm = {
#     "mlp_symmetry0": "outputs/2026-01-15/17-07-22_mlp_mnist_sym-0__db8ehws3__y5oujjmq",
# }

def compute_subspace_coherence_results(
    checkpoint_path,
    device="cuda:0",
    data_info=None
) -> List[Dict]:
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path).parent)):
        cfg = hydra.compose(config_name="config")
    cfg.dataset.batch_size = 500
    print(cfg.dataset)
    if data_info is None:
        data_info = train.setup_data_loaders(cfg)

    model = train.create_model(cfg, mask_seed=-1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])

    activation_recording_points = [("lins.0", RecordInput), ("lins.1", RecordInput), ("lins.2", RecordInput), ("lins.3", RecordInput), MODEL_OUTPUT_RECORDING_POINT]
    coherence_results: Dict[ActivationRecordingPoint, SubspaceCoherenceResult] = (
        src.utils.rebasin.subspace_coherence.compute_representation_subspace_coherences(
            activation_recording_points=activation_recording_points,
            model=model,
            data_loader=data_info["train_loader"],
            target_explained_variance_ratio=0.9,
            device=device
        )
    )
    results = []
    with torch.no_grad():
        for recording_point, coherence_result in coherence_results.items():
                layer_name, hook_mode = recording_point
                results.append({
                    "layer": layer_name if layer_name != "" else ("model-input" if hook_mode == RecordInput else "model-output"),
                    "hook_mode": hook_mode.name,
                    "subspace_coherence": coherence_result.subspace_coherence,
                    "subspace_dimension": coherence_result.subspace_dimension,
                    "full_dimension": coherence_result.full_dimension,
                    "explained_variance_ratio": coherence_result.explained_variance_ratio,
                    "anisotropy_operator_norms": coherence_result.anisotropy_operator_norms,
                    "mean_anisotropy_operator_norm": coherence_result.mean_anisotropy_operator_norm,
                    "anisotropy_bounds": coherence_result.anisotropy_bounds,
                })
    return results

def main():
    DEFAULT_MODEL_RANGE = list(range(1, 5))
    DEFAULT_EPOCH_RANGE = list(range(0, 101, 5))
    DEFAULT_MAX_PARALLEL_PROCESSES = 22

    all_results = []

    parser = argparse.ArgumentParser(prog="measure_subspace_coherence.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--checkpoint-directory", type=str)
    parser.add_argument("--run-key")
    parser.add_argument("--no-suppress-prints", action="store_true")
    parser.add_argument("--parallel-processes", type=int, default=DEFAULT_MAX_PARALLEL_PROCESSES)
    parser.add_argument("--epochs", type=int, nargs="+", required=False)
    parser.add_argument("--model-indices", type=int, nargs="+", required=False)
    args = parser.parse_args()

    epoch_range = args.epochs if args.epochs is not None else DEFAULT_EPOCH_RANGE
    model_range = args.model_indices if args.model_indices is not None else DEFAULT_MODEL_RANGE

    if (args.checkpoint_directory is None) != (args.run_key is None):
        raise ValueError("Arguments --checkpoint-directory and --run-key must both be specified if one of them is specified.")
    if (args.checkpoint_directory is None) == (args.architecture is None):
        raise ValueError("Exactly one of the arguments --architecture and --checkpoint-directory must be specified!")

    if args.architecture:
        checkpoint_directories = checkpoint_directories_by_architecture[args.architecture]
    else:
        checkpoint_directories = {args.run_key: args.checkpoint_directory}

    # Assuming all are using the same dataset
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(list(checkpoint_directories.values())[0]))):
        _cfg = hydra.compose(config_name="config")
    _cfg.dataset.batch_size = 2**15
    epoch_range = [epoch for epoch in epoch_range if epoch <= _cfg.training.num_epochs] or [_cfg.training.num_epochs]
    data_info = train.setup_data_loaders(_cfg)

    checkpoint_path = lambda model_index, epoch: f"{checkpoint_directories[run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"

    for run_key in (tqdm_run := tqdm.tqdm(checkpoint_directories.keys())):
        tqdm_run.set_description(f"Run: {run_key}")
        for model_index in (tqdm_model := tqdm.tqdm(model_range, desc=run_key, leave=False)):
            tqdm_model.set_description(f"Model {model_index}")
            with suppress_prints(suppress=not args.no_suppress_prints):
                if args.parallel_processes > 1:
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_processes)
                else:
                    import types
                    executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
                model_results = [future.result() for future in [
                    executor.submit(compute_subspace_coherence_results,
                        **dict(checkpoint_path=checkpoint_path(model_index, epoch), data_info=data_info)
                    ) for epoch in epoch_range]
                ]
                if args.parallel_processes > 1:
                    executor.shutdown()
            for coherence_results, epoch in zip(model_results, epoch_range):
                all_results.append({
                    "run_key": run_key,
                    "model_index": model_index,
                    "epoch": epoch,
                    "coherence_results": coherence_results
                })
            with open(args.output_file, "w") as f:
                json.dump(all_results, f)

if __name__ == "__main__":
    main()
