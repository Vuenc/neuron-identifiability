import concurrent.futures
import hydra
import torch
import pathlib
import train
import src.utils.rebasin.activation_matching
from src.utils.rebasin import ActivationCorrelationMode
import numpy as np
from contextlib import contextmanager
from .checkpoint_directories import checkpoint_directories_by_architecture

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

def compute_activation_matching_results(checkpoint_path_1, checkpoint_path_2, device="cuda:0", data_info=None):
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path_1).parent)):
        cfg = hydra.compose(config_name="config")
    cfg.dataset.batch_size = 512
    if data_info is None:
        data_info = train.setup_data_loaders(cfg)

    model1, model2 = [train.create_model(cfg, mask_seed=-1).to(device) for _ in range(2)]
    model1.load_state_dict(torch.load(checkpoint_path_1, map_location=device)["model_state_dict"])
    model2.load_state_dict(torch.load(checkpoint_path_2, map_location=device)["model_state_dict"])
    model1.eval()
    model2.eval()

    correlation_modes = None

    if cfg.model.name == "mlp_mnist":
        permutation_spec = src.utils.rebasin.mlp_permutation_spec(3, norm=True, bias=True)
        correlation_modes = [ActivationCorrelationMode.DotProduct, ActivationCorrelationMode.PearsonCorrelationWithZeroForConstant]
    elif cfg.model.name == "resnet_cifar":
        permutation_spec = src.utils.rebasin.resnet20_permutation_spec()
        # ReLU resnets tend to have all-zero recorded activations for one channel, which crashes usual Pearson correlation
        correlation_modes = [ActivationCorrelationMode.PearsonCorrelationWithZeroForConstant]
        # We shorten the train loader: not enough memory for everything
        data_info["train_loader"] = [d for _, d in zip(range(10), data_info["train_loader"])]
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")
    activation_matching_results = src.utils.rebasin.activation_matching.activation_matching(
        permutation_spec, model1, model2, data_info["train_loader"], device=device, correlation_modes=correlation_modes
    )
    results = []
    with torch.no_grad():
        for (matching_mode, correlation_mode), results_per_layer in activation_matching_results.items():
            for layer_name, matching_result in results_per_layer.items():
                d = len(matching_result.optimal_permutation)
                optimal_permutation_objective = matching_result.activation_similarities[range(d), matching_result.optimal_permutation].mean().item()
                identity_objective = matching_result.activation_similarities[range(d), range(d)].mean().item()
                random_permutation_objectives = [matching_result.activation_similarities[range(d), torch.randperm(d)].mean().item() for _ in range(100)]

                results.append({
                    "matching_mode": matching_mode,
                    "correlation_mode": correlation_mode.value,
                    "layer": layer_name,
                    "objectives": {
                        "optimal": optimal_permutation_objective,
                        "identity": identity_objective,
                        "random": random_permutation_objectives,
                        "random_mean": np.mean(random_permutation_objectives),
                        "random_std": np.std(random_permutation_objectives, ddof=1)
                    },
                    "permutations": {
                        "optimal": matching_result.optimal_permutation.tolist(),
                    }
                    # "activation_similarities": matching_result.activation_similarities.tolist()
                })

    return results

def main():
    MODEL_1_RANGE = list(range(1, 5, 2))
    EPOCH_RANGE = list(range(0, 101, 5))
    # EPOCH_RANGE = list(range(0, 11, 1))

    all_results = []
    import tqdm
    import json
    import argparse

    parser = argparse.ArgumentParser("measure_rebasinability.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--parallel-processes", type=int, default=1)
    args = parser.parse_args()
    checkpoint_directories = checkpoint_directories_by_architecture[args.architecture]

    # Assuming all are using the same dataset
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(list(checkpoint_directories.values())[0]))):
        _cfg = hydra.compose(config_name="config")
    _cfg.dataset.enable_ffcv = False
    data_info = train.setup_data_loaders(_cfg)

    checkpoint_path = lambda model_index, epoch: f"{checkpoint_directories[run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"

    for run_key in (tqdm_run := tqdm.tqdm(checkpoint_directories.keys())):
        tqdm_run.set_description(f"Run: {run_key}")
        for model1_index in (tqdm_model_pair := tqdm.tqdm(MODEL_1_RANGE, desc=run_key, leave=False)):
            model2_index = model1_index + 1
            tqdm_model_pair.set_description(f"Model pair: (Model {model1_index}, Model {model2_index})")
            with suppress_prints(suppress=False):
                if args.parallel_processes > 1:
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_processes)
                else:
                    import types
                    executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
                model_pair_results = list(tqdm.tqdm([future.result() for future in [
                    executor.submit(compute_activation_matching_results,
                        **dict(checkpoint_path_1=checkpoint_path(model1_index, epoch), checkpoint_path_2=checkpoint_path(model2_index, epoch), data_info=data_info)
                    ) for epoch in EPOCH_RANGE]
                ]))
                if args.parallel_processes > 1:
                    executor.shutdown()
            for matching_results, epoch in zip(model_pair_results, EPOCH_RANGE):
                all_results.append({
                    "run_key": run_key,
                    "model1_index": model1_index,
                    "model2_index": model2_index,
                    "epoch": epoch,
                    "matching_results": matching_results
                })
            with open(args.output_file, "w") as f:
                json.dump(all_results, f)

if __name__ == "__main__":
    main()
