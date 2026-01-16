import concurrent.futures
import hydra
import torch
import pathlib
import train
import src.utils.rebasin.activation_matching
import numpy as np
from contextlib import contextmanager

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

checkpoint_directories = {
    "mlp_symmetry0": "outputs/2025-12-17/12-18-16_mlp_mnist_sym-0__3pb1rw8b__kk13eg0q",
    "mlp_symmetry1_kappa0": "outputs/2025-12-17/14-20-51_mlp_mnist_sym-1__m3z8jvf9__iwzo2u22",
    "mlp_symmetry1_kappa1": "outputs/2025-12-17/12-30-55_mlp_mnist_sym-1__3pb1rw8b__4b0gt1xm",
    "mlp_symmetry1_kappaPerLayer": "outputs/2025-12-17/12-46-23_mlp_mnist_sym-1__3pb1rw8b__8fvvk8f3",
    "mlp_symmetry2": "outputs/2025-12-17/13-01-05_mlp_mnist_sym-2__3pb1rw8b__48q9fddu",
    "mlp_symmetry3_kappa0": "outputs/2025-12-17/13-16-26_mlp_mnist_sym-3__3pb1rw8b__tq3utpfq",
    "mlp_symmetry3_kappa1": "outputs/2025-12-17/13-30-38_mlp_mnist_sym-3__3pb1rw8b__ruzzxkpy",
    "mlp_symmetry3_kappaPerLayer": "outputs/2025-12-17/13-45-10_mlp_mnist_sym-3__3pb1rw8b__e4vv3n8v",
}

checkpoint_directories_resnet = {
    "resnet_symmetry0": "outputs/2025-12-18/19-14-23_resnet_cifar_sym-0__r3aiubzb__au3i07iw",
    "resnet_symmetry1_kappa0": "outputs/2025-12-18/23-27-30_resnet_cifar_sym-1__r3aiubzb__6xrlc0ln",
    "resnet_symmetry1_kappa2": "outputs/2025-12-19/17-19-27_resnet_cifar_sym-1__38uctfm6__2y80rlyo",
    # "resnet_symmetry2": "",
    # "resnet_symmetry3_kappa0": "outputs/2025-12-19/03-50-13_resnet_cifar_sym-3__r3aiubzb__86k7951d/config.yaml",
    # "resnet_symmetry3_kappa2": "",
}

def compute_activation_matching_results(checkpoint_path_1, checkpoint_path_2, device="cuda:0", data_info=None):
    # epoch, model_1_index, model_2_index):
    # checkpoint_paths = [
    #     f"{checkpoint_directories[RUN_KEY]}/checkpoint_epoch_{epoch}_model_{model_1_index}.pt",
    #     f"{checkpoint_directories[RUN_KEY]}/checkpoint_epoch_{epoch}_model_{model_2_index}.pt"
    # ]

    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path_1).parent)):
        cfg = hydra.compose(config_name="config")
    cfg.dataset.batch_size = 500
    if data_info is None:
        data_info = train.setup_data_loaders(cfg)

    model1, model2 = [train.create_model(cfg, mask_seed=-1).to(device) for _ in range(2)]
    model1.load_state_dict(torch.load(checkpoint_path_1, map_location=device)["model_state_dict"])
    model2.load_state_dict(torch.load(checkpoint_path_2, map_location=device)["model_state_dict"])

    permutation_spec = src.utils.rebasin.mlp_permutation_spec(3, norm=True, bias=True)
    activation_matching_results = src.utils.rebasin.activation_matching.activation_matching(
        permutation_spec, model1, model2, data_info["train_loader"], device=device
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
        # print(f"\nLayer: {layer_name}")
        # print(f"  (mean) Frobenius inner product (identity)               : {activation_similarities[range(d), range(d)].mean():.1f}")
        # print(f"  (mean) Frobenius inner product (best permutation)       : {activation_similarities[*permutation].mean():.1f}")
        # random_permutation_objectives = [activation_similarities[np.arange(d), np.random.permutation(d)].mean() for _ in range(100)]
        # print(f"  (mean) Frobenius inner product (100 random permutations): {np.mean(random_permutation_objectives):.1f} ± {np.std(random_permutation_objectives, ddof=1):.1f}")

    return results

def main():
    MODEL_1_RANGE = list(range(1, 17, 2))
    EPOCH_RANGE = list(range(0, 101, 5))
    # EPOCH_RANGE = list(range(0, 11, 1))
    MAX_PARALLEL_PROCESSES = 4 

    all_results = []
    import tqdm
    import json
    import argparse

    parser = argparse.ArgumentParser("measure_rebasinability.py")
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    # Assuming all are using the same dataset
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(list(checkpoint_directories.values())[0]))):
        _cfg = hydra.compose(config_name="config")
    data_info = train.setup_data_loaders(_cfg)

    checkpoint_path = lambda model_index, epoch: f"{checkpoint_directories[run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"

    for run_key in (tqdm_run := tqdm.tqdm(checkpoint_directories.keys())):
        tqdm_run.set_description(f"Run: {run_key}")
        for model1_index in (tqdm_model_pair := tqdm.tqdm(MODEL_1_RANGE, desc=run_key, leave=False)):
            model2_index = model1_index + 1
            tqdm_model_pair.set_description(f"Model pair: (Model {model1_index}, Model {model2_index})")
            with suppress_prints(suppress=False):
                with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES) as executor:
                    # import types
                    # executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
                    model_pair_results = [future.result() for future in [
                        executor.submit(compute_activation_matching_results,
                            **dict(checkpoint_path_1=checkpoint_path(model1_index, epoch), checkpoint_path_2=checkpoint_path(model2_index, epoch), data_info=data_info)
                        ) for epoch in EPOCH_RANGE]
                    ]
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
