import concurrent.futures
import hydra
import torch
import pathlib
import train
import numpy as np
from contextlib import contextmanager

import collections
from typing import DefaultDict, Dict, List, NamedTuple
import src.utils.rebasin
from src.utils.rebasin.common import PermutationSpec, HookMode, LayerName
from .checkpoint_directories import checkpoint_directories_by_architecture

class ActivationPatternResult(NamedTuple):
  agreement_ratio: float
  positive_ratio_a: float
  positive_ratio_b: float
  std_per_neuron_a: float
  std_per_neuron_b: float
  std_per_input_a: float
  std_per_input_b: float
  num_activations: int

def activation_patterns(
    permutation_spec: PermutationSpec,
    model_a,
    model_b,
    data_loader,
    device="cuda:0"
) -> Dict[str, Dict[LayerName, ActivationPatternResult]]:
    recorded_activations_by_mode_a: Dict[str, DefaultDict[LayerName, List[torch.Tensor]]] = {}
    recorded_activations_by_mode_b: Dict[str, DefaultDict[LayerName, List[torch.Tensor]]] = {}
    needed_matching_modes = [mode for mode in permutation_spec.activation_matching_modes.items() if mode[0] == "post_activation_function"]

    # Register hooks to save activations from layers that should be permuted
    for (model, recorded_activations_by_mode) in (model_a, recorded_activations_by_mode_a), (model_b, recorded_activations_by_mode_b):
        for activation_matching_mode_name, perm_to_hook_description in needed_matching_modes:

            recorded_activations_by_mode[activation_matching_mode_name] = collections.defaultdict(lambda: [])
            recorded_activations_current_mode = recorded_activations_by_mode[activation_matching_mode_name]
            named_modules = dict(model.named_modules())
            for (layer_name, hook_mode) in perm_to_hook_description.values():
                if hook_mode == HookMode.RecordInput:
                    def record_input_hook(module, input, output, layer_name=layer_name, recorded_activations_current_mode=recorded_activations_current_mode):
                        recorded_activations_current_mode[layer_name].append(input[0])
                    forward_hook = record_input_hook
                else:
                    def record_output_hook(module, input, output, layer_name=layer_name, recorded_activations_current_mode=recorded_activations_current_mode):
                        recorded_activations_current_mode[layer_name].append(output)
                    forward_hook = record_output_hook
                named_modules[layer_name].register_forward_hook(forward_hook)

    # Forward the dataset through the models
    with torch.no_grad():
        # Only iterate the data loader once, so the models see the data in the same order
        for input, _ in data_loader:
            for model in [model_a, model_b]:
                model.forward(input.to(device))

    # Compute efficient permutations
    results_by_mode: DefaultDict[str, Dict[LayerName, ActivationPatternResult]] = collections.defaultdict(lambda: {})
    for activation_matching_mode_name, perm_to_hook_description in needed_matching_modes:
        for layer_name, _ in perm_to_hook_description.values():
            activations_a = torch.cat(recorded_activations_by_mode_a[activation_matching_mode_name][layer_name])
            activations_b = torch.cat(recorded_activations_by_mode_b[activation_matching_mode_name][layer_name])

            activation_pattern_a = (activations_a > 0)
            activation_pattern_b = (activations_b > 0)

            results_by_mode[activation_matching_mode_name][layer_name] = ActivationPatternResult(
                agreement_ratio=(activation_pattern_a == activation_pattern_b).float().mean().item(),
                positive_ratio_a=activation_pattern_a.float().mean().item(),
                positive_ratio_b=activation_pattern_b.float().mean().item(),
                std_per_neuron_a=activation_pattern_a.float().std(dim=0).mean().item(),
                std_per_neuron_b=activation_pattern_b.float().std(dim=0).mean().item(),
                std_per_input_a=activation_pattern_a.float().std(dim=1).mean().item(),
                std_per_input_b=activation_pattern_b.float().std(dim=1).mean().item(),
                num_activations=activation_pattern_a.shape[1]
            )

    return dict(results_by_mode)


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

def compute_activation_pattern_results(checkpoint_path_1, checkpoint_path_2, device="cuda:0", data_info=None):
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path_1).parent)):
        cfg = hydra.compose(config_name="config")
    cfg.dataset.batch_size = 500
    if data_info is None:
        data_info = train.setup_data_loaders(cfg)

    model1, model2 = [train.create_model(cfg, mask_seed=-1).to(device) for _ in range(2)]
    model1.load_state_dict(torch.load(checkpoint_path_1, map_location=device)["model_state_dict"])
    model2.load_state_dict(torch.load(checkpoint_path_2, map_location=device)["model_state_dict"])

    permutation_spec = src.utils.rebasin.mlp_permutation_spec(3, norm=True, bias=True)
    activation_pattern_results = activation_patterns(
        permutation_spec, model1, model2, data_info["train_loader"], device=device
    )
    results = []
    with torch.no_grad():
        for matching_mode, results_per_layer in activation_pattern_results.items():
            for layer_name, result in results_per_layer.items():
                results.append({
                    "hook_location": matching_mode,
                    "layer": layer_name,
                    **{key: value for key, value in result._asdict().items()}
                })

    return results

def main():
    MODEL_1_RANGE = list(range(1, 17, 2))
    EPOCH_RANGE = list(range(0, 101, 20))
    # EPOCH_RANGE = list(range(0, 11, 1))
    MAX_PARALLEL_PROCESSES = 4 

    all_results = []
    import tqdm
    import json
    import argparse

    parser = argparse.ArgumentParser("compare_activation_patterns.py")
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    checkpoint_directories = checkpoint_directories_by_architecture["mlp"]

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
            with suppress_prints(suppress=True):
                if args.parallel_processes > 1:
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel_processes)
                else:
                    import types
                    executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
                model_pair_results = [future.result() for future in [
                    executor.submit(compute_activation_pattern_results,
                        **dict(checkpoint_path_1=checkpoint_path(model1_index, epoch), checkpoint_path_2=checkpoint_path(model2_index, epoch), data_info=data_info)
                    ) for epoch in EPOCH_RANGE]
                ]
                if args.parallel_processes > 1:
                    executor.shutdown()
            for activation_pattern_results, epoch in zip(model_pair_results, EPOCH_RANGE):
                all_results.append({
                    "run_key": run_key,
                    "model1_index": model1_index,
                    "model2_index": model2_index,
                    "epoch": epoch,
                    "activation_pattern_results": activation_pattern_results
                })
            with open(args.output_file, "w") as f:
                json.dump(all_results, f)

if __name__ == "__main__":
    main()
