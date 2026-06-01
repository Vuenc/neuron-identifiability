import concurrent.futures
from typing import Dict
import hydra
import torch
import pathlib
from src.models.mlp import MLP
from src.models.resnet import ResNet
from src.utils.interpolation import interpolate_models
import train
import src.utils.rebasin.activation_matching
from src.utils.rebasin import ActivationCorrelationMode
import numpy as np
from contextlib import contextmanager
from checkpoint_directories import checkpoint_directories_by_architecture

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

def align_activation_matching(model1, model2, cfg, data_info, device="cuda:0"):
    if cfg.model.name == "mlp_mnist":
        permutation_spec = src.utils.rebasin.mlp_permutation_spec(cfg.model.num_layers-1, norm=True, bias=True)
        correlation_modes = [
            # ActivationCorrelationMode.PearsonCorrelationWithZeroForConstant, ActivationCorrelationMode.DotProduct
            ActivationCorrelationMode.PearsonUncorrelatednessWithOneForConstant
        ]
    elif cfg.model.name == "resnet_cifar":
        permutation_spec = src.utils.rebasin.resnet20_permutation_spec()
        # ReLU resnets tend to have all-zero recorded activations for one channel, which crashes usual Pearson correlation
        correlation_modes = [
            # ActivationCorrelationMode.PearsonCorrelationWithZeroForConstant
            ActivationCorrelationMode.PearsonUncorrelatednessWithOneForConstant
        ]
        # We shorten the train loader: not enough memory for everything
        data_info["train_loader"] = [d for _, d in zip(range(10), data_info["train_loader"])]
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")
    activation_matching_results: Dict[src.utils.rebasin.activation_matching.ActivationRecordingPoint[str, ActivationCorrelationMode], Dict[str, src.utils.rebasin.activation_matching.MatchingResult]] = src.utils.rebasin.activation_matching.activation_matching(
        permutation_spec, model1, model2, data_info["train_loader"], device=device, correlation_modes=correlation_modes
    )
    matching_results_by_permutation_name = activation_matching_results[('post_activation_function', correlation_modes[0])]
    perm = {
        perm_name: matching_result.optimal_permutation.to("cuda:0")
        for perm_name, matching_result in matching_results_by_permutation_name.items()
    }
    permuted_params_2 = src.utils.rebasin.common.apply_permutation(
        permutation_spec,
        perm,
        model2.state_dict()
    )
    return permuted_params_2

def align_weight_matching(model1, model2, cfg, device="cuda:0"):
    if cfg.model.name == "mlp_mnist":
        permutation_spec = src.utils.rebasin.mlp_permutation_spec(3, norm=True, bias=True)
    elif cfg.model.name == "resnet_cifar":
        permutation_spec = src.utils.rebasin.resnet20_permutation_spec()
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")
    weight_matching_result_permutation = src.utils.rebasin.weight_matching.weight_matching(permutation_spec, model1.state_dict(), model2.state_dict(), max_iter=100, restarts=10)
    permuted_params_2 = src.utils.rebasin.common.apply_permutation(
        permutation_spec,
        weight_matching_result_permutation,
        model2.state_dict()
    )
    return permuted_params_2


def compute_lmc_results(checkpoint_path_1, checkpoint_path_2, num_interpolation_steps=10, data_info=None, device="cuda:0") -> Dict[str, Dict]:
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path_1).parent)):
        cfg = hydra.compose(config_name="config")
    cfg.dataset.batch_size = 512
    if data_info is None:
        data_info = train.setup_data_loaders(cfg)

    model1, model2 = [train.create_model(cfg, mask_seed=-1).to(device) for _ in range(2)]
    model1_state_dict = torch.load(checkpoint_path_1, map_location=device)["model_state_dict"]
    model2_state_dict = torch.load(checkpoint_path_2, map_location=device)["model_state_dict"]
    model1.load_state_dict(model1_state_dict)
    model2.load_state_dict(model2_state_dict)
    model1.eval()
    model2.eval()

    # Unaligned LMC
    interpolation_results_unaligned = interpolate_models(
        model1, model1_state_dict, model2_state_dict,
        data_info['train_loader'], data_info['val_loader'], data_info['test_loader'],
        steps=num_interpolation_steps,
        device=cfg.device,
    )

    # Transfer asymmetric models to non-asymmetric versions
    if not (isinstance(model1, MLP) or isinstance(model1, ResNet)):
        model1 = model1.convert_to_non_asymmetric_model()
        model2 = model2.convert_to_non_asymmetric_model()
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()

    # # Align with weight matching
    # model2_state_dict_weight_aligned = align_weight_matching(model1, model2, cfg, device=device)
    # interpolation_results_weight_aligned = interpolate_models(
    #     model1, model1_state_dict, model2_state_dict_weight_aligned,
    #     data_info['train_loader'], data_info['val_loader'], data_info['test_loader'],
    #     steps=num_interpolation_steps,
    #     device=cfg.device,
    # )

    # Align with activation matching
    model2_state_dict_activation_aligned = align_activation_matching(model1, model2, cfg, data_info, device=device)
    interpolation_results_activation_aligned = interpolate_models(
        model1, model1_state_dict, model2_state_dict_activation_aligned,
        data_info['train_loader'], data_info['val_loader'], data_info['test_loader'],
        steps=num_interpolation_steps,
        device=cfg.device,
    )

    return dict(
        interpolation_results_unaligned=interpolation_results_unaligned,
        interpolation_results_activation_aligned=interpolation_results_activation_aligned,
        # interpolation_results_weight_aligned=interpolation_results_weight_aligned
    )

def main():
    # MODEL_1_RANGE = list(range(1, 17, 2))
    MODEL_1_RANGE = list(range(1, 5, 2))
    EPOCH = 50

    all_results = []
    import tqdm
    import json
    import argparse

    parser = argparse.ArgumentParser("measure_lmc_unaligned_aligned.py")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--checkpoint-directory", type=str)
    parser.add_argument("--run-key")
    args = parser.parse_args()

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
    data_info = train.setup_data_loaders(_cfg)

    checkpoint_path = lambda model_index, epoch: f"{checkpoint_directories[run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"

    for run_key in (tqdm_run := tqdm.tqdm(checkpoint_directories.keys())):
        tqdm_run.set_description(f"Run: {run_key}")
        for model1_index in (tqdm_model_pair := tqdm.tqdm(MODEL_1_RANGE, desc=run_key, leave=False)):
            model2_index = model1_index + 1
            tqdm_model_pair.set_description(desc=f"Model pair: (Model {model1_index}, Model {model2_index})")
            with suppress_prints(suppress=False):
                checkpoint_path_1 = checkpoint_path(model1_index, EPOCH)
                checkpoint_path_2 = checkpoint_path(model2_index, EPOCH)
                model_pair_results = compute_lmc_results(checkpoint_path_1=checkpoint_path(model1_index, EPOCH), checkpoint_path_2=checkpoint_path(model2_index, EPOCH), data_info=data_info, num_interpolation_steps=8)
            all_results.append({
                "run_key": run_key,
                "model1_index": model1_index,
                "model2_index": model2_index,
                "checkpoint_path_1": checkpoint_path_1,
                "checkpoint_path_2": checkpoint_path_2,
                "epoch": EPOCH,
                **model_pair_results
            })
            with open(args.output_file, "w") as f:
                json.dump(all_results, f)

if __name__ == "__main__":
    main()
