import concurrent.futures
import hydra
import torch
import pathlib
import train
import src.utils.rebasin.subspace_coherence
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

# checkpoint_directories_mlp_nonorm = {
#     "mlp_symmetry0": "outputs/2026-01-15/17-07-22_mlp_mnist_sym-0__db8ehws3__y5oujjmq",
# }

checkpoint_directories = checkpoint_directories_by_architecture["mlp"]

def compute_subspace_coherence_results(checkpoint_path, device="cuda:0", data_info=None):
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(checkpoint_path).parent)):
        cfg = hydra.compose(config_name="config")
    cfg.dataset.batch_size = 500
    if data_info is None:
        data_info = train.setup_data_loaders(cfg)

    model = train.create_model(cfg, mask_seed=-1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])

    permutation_spec = src.utils.rebasin.mlp_permutation_spec(3, norm=True, bias=True)
    coherence_results = src.utils.rebasin.subspace_coherence.compute_representation_subspace_coherences(
        permutation_spec, model, data_info["train_loader"], target_explained_variance_ratio=0.9, device=device
    )
    results = []
    with torch.no_grad():
        for matching_mode, results_per_layer in coherence_results.items():
            for layer_name, coherence_result in results_per_layer.items():
                results.append({
                    "matching_mode": matching_mode,
                    "layer": layer_name,
                    "subspace_coherence": coherence_result.subspace_coherence,
                    "subspace_dimension": coherence_result.subspace_dimension,
                    "full_dimension": coherence_result.full_dimension,
                    "explained_variance_ratio": coherence_result.explained_variance_ratio,
                })
    return results

MODEL_RANGE = list(range(1, 17, 1))
EPOCH_RANGE = list(range(0, 101, 5))
# EPOCH_RANGE = list(range(0, 11, 1))
MAX_PARALLEL_PROCESSES = 22

all_results = []
import tqdm
import json

# Assuming all are using the same dataset
with hydra.initialize(version_base=None, config_path=str(pathlib.Path(list(checkpoint_directories.values())[0]))):
    _cfg = hydra.compose(config_name="config")
_cfg.dataset.batch_size = 2**15
data_info = train.setup_data_loaders(_cfg)

checkpoint_path = lambda model_index, epoch: f"{checkpoint_directories[run_key]}/checkpoint_epoch_{epoch}_model_{model_index}.pt"

for run_key in (tqdm_run := tqdm.tqdm(checkpoint_directories.keys())):
    tqdm_run.set_description(f"Run: {run_key}")
    for model_index in (tqdm_model := tqdm.tqdm(MODEL_RANGE, desc=run_key, leave=False)):
        tqdm_model.set_description(f"Model {model_index}")
        with suppress_prints(suppress=True):
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES) as executor:
                # import types
                # executor = types.SimpleNamespace(submit=lambda f, **kwargs: types.SimpleNamespace(result = lambda: f(**kwargs)))
                model_results = [future.result() for future in [
                    executor.submit(compute_subspace_coherence_results,
                        **dict(checkpoint_path=checkpoint_path(model_index, epoch), data_info=data_info)
                    ) for epoch in EPOCH_RANGE]
                ]
        for coherence_results, epoch in zip(model_results, EPOCH_RANGE):
            all_results.append({
                "run_key": run_key,
                "model_index": model_index,
                "epoch": epoch,
                "coherence_results": coherence_results
            })
        with open("outputs/subspace-coherence-results-mlp.json", "w") as f:
            json.dump(all_results, f)
