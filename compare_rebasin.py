

from typing import Literal
import src.utils.rebasin.weight_matching
import torch
import train
import hydra
import pathlib
import copy
import src.utils.interpolation
import tqdm
import concurrent.futures

def load_model_with_symmetry_breaking_removed(cfg, path, example_input: torch.Tensor):
    state = torch.load(path)
    state_dict = state["model_state_dict"]
    w_asym_model = None
    if cfg.model.symmetry == 1:
        w_asym_model = train.create_model(cfg, mask_seed=-1)
        print(f"Loaded w-asym. model: type '{type(w_asym_model).__name__}'")
        w_asym_model.load_state_dict(state_dict)

        # Multiply in mask constants, remove mask parameters (depends on architecture)
        if cfg.model.name == "mlp_mnist":
            for i in range(cfg.model.num_layers):
                state_dict[f"lins.{i}.weight"] = state_dict[f"lins.{i}.weight"] * state_dict[f"lins.{i}.mask"] + state_dict[f"lins.{i}.normal_mask"] * (1 - state_dict[f"lins.{i}.mask"]) * cfg.model.mask_params.default.mask_constant
                del state_dict[f"lins.{i}.mask"]
                del state_dict[f"lins.{i}.normal_mask"]
        elif cfg.model.name == "resnet_cifar":
            # for i in range(cfg.model.num_layers):
            # module_names = ["conv1", "linear", *(f"layer{layer_id+1}.{block_id}.{module_type}" for layer_id in range(3) for block_id in range(3) for module_type in ["conv1", "conv2", "shortcut0"])]
            module_names = [key.removesuffix(".mask") for key in state_dict.keys() if key.endswith(".mask")]
            for module_name in module_names:
                state_dict[f"{module_name}.weight"] = state_dict[f"{module_name}.weight"] * state_dict[f"{module_name}.mask"] + state_dict[f"{module_name}.normal_mask"] * (1 - state_dict[f"{module_name}.mask"]) * cfg.model.mask_params.default.mask_constant
                del state_dict[f"{module_name}.mask"]
                del state_dict[f"{module_name}.normal_mask"]
        else:
            raise ValueError(f"Unsupported model name: {cfg.model.name}")

        print("Baked in masks, remaining keys:")
        print(state_dict.keys())
        cfg = copy.deepcopy(cfg)
        cfg.model.symmetry = 0

    model = train.create_model(cfg, mask_seed=-1)
    print(f"Loaded non-asym. model: type '{type(model).__name__}'")
    model.load_state_dict(state_dict)
    if w_asym_model is not None:
        assert torch.allclose(w_asym_model.forward(example_input.cpu()), model.forward(example_input.cpu())) # type: ignore
    return model

# def load_wmlp(cfg, path, example_input):
    
#     assert cfg.model.symmetry == 1
#     wmlp = train.create_model(cfg, mask_seed=-1)
#     wmlp.load_state_dict(state_dict)
#     print(f"Loaded {type(wmlp).__name__} (expected: WMLP)")

#     print("Warning: this code assumes no per-layer specified mask constants and only uses the global cfg.model.mask_params.default.mask_constant")

#     # print(state_dict.keys())
    
#     cfg = copy.deepcopy(cfg)
#     cfg.model.symmetry = 0
#     mlp = train.create_model(cfg, mask_seed=-1)
#     mlp.load_state_dict(state_dict)

#     print(f"Loaded {type(mlp).__name__} (expected: MLP)")

#     x = torch.randn(1, 784)
    
#     return mlp

# def load_mlp(cfg, path):
#     state = torch.load(path)
#     state_dict = state["model_state_dict"]
#     print(state_dict.keys())

#     with hydra.initialize(version_base=None, config_path=str(pathlib.Path(path).parent)):
#         cfg = hydra.compose(config_name="config")

#     assert cfg.model.symmetry == 0
#     mlp = train.create_model(cfg, mask_seed=-1)
#     mlp.load_state_dict(state_dict)
#     print(f"Loaded {type(mlp).__name__} (expected: MLP)")

#     return mlp


def aligned_lmc(path1, path2, model_index_1, model_index_2, max_iter=1000, restarts=10, steps=10):
    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(path1).parent)):
        cfg = hydra.compose(config_name="config")
    # cfg.dataset.batch_size = 1024
    cfg.dataset.enable_ffcv = False
    data_info = train.setup_data_loaders(cfg)
    example_input = next(iter(data_info["train_loader"]))[0][:1].to("cuda:0")

    if cfg.model.symmetry in [0, 1]:
        model1, model2 = load_model_with_symmetry_breaking_removed(cfg, path1, example_input), load_model_with_symmetry_breaking_removed(cfg, path2, example_input)
    else:
        raise ValueError("cfg.model.symmetry must be 0 or 1!")
    
    # print(({key: model1.state_dict()[key].shape for key in model1.state_dict().keys()}.items()))
    # print(sorted({key: torch.linalg.norm((model1.state_dict()[key]-model2.state_dict()[key]).reshape(-1)) for key in model1.state_dict()}.items(), key=lambda p: -p[1]))
    # print(model1.state_dict()["layer3.0.conv2.weight"].shape)
    # print(model1.state_dict()["layer3.0.conv2.weight"].abs().mean())
    # print((model1.state_dict()["layer3.0.conv2.weight"]==0).float().mean())
    # print(sum(w.std() for _, w in model1.state_dict().items()))
    # import sys; sys.exit();

    permutation_spec = (
        src.utils.rebasin.weight_matching.mlp_permutation_spec(num_hidden_layers=3, norm=True) if cfg.model.name == "mlp_mnist"
        else src.utils.rebasin.weight_matching.resnet20_permutation_spec() if cfg.model.name == "resnet_cifar"
        else None
    )
    if permutation_spec is None:
        raise ValueError(f"Unsupported model name in config: {cfg.model.name}")
    permutation = src.utils.rebasin.weight_matching.weight_matching(
        permutation_spec, model1.state_dict(), model2.state_dict(),
        max_iter=max_iter, restarts=restarts)

    model2_permuted = copy.deepcopy(model2)
    model2_permuted.load_state_dict(src.utils.rebasin.weight_matching.apply_permutation(permutation_spec, permutation, model2_permuted.state_dict()))
    model1.to("cuda:0")
    model2.to("cuda:0")
    model2_permuted.to("cuda:0")

    if not torch.allclose(o1 := model2.forward(example_input), o2 := model2_permuted.forward(example_input)): # type: ignore
        print("WARNING! model2 and model2_permuted not close: mean difference", (o1 - o2).mean().item())

    with hydra.initialize(version_base=None, config_path=str(pathlib.Path(path1).parent)):
        cfg = hydra.compose(config_name="config")

    interpolation_output_unaligned = src.utils.interpolation.interpolate_models(model1, model1.state_dict(), model2.state_dict(), data_info["train_loader"], data_info["val_loader"], data_info["test_loader"], steps=steps)
    interpolation_output_aligned = src.utils.interpolation.interpolate_models(model1, model1.state_dict(), model2_permuted.state_dict(), data_info["train_loader"], data_info["val_loader"], data_info["test_loader"], steps=steps)
    return dict(
        model_index_1=model_index_1, model_index_2=model_index_2,
        path1=path1, path2=path2,
        interpolation_output_aligned=interpolation_output_aligned, interpolation_output_unaligned=interpolation_output_unaligned
    )

def main(run_directory, num_models, epoch, max_iter, restarts, interpolation_steps, max_parallel_processes):
    paths = [f"{run_directory}/checkpoint_epoch_{epoch}_model_{j+1}.pt" for j in range(num_models)]

    if max_parallel_processes > 1 and num_models > 2:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_processes) as executor:
            model_results = list(tqdm.tqdm((future.result() for future in concurrent.futures.as_completed(
                [executor.submit(aligned_lmc, *(paths[i], paths[i+1], i, i+1,  max_iter, restarts, interpolation_steps)) for i in range(0, num_models, 2)]
            )),  total=num_models//2))
    else:
        model_results = list(tqdm.tqdm((aligned_lmc(paths[i], paths[i+1], i, i+1,  max_iter, restarts, interpolation_steps) for i in range(0, num_models, 2)), total=num_models//2))
    return model_results

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=str)
    parser.add_argument("--num-models", required=True, type=int)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--rebasin-max-iter", default=1000, type=int)
    parser.add_argument("--rebasin-restarts", default=10, type=int)
    parser.add_argument("--interpolation-steps", default=10, type=int)
    parser.add_argument("--max-parallel-processes", default=5, type=int)
    args = parser.parse_args()

    model_results = main(args.run_directory, args.num_models, args.epoch, args.rebasin_max_iter, args.rebasin_restarts, args.interpolation_steps, args.max_parallel_processes)
    with open(f"{args.run_directory}/rebasin_lmc_results_epoch_{args.epoch}.json", "w") as f:
        json.dump(model_results, f)
