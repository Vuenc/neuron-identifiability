import argparse
import copy
from typing import Any, Dict, List, Tuple
import torch
import torch.utils.data
import torchvision
import tqdm
import itertools
from models.models_mlp import MLP, WMLP
import pickle
import multiprocessing
import datasets

def default_mask_params():
    linear_mask_params_0 = {'mask_constant' : 1, 'mask_type' : 'random_subsets',
                            'do_normal_mask' : True, 'num_fixed': 64}
    linear_mask_params_1 = {'mask_constant' : 1, 'mask_type' : 'random_subsets',
                            'do_normal_mask' : True, 'num_fixed': 64}
    linear_mask_params_2 = {'mask_constant' : 1/2, 'mask_type' : 'random_subsets',
                            'do_normal_mask' : True, 'num_fixed': 64}
    linear_mask_params_3 = {'mask_constant' : 1/4, 'mask_type' : 'random_subsets',
                            'do_normal_mask' : True, 'num_fixed': 256}
    mask_params = {
        0 : linear_mask_params_0,
        1 : linear_mask_params_1,
        2: linear_mask_params_2,
        3: linear_mask_params_3
    }
    return mask_params

def default_mlp_hyperparams():
    return dict(
        in_dim=784,
        hidden_dim=512,
        out_dim=10,
        num_layers=4
    )

def default_wmlp_hyperparams():
    return dict(
        **default_mlp_hyperparams(),
        mask_params=default_mask_params()
    )


device = torch.device("cuda")

def interpolate_models(model1, model2, target_model, alpha):
    for param1, param2, param_target in zip(model1.parameters(), model2.parameters(), target_model.parameters()):
        param_target[:] = param1 * (1-alpha) + param2 * alpha
    return target_model

def parameters_to_vector(model, only_unmasked_parameters=False):
    if only_unmasked_parameters and isinstance(model, WMLP):
        mask_by_ptr = {lin.weight.data_ptr(): lin.mask for lin in model.lins}
        return torch.concat([
            param_tensor[mask_by_ptr[param_tensor.data_ptr()] > 0].reshape(-1)
            if param_tensor.data_ptr() in mask_by_ptr else param_tensor.reshape(-1)
            for param_tensor in model.parameters()])
    else:
        return torch.concat([param_tensor.reshape(-1) for param_tensor in model.parameters()])

def sample_models(num_models: int, model_class: type, model_args: Dict) -> Tuple[Any, List[Any]]:
    [tmp_model, *models] = [
        model_class(**(model_args)).to(device)
        for _ in range(num_models+1)
    ]
    return tmp_model, models

# def compute_losses_on_line(model1, model2, tmp_model, num_steps):
#     loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
#     # eval the losses at 100 equally spaced points interpolating between models[0] and models[1]
#     with torch.no_grad():
#         mean_loss_by_model = [
#             (
#                 torch.stack([
#                     loss_fn(model.forward(data.to(device)), target.to(device))
#                     for data, target in tqdm.tqdm(train_loader, leave=False)
#                 ]).sum() / len(train_dataset)
#             )
#             for model in tqdm.tqdm(
#                 interpolate_models(model1, model2 , tmp_model, alpha)
#                 for alpha in torch.linspace(0, 1, num_steps)
#             )
#         ]
#     return torch.stack(mean_loss_by_model).tolist()

# IMPORTANT! Check: Can we even trivially interpolate between WMLPs? What about the masks? Do they have to be the same? Check with Derek's code.
def compute_losses_on_line_batched(model1, model2, num_steps, train_dataset, train_loader, batch_size_params=100):
    with torch.no_grad():
        mean_loss_by_model = []
        for batch_step_from in range(0, num_steps, batch_size_params):
            model1_params = dict(model1.named_parameters())
            model2_params = dict(model2.named_parameters())
            param_dicts = [
                {param_key: param1 * (1-alpha) + model2_params[param_key] * alpha for param_key, param1 in model1_params.items()}
                for alpha in torch.linspace(0, 1, num_steps)[batch_step_from:batch_step_from+batch_size_params]
            ]

            def inference_with_params(params, inputs):
                return torch.func.functional_call(model1, params, (inputs,), strict=False)

            # Vectorise it over the param tree (dim 0) but NOT over inputs (None) ------
            # You could also vmap over inputs if you want both batch and param sweep.
            batched_f = torch.vmap(inference_with_params, in_dims=(0, None))

            # Each tensor gains a leading dimension (N, …)
            batched_params = torch.utils._pytree.tree_map(lambda *t: torch.stack(t, 0), *param_dicts)
            
            actual_num_steps = min(num_steps, batch_size_params)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            mean_loss_by_model.extend((
                torch.concat([
                    loss_fn(
                    # output will be of shape (num_steps, batch_size, C)
                    batched_f(batched_params, data.to(device))
                    # combine first two dimensions for the loss: now (num_steps * batch_size, C)
                    .reshape(actual_num_steps * data.shape[0], -1),
                    target.to(device).repeat(actual_num_steps)
                    ).reshape(actual_num_steps, data.shape[0], -1)
                    for data, target in tqdm.tqdm(train_loader, leave=False)
                ], dim=1).sum(dim=1) / len(train_dataset)
            ).squeeze(1).tolist())
    return mean_loss_by_model

# def interpolate_one_pair(i, wmlps=None, mlps=None, pickle_paths_wmlps=None, pickle_paths_mlps=None, save_result=True):
    # if wmlps is not None:
    #     wmlp1, wmlp2 = wmlps
    # elif pickle_paths_wmlps is not None:
    #     with open(pickle_paths_wmlps[0], "rb") as f1, open(pickle_paths_wmlps[1], "rb") as f2:
    #         wmlp1 = pickle.load(f1)["model"]
    #         wmlp2 = pickle.load(f2)["model"]
    #     # tmp_wmlp = copy.deepcopy(wmlp1)
    # else:
    #     _, [wmlp1, wmlp2] = sample_models(
    #         num_models=2, model_class=WMLP, model_args=default_wmlp_hyperparams())

    # TODO this is suboptimal; check if we can switch to always using this mode without activating it, or if it affects training
    # wmlp1.disable_sparse_linear_data_replacement()
        
    # if mlps is not None:
    #     mlp1, mlp2 = mlps
    # if pickle_paths_mlps is not None:
    #     with open(pickle_paths_mlps[0], "rb") as f1, open(pickle_paths_mlps[1], "rb") as f2:
    #         mlp1 = pickle.load(f1)["model"]
    #         mlp2 = pickle.load(f2)["model"]
    #     tmp_mlp = copy.deepcopy(mlp1)
    # else:
    #     tmp_mlp, [mlp1, mlp2] = sample_models(
    #         num_models=2, model_class=MLP, model_args=default_mlp_hyperparams())


    # mean_losses_wmlp = compute_losses_on_line_batched(wmlp1, wmlp2, num_steps=15)
    mean_losses_mlp = compute_losses_on_line_batched(mlp1, mlp2, num_steps=15)

    # if save_result:
    #     BASEPATH = "experiments/interpolations-trained"
    #     with open(f'{BASEPATH}/mlp-wmlp-{i}.pickle', 'wb') as handle:
    #         pickle.dump(dict(wmlp1=wmlp1, wmlp2=wmlp2, mean_losses_wmlp=mean_losses_wmlp,
    #                             #mlp1=mlp1, mlp2=mlp2, mean_losses_mlp=mean_losses_mlp
    #                             ),
    #         handle, protocol=pickle.HIGHEST_PROTOCOL)
    return mean_losses_mlp

def convexity_errors(mean_losses: List[float]):
    """
    Compute an ad-hoc measure of non-convexity: mean_losses should be a list of values on the y axis that are evenly spaced on the x axis.
    For every point p, we find the pair of points p1, p2 such that p1[1] < p[1] < p2[1] and the line segment [p1, p2] has maximum distance from p at x=p[0].
    The convexity error is 0 if that line segment is above p, and proportional to its distance otherwise.

    If max(convexity_errors(values)) == 0.0, the list values does not exhibit non-convexity.
    """

    import numpy as np
    def convexity_error(x, y, x1, y1, x2, y2):
        if not x1 <= x < x2:
            return 0.
        # if y > y1 + (y2-y1)/(x2-x1)*(x-x1):
        return (y - (y1 + (y2-y1)/(x2-x1)*(x-x1))) # / (max(y, y1, y2) - min(y, y1, y2))

    loss_points = np.stack([np.linspace(0, 1, len(mean_losses)), mean_losses], axis=1)
    convexity_errors = [
        (max(
            convexity_error(*p, *p1, *p2)
            for p1 in loss_points[:i]
            for p2 in loss_points[i+1:]
        ) / (max(mean_losses) - min(mean_losses)))
        if i > 0 and i < len(loss_points) - 1 else 0.
        for i, p in enumerate(loss_points)
    ]
    return convexity_errors

if __name__ == "__main__":
    # with multiprocessing.Pool(64) as pool:
    #     BASEPATH = "experiments/interpolations-trained-trajectories"
    #     pool.starmap(interpolate_one_pair, [(i, [f"{BASEPATH}/trained_wmlp_{j}.pickle" for j in [2*i, 2*i+1]], [f"{BASEPATH}/trained_mlp_{j}.pickle" for j in [2*i, 2*i+1]]) for i in range(64)])

    parser = argparse.ArgumentParser(description='Interpolations for LMC of trained networks')
    parser.add_argument('--symmetry',  required=True, type=int,
                    metavar='s', help='Symmetry: 0 (Standard) 1 (W) 2 (Sigma)')
    parser.add_argument('--output_path',  required=True,
                        metavar='o', help='Path of directory where to read trained models from, and where to write output files')
    parser.add_argument("--run_id_range", required=True, type=int, nargs="+", help="Range of runs to include. Pass one or more arguments that will be passed to Python's range(...) function (e.g. '-run_id_range 16 32' for models 16 to 31)")
    parser.add_argument("--epoch_range", required=True, type=int, nargs="+", help="Range of epochs to include. Pass one or more arguments that will be passed to Python's range(...) function (e.g. '-epochs_range 0 151 5' for epochs 0, 5, 10, ..., 150)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(datasets.dataset_factories.keys()))
    parser.add_argument("--interpolation_steps", type=int, required=True)

    args = parser.parse_args()
    assert args.symmetry in [0, 1]

    train_dataset = datasets.dataset_factories[args.dataset](device)
    train_loader = torch.utils.data.DataLoader(
        [(t.to(device), l) for t, l in train_dataset], 
        batch_size=5096*16,
        shuffle=False,
        # pin_memory = True
    )

    run_id_range = list(range(*args.run_id_range))
    epoch_range = [*range(*args.epoch_range)]
    models_cache = {i: {} for i in run_id_range}
    for run_id in tqdm.tqdm(run_id_range, "Loading models to cache..."):
        for epoch in epoch_range:
            # TODO
            with open(f"{args.output_path}/trained_{'wmlp' if args.symmetry == 1 else 'mlp'}_{run_id}_epoch-{epoch}.pickle", "rb") as f:
                models_cache[run_id][epoch] = pickle.load(f)["model"]

    for i, run_id_1 in enumerate(tqdm.tqdm(run_id_range, desc="model 1")):
        for run_id_2 in tqdm.tqdm(run_id_range[i+1:], desc="model 2", leave=False):
            losses_by_epoch = [
                compute_losses_on_line_batched(models_cache[run_id_1][epoch], models_cache[run_id_2][epoch], args.interpolation_steps, train_dataset, train_loader)
                for epoch in tqdm.tqdm(epoch_range, desc="epoch", leave=False)
            ]
            with open(f'{args.output_path}/{'wmlp' if args.symmetry == 1 else 'mlp'}-trajectory-interpolations-{run_id_1}-{run_id_2}.pickle', 'wb') as handle:
                pickle.dump(dict(mean_losses_by_epoch=losses_by_epoch,
                                 epoch_range=epoch_range
                    # wmlp1=wmlp1, wmlp2=wmlp2, mean_losses_wmlp=mean_losses_wmlp,
                    # mlp1=mlp1, mlp2=mlp2, mean_losses_mlp=mean_losses_mlp
                                    ),
                handle, protocol=pickle.HIGHEST_PROTOCOL)
