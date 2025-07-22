from typing import Any, Dict, List, Tuple
import torch
import torch.utils.data
import torchvision
import tqdm
import itertools

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

normalize_transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform,
    ])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=5096, 
    shuffle=False,
    pin_memory = True
)

def interpolate_models(model1, model2, target_model, alpha):
    for param1, param2, param_target in zip(model1.parameters(), model2.parameters(), target_model.parameters()):
        param_target[:] = param1 * (1-alpha) + param2 * alpha
    return target_model

def sample_models(num_models: int, model_class: type, model_args: Dict) -> Tuple[Any, List[Any]]:
    [tmp_model, *models] = [
        model_class(**(model_args)).to(device)
        for _ in range(num_models+1)
    ]
    return tmp_model, models

def compute_losses_on_line(model1, model2, tmp_model, num_steps):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    # eval the losses at 100 equally spaced points interpolating between models[0] and models[1]
    with torch.no_grad():
        mean_loss_by_model = [
            (
                torch.stack([
                    loss_fn(model.forward(data.to(device)), target.to(device))
                    for data, target in tqdm.tqdm(train_loader, leave=False)
                ]).sum() / len(train_dataset)
            )
            for model in tqdm.tqdm(
                interpolate_models(model1, model2 , tmp_model, alpha)
                for alpha in torch.linspace(0, 1, num_steps)
            )
        ]
    return torch.stack(mean_loss_by_model).tolist()

if __name__ == "__main__":
    from lmc.models.models_mlp import MLP, WMLP
    import pickle

    for i in tqdm.tqdm(range(29, 75)):
        tmp_wmlp, [wmlp1, wmlp2] = sample_models(
            num_models=2, model_class=WMLP, model_args=default_wmlp_hyperparams())
        tmp_mlp, [mlp1, mlp2] = sample_models(
            num_models=2, model_class=MLP, model_args=default_mlp_hyperparams())

        mean_losses_wmlp = compute_losses_on_line(wmlp1, wmlp2, tmp_wmlp, num_steps=100)
        mean_losses_mlp = compute_losses_on_line(mlp1, mlp2, tmp_mlp, num_steps=100)

        with open(f'mlp-wmlp-{i}.pickle', 'wb') as handle:
            pickle.dump(dict(wmlp1=wmlp1, wmlp2=wmlp2, mlp1=mlp1, mlp2=mlp2, mean_losses_wmlp=mean_losses_wmlp, mean_losses_mlp=mean_losses_mlp),
            handle, protocol=pickle.HIGHEST_PROTOCOL)