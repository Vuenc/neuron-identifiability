from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PATTERN_ORDER = ["00", "01", "10", "11"]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class TrainConfig:
    copies: int = 10
    width: int = 128
    active_prob: float = 0.15
    n_train: int = 4096
    n_test: int = 4096
    batch_size: int = 512
    epochs: int = 300
    lr: float = 1e-2
    weight_decay: float = 1e-2
    bias_weight_decay: float = 0.0
    num_runs: int = 6
    outer_seeds: List[int] | None = None
    device: str = "cpu"
    torch_threads: int = 1
    interpolation_steps: int = 21
    optimizer_name: str = "adamw"
    train_hidden_bias: bool = True


def ambient_dim(copies: int) -> int:
    return 2 * copies


def pattern_histogram(mask: torch.Tensor) -> Dict[str, int]:
    hist = {pattern: 0 for pattern in PATTERN_ORDER}
    for row in mask:
        pattern = "".join(str(int(v.item())) for v in row[:2])
        hist[pattern] += 1
    return hist


def make_embedding(mode: str, copies: int) -> torch.Tensor:
    dim = ambient_dim(copies)
    basis = torch.zeros(dim, 2)
    if mode == "aligned":
        basis[0, 0] = 1.0
        basis[1, 1] = 1.0
        return basis
    if mode == "dispersed":
        scale = 1.0 / np.sqrt(copies)
        for t in range(copies):
            basis[2 * t + 0, 0] = scale
            basis[2 * t + 1, 1] = scale
        return basis
    raise ValueError(f"Unknown embedding mode: {mode}")


def teacher_output(latent: torch.Tensor) -> torch.Tensor:
    return F.relu(latent[:, 0] + latent[:, 1]).unsqueeze(1)


def make_dataset(cfg: TrainConfig, embedding_mode: str, dataset_seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(dataset_seed)
    train_latent = torch.randn(cfg.n_train, 2, generator=generator)
    test_latent = torch.randn(cfg.n_test, 2, generator=generator)
    basis = make_embedding(embedding_mode, cfg.copies)
    train_x = train_latent @ basis.T
    test_x = test_latent @ basis.T
    train_y = teacher_output(train_latent)
    test_y = teacher_output(test_latent)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "teacher_test": test_y,
        "basis": basis,
    }


class ScalarMaskedReLUNet(nn.Module):
    def __init__(self, mask: torch.Tensor, train_hidden_bias: bool):
        super().__init__()
        self.register_buffer("mask", mask.float(), persistent=False)
        self.hidden_weight = nn.Parameter(torch.empty(mask.shape[0], mask.shape[1]))
        if train_hidden_bias:
            self.hidden_bias = nn.Parameter(torch.zeros(mask.shape[0]))
        else:
            self.register_buffer("hidden_bias", torch.zeros(mask.shape[0]), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.hidden_weight, mean=0.0, std=0.05)
        if isinstance(self.hidden_bias, nn.Parameter):
            nn.init.zeros_(self.hidden_bias)

    def hidden_preact(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.hidden_weight * self.mask).T + self.hidden_bias

    def hidden_act(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.hidden_preact(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden_act(x).sum(dim=1, keepdim=True)


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    perm = torch.randperm(x.shape[0], device=x.device)
    for start in range(0, x.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        yield x[idx], y[idx]


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x)
        return F.mse_loss(pred, y).item()


def train_single_model(mask: torch.Tensor, cfg: TrainConfig, data: Dict[str, torch.Tensor], seed: int) -> ScalarMaskedReLUNet:
    set_seed(seed)
    model = ScalarMaskedReLUNet(mask=mask.to(cfg.device), train_hidden_bias=cfg.train_hidden_bias).to(cfg.device)
    param_groups = [{"params": [model.hidden_weight], "weight_decay": cfg.weight_decay}]
    if isinstance(model.hidden_bias, nn.Parameter):
        param_groups.append({"params": [model.hidden_bias], "weight_decay": cfg.bias_weight_decay})
    if cfg.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    elif cfg.optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")
    train_x = data["train_x"].to(cfg.device)
    train_y = data["train_y"].to(cfg.device)
    model.train()
    for _ in range(cfg.epochs):
        for batch_x, batch_y in iterate_minibatches(train_x, train_y, cfg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


def centered_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return a.T @ b


def hidden_matching_metrics(model_a: ScalarMaskedReLUNet, model_b: ScalarMaskedReLUNet, x: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        hidden_a = model_a.hidden_act(x).cpu()
        hidden_b = model_b.hidden_act(x).cpu()
    sim = centered_correlation(hidden_a, hidden_b)
    cost = (-sim).numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    identity_obj = sim.diag().mean().item()
    optimal_obj = sim[row_ind, col_ind].mean().item()
    return {
        "identity_obj": identity_obj,
        "optimal_obj": optimal_obj,
        "gap": optimal_obj - identity_obj,
    }


def winner_metrics(model: ScalarMaskedReLUNet, x: torch.Tensor, teacher: torch.Tensor) -> Dict[str, float | int]:
    with torch.no_grad():
        hidden = model.hidden_act(x).cpu()
    sim = centered_correlation(hidden, teacher.cpu()).squeeze(1)
    values, positions = torch.sort(sim, descending=True)
    second = float(values[1].item()) if values.numel() > 1 else 0.0
    return {
        "winner_index": int(positions[0].item()),
        "winner_corr": float(values[0].item()),
        "top2_margin": float(values[0].item()) - second,
    }


def lerp_state_dicts(model_a: nn.Module, model_b: nn.Module, lam: float) -> Dict[str, torch.Tensor]:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for key in state_a:
        out[key] = (1.0 - lam) * state_a[key] + lam * state_b[key]
    return out


def interpolation_barrier(
    mask: torch.Tensor,
    cfg: TrainConfig,
    data: Dict[str, torch.Tensor],
    model_a: ScalarMaskedReLUNet,
    model_b: ScalarMaskedReLUNet,
) -> Dict[str, object]:
    eval_model = ScalarMaskedReLUNet(mask=mask.to(cfg.device), train_hidden_bias=cfg.train_hidden_bias).to(cfg.device)
    test_x = data["test_x"].to(cfg.device)
    test_y = data["test_y"].to(cfg.device)
    losses: List[float] = []
    lambdas = np.linspace(0.0, 1.0, cfg.interpolation_steps)
    for lam in lambdas:
        eval_model.load_state_dict(lerp_state_dicts(model_a, model_b, float(lam)))
        losses.append(mse(eval_model, test_x, test_y))
    endpoint_max = max(losses[0], losses[-1])
    return {
        "lambdas": lambdas.tolist(),
        "test_losses": losses,
        "peak_barrier": max(losses) - endpoint_max,
        "midpoint_barrier": losses[len(losses) // 2] - endpoint_max,
    }


def aggregate_winner_agreement(winner_records: List[Dict[str, float | int]]) -> Dict[str, object]:
    winners = np.array([record["winner_index"] for record in winner_records], dtype=int)
    pair_scores = []
    for i, j in itertools.combinations(range(len(winners)), 2):
        pair_scores.append(float(winners[i] == winners[j]))
    return {
        "winner_indices": winners.tolist(),
        "winner_agreement_mean": float(np.mean(pair_scores)) if pair_scores else 1.0,
    }


def summarize_condition(cfg: TrainConfig, condition: str, embedding_mode: str, trial_outputs: List[Dict[str, object]]) -> Dict[str, object]:
    train_mses = [trial["mean_train_mse"] for trial in trial_outputs]
    test_mses = [trial["mean_test_mse"] for trial in trial_outputs]
    pair_metrics = [metric for trial in trial_outputs for metric in trial["pair_metrics"]]
    winner_records = [metric for trial in trial_outputs for metric in trial["winner_metrics"]]
    winner_agg = aggregate_winner_agreement(winner_records)
    summary: Dict[str, object] = {
        "condition": condition,
        "embedding_mode": embedding_mode,
        "config": asdict(cfg),
        "trial_summaries": trial_outputs,
        "mean_train_mse": float(np.mean(train_mses)),
        "mean_test_mse": float(np.mean(test_mses)),
        "pair_metrics": pair_metrics,
        "winner_metrics": winner_records,
        "winner_agreement": winner_agg,
        "mean_winner_corr": float(np.mean([record["winner_corr"] for record in winner_records])),
        "mean_top2_margin": float(np.mean([record["top2_margin"] for record in winner_records])),
        "winner_agreement_mean": winner_agg["winner_agreement_mean"],
    }
    if pair_metrics:
        summary["mean_identity_gap"] = float(np.mean([p["matching"]["gap"] for p in pair_metrics]))
        summary["mean_midpoint_barrier"] = float(np.mean([p["interpolation"]["midpoint_barrier"] for p in pair_metrics]))
        summary["mean_peak_barrier"] = float(np.mean([p["interpolation"]["peak_barrier"] for p in pair_metrics]))
    return summary


def run_single_trial(
    cfg: TrainConfig,
    condition: str,
    mask: torch.Tensor,
    embedding_mode: str,
    dataset_seed: int,
    init_seed_offset: int,
) -> Dict[str, object]:
    data = make_dataset(cfg, embedding_mode=embedding_mode, dataset_seed=dataset_seed)
    models = []
    train_mses = []
    test_mses = []
    winner_records = []
    for run_idx in range(cfg.num_runs):
        model = train_single_model(mask, cfg, data, seed=init_seed_offset + run_idx)
        models.append(model)
        train_mses.append(mse(model, data["train_x"].to(cfg.device), data["train_y"].to(cfg.device)))
        test_mses.append(mse(model, data["test_x"].to(cfg.device), data["test_y"].to(cfg.device)))
        winner_records.append(winner_metrics(model, data["test_x"].to(cfg.device), data["teacher_test"].to(cfg.device)))
    pair_metrics = []
    eval_x = data["test_x"].to(cfg.device)
    for i, j in itertools.combinations(range(cfg.num_runs), 2):
        pair_metrics.append(
            {
                "pair": [i, j],
                "matching": hidden_matching_metrics(models[i], models[j], eval_x),
                "interpolation": interpolation_barrier(mask, cfg, data, models[i], models[j]),
            }
        )
    return {
        "condition": condition,
        "embedding_mode": embedding_mode,
        "dataset_seed": dataset_seed,
        "mean_train_mse": float(np.mean(train_mses)),
        "mean_test_mse": float(np.mean(test_mses)),
        "pair_metrics": pair_metrics,
        "winner_metrics": winner_records,
    }


def parse_outer_seeds(text: str) -> List[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--active-prob", type=float, default=0.15)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--num-runs", type=int, default=6)
    parser.add_argument("--outer-seeds", type=str, default="0,1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--tag", type=str, default="scalar_default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        copies=args.copies,
        width=args.width,
        active_prob=args.active_prob,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bias_weight_decay=args.bias_weight_decay,
        num_runs=args.num_runs,
        outer_seeds=parse_outer_seeds(args.outer_seeds),
        device=args.device,
        torch_threads=args.torch_threads,
        optimizer_name=args.optimizer,
    )
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trial_records = []
    mask_metadata = {
        "mode": "random",
        "outer_masks": [],
    }
    aggregated: Dict[str, object] = {}
    for outer_seed in cfg.outer_seeds or []:
        mask_seed = 10_000 + outer_seed
        generator = torch.Generator()
        generator.manual_seed(mask_seed)
        mask = torch.bernoulli(torch.full((cfg.width, ambient_dim(cfg.copies)), cfg.active_prob), generator=generator)
        hist = pattern_histogram(mask)
        mask_metadata["outer_masks"].append({"outer_seed": outer_seed, "seed": mask_seed, "visible_histogram": hist})
        dataset_seed = 20_000 + outer_seed
        init_seed_offset = 40_000 + 100 * outer_seed
        per_trial_conditions: Dict[str, object] = {}
        print(f"Outer seed {outer_seed}")
        for embedding_mode in ["aligned", "dispersed"]:
            condition = f"random_{embedding_mode}"
            result = run_single_trial(
                cfg=cfg,
                condition=condition,
                mask=mask,
                embedding_mode=embedding_mode,
                dataset_seed=dataset_seed,
                init_seed_offset=init_seed_offset,
            )
            per_trial_conditions[condition] = result
            print(
                f"  {condition}: "
                f"test_mse={result['mean_test_mse']:.6f} "
                f"winner_corr={np.mean([m['winner_corr'] for m in result['winner_metrics']]):.6f} "
                f"winner_margin={np.mean([m['top2_margin'] for m in result['winner_metrics']]):.6f} "
                f"pair_gap={np.mean([p['matching']['gap'] for p in result['pair_metrics']]):.6f} "
                f"mid_barrier={np.mean([p['interpolation']['midpoint_barrier'] for p in result['pair_metrics']]):.6f}"
            )
        trial_records.append(
            {
                "outer_seed": outer_seed,
                "dataset_seed": dataset_seed,
                "init_seed_offset": init_seed_offset,
                "conditions": per_trial_conditions,
            }
        )
    for embedding_mode in ["aligned", "dispersed"]:
        condition = f"random_{embedding_mode}"
        condition_trials = [record["conditions"][condition] for record in trial_records]
        aggregated[condition] = summarize_condition(cfg, condition=condition, embedding_mode=embedding_mode, trial_outputs=condition_trials)
    output = {
        "config": asdict(cfg),
        "mask_metadata": mask_metadata,
        "trial_records": trial_records,
        "aggregate": aggregated,
    }
    output_path = RESULTS_DIR / f"run_{args.tag}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
