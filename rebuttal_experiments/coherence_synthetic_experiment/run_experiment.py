from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_outer_seeds(text: str) -> list[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


@dataclass
class ExperimentConfig:
    copies: int = 10
    width: int = 128
    active_prob: float = 0.15
    n_train: int = 1024
    n_test: int = 2048
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-2
    bias_weight_decay: float = 0.0
    outer_seeds: list[int] | None = None
    device: str = "cpu"
    torch_threads: int = 1
    interpolation_steps: int = 21
    optimizer_name: str = "adamw"
    train_hidden_bias: bool = True
    num_pairs_per_outer_seed: int = 10

    def output_config(self) -> dict:
        config = asdict(self)
        config["num_runs"] = 2 * self.num_pairs_per_outer_seed
        return config


def ambient_dim(copies: int) -> int:
    return 2 * copies


def teacher_output(latent: torch.Tensor) -> torch.Tensor:
    return F.relu(latent[:, 0] + latent[:, 1]).unsqueeze(1)


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

    def hidden_act(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x @ (self.hidden_weight * self.mask).T + self.hidden_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden_act(x).sum(dim=1, keepdim=True)


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    perm = torch.randperm(x.shape[0], device=x.device)
    for start in range(0, x.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        yield x[idx], y[idx]


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return F.mse_loss(model(x), y).item()


def train_single_model(
    mask: torch.Tensor,
    cfg: ExperimentConfig,
    data: dict[str, torch.Tensor],
    seed: int,
) -> ScalarMaskedReLUNet:
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


def lerp_state_dicts(model_a: nn.Module, model_b: nn.Module, lam: float) -> dict[str, torch.Tensor]:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    return {key: (1.0 - lam) * state_a[key] + lam * state_b[key] for key in state_a}


def interpolation_barrier(
    mask: torch.Tensor,
    cfg: ExperimentConfig,
    data: dict[str, torch.Tensor],
    model_a: ScalarMaskedReLUNet,
    model_b: ScalarMaskedReLUNet,
) -> dict[str, object]:
    eval_model = ScalarMaskedReLUNet(mask=mask.to(cfg.device), train_hidden_bias=cfg.train_hidden_bias).to(cfg.device)
    test_x = data["test_x"].to(cfg.device)
    test_y = data["test_y"].to(cfg.device)
    losses: list[float] = []
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


def make_exact_copy_embedding(copies: int, active_copies: int) -> torch.Tensor:
    basis = torch.zeros(ambient_dim(copies), 2)
    scale = 1.0 / np.sqrt(active_copies)
    for idx in range(active_copies):
        basis[2 * idx + 0, 0] = scale
        basis[2 * idx + 1, 1] = scale
    return basis


def make_random_tight_embedding(
    copies: int,
    active_copies: int,
    perm: torch.Tensor,
    angles: torch.Tensor,
    anchor_aligned_endpoint: bool,
) -> torch.Tensor:
    basis = torch.zeros(ambient_dim(copies), 2)
    if active_copies == 1 and anchor_aligned_endpoint:
        basis[int(perm[0].item()), 0] = 1.0
        basis[int(perm[1].item()), 1] = 1.0
        return basis

    scale = 1.0 / np.sqrt(active_copies)
    for idx in range(active_copies):
        theta = float(angles[idx].item())
        row_a = scale * torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32)
        row_b = scale * torch.tensor([-np.sin(theta), np.cos(theta)], dtype=torch.float32)
        basis[int(perm[2 * idx].item())] = row_a
        basis[int(perm[2 * idx + 1].item())] = row_b
    return basis


def make_dataset_from_basis(
    train_latent: torch.Tensor,
    test_latent: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    train_x = train_latent @ basis.T
    test_x = test_latent @ basis.T
    train_y = teacher_output(train_latent)
    test_y = teacher_output(test_latent)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }


def run_family(
    family: str,
    cfg: ExperimentConfig,
    anchor_aligned_endpoint: bool = False,
) -> dict:
    outer_records = []
    for outer_seed in cfg.outer_seeds or []:
        mask_generator = torch.Generator().manual_seed(10_000 + outer_seed)
        mask = torch.bernoulli(
            torch.full((cfg.width, ambient_dim(cfg.copies)), cfg.active_prob),
            generator=mask_generator,
        )

        data_generator = torch.Generator().manual_seed(20_000 + outer_seed)
        train_latent = torch.randn(cfg.n_train, 2, generator=data_generator)
        test_latent = torch.randn(cfg.n_test, 2, generator=data_generator)

        perm = None
        angles = None
        if family == "random_tight_frame":
            frame_generator = torch.Generator().manual_seed(30_000 + outer_seed)
            perm = torch.randperm(ambient_dim(cfg.copies), generator=frame_generator)
            angles = torch.rand(cfg.copies, generator=frame_generator) * np.pi

        spread_records = []
        print(f"[{family}] outer seed {outer_seed}")
        for active_copies in range(1, cfg.copies + 1):
            if family == "exact_copy":
                basis = make_exact_copy_embedding(cfg.copies, active_copies)
            elif family == "random_tight_frame":
                basis = make_random_tight_embedding(
                    cfg.copies,
                    active_copies,
                    perm=perm,
                    angles=angles,
                    anchor_aligned_endpoint=anchor_aligned_endpoint,
                )
            else:
                raise ValueError(f"Unknown family: {family}")

            data = make_dataset_from_basis(train_latent, test_latent, basis)
            pair_records = []
            pair_curves = []
            pair_peak_barriers = []
            pair_midpoint_barriers = []
            for pair_idx in range(cfg.num_pairs_per_outer_seed):
                seed0 = 40_000 + 1_000 * outer_seed + 2 * pair_idx
                seed1 = seed0 + 1
                model_a = train_single_model(mask, cfg, data, seed=seed0)
                model_b = train_single_model(mask, cfg, data, seed=seed1)
                pair_interp = interpolation_barrier(mask, cfg, data, model_a, model_b)
                pair_records.append(
                    {
                        "pair_index": pair_idx,
                        "seeds": [seed0, seed1],
                        "interpolation": pair_interp,
                    }
                )
                pair_curves.append(np.array(pair_interp["test_losses"], dtype=float))
                pair_peak_barriers.append(float(pair_interp["peak_barrier"]))
                pair_midpoint_barriers.append(float(pair_interp["midpoint_barrier"]))

            mean_curve = np.stack(pair_curves, axis=0).mean(axis=0)
            coherence = 1.0 / active_copies
            spread_records.append(
                {
                    "active_copies": active_copies,
                    "support_dims": 2 * active_copies,
                    "coherence": coherence,
                    "num_pairs": cfg.num_pairs_per_outer_seed,
                    "pair_records": pair_records,
                    "interpolation": {
                        "lambdas": pair_records[0]["interpolation"]["lambdas"],
                        "test_losses": mean_curve.tolist(),
                        "peak_barrier": float(np.mean(pair_peak_barriers)),
                        "midpoint_barrier": float(np.mean(pair_midpoint_barriers)),
                    },
                }
            )
            print(
                f"  support_dims={2 * active_copies:2d} "
                f"coherence={coherence:.3f} "
                f"midpoint_barrier={np.mean(pair_midpoint_barriers):.6f}"
            )

        outer_record = {
            "outer_seed": outer_seed,
            "mask_seed": 10_000 + outer_seed,
            "dataset_seed": 20_000 + outer_seed,
            "num_pairs": cfg.num_pairs_per_outer_seed,
            "spread_records": spread_records,
        }
        if family == "random_tight_frame":
            outer_record["frame_seed"] = 30_000 + outer_seed
        outer_records.append(outer_record)

    aggregate = []
    for active_copies in range(1, cfg.copies + 1):
        midpoint_barriers = []
        for outer_record in outer_records:
            record = outer_record["spread_records"][active_copies - 1]
            midpoint_barriers.extend(
                float(pair_record["interpolation"]["midpoint_barrier"])
                for pair_record in record["pair_records"]
            )
        values = np.array(midpoint_barriers, dtype=float)
        aggregate.append(
            {
                "active_copies": active_copies,
                "support_dims": 2 * active_copies,
                "coherence": 1.0 / active_copies,
                "mean_midpoint_barrier": float(values.mean()),
                "std_midpoint_barrier": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "midpoint_barriers": values.tolist(),
            }
        )

    return {
        "config": cfg.output_config(),
        "embedding_family": family,
        "anchor_aligned_endpoint": bool(anchor_aligned_endpoint),
        "num_pairs_per_outer_seed": cfg.num_pairs_per_outer_seed,
        "outer_records": outer_records,
        "aggregate": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--active-prob", type=float, default=0.15)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--outer-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--anchor-aligned-endpoint", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--exact-output",
        type=Path,
        default=RESULTS_DIR / "run_scalar_coherence_sweep_o5_pairs10.json",
    )
    parser.add_argument(
        "--random-output",
        type=Path,
        default=RESULTS_DIR / "run_scalar_coherence_sweep_random_frame_o5_pairs10.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
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
        outer_seeds=parse_outer_seeds(args.outer_seeds),
        device=args.device,
        torch_threads=args.torch_threads,
        optimizer_name=args.optimizer,
        num_pairs_per_outer_seed=args.num_pairs,
    )
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    exact_payload = run_family("exact_copy", cfg)
    random_payload = run_family(
        "random_tight_frame",
        cfg,
        anchor_aligned_endpoint=args.anchor_aligned_endpoint,
    )

    args.exact_output.write_text(json.dumps(exact_payload, indent=2))
    args.random_output.write_text(json.dumps(random_payload, indent=2))
    print(f"Saved exact-copy results to {args.exact_output}")
    print(f"Saved random-frame results to {args.random_output}")


if __name__ == "__main__":
    main()
