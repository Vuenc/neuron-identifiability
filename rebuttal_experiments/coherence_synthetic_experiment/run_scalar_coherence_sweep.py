from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from run_scalar_experiment import (
    RESULTS_DIR,
    TrainConfig,
    ambient_dim,
    hidden_matching_metrics,
    interpolation_barrier,
    mse,
    pattern_histogram,
    teacher_output,
    train_single_model,
    winner_metrics,
)


def make_partial_embedding(total_copies: int, active_copies: int) -> torch.Tensor:
    dim = ambient_dim(total_copies)
    basis = torch.zeros(dim, 2)
    scale = 1.0 / np.sqrt(active_copies)
    for t in range(active_copies):
        basis[2 * t + 0, 0] = scale
        basis[2 * t + 1, 1] = scale
    return basis


def make_dataset_from_latent(
    cfg: TrainConfig,
    train_latent: torch.Tensor,
    test_latent: torch.Tensor,
    active_copies: int,
) -> Dict[str, torch.Tensor]:
    basis = make_partial_embedding(cfg.copies, active_copies)
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


def parse_outer_seeds(text: str) -> List[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--active-prob", type=float, default=0.15)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--outer-seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--num-pairs", type=int, default=1)
    parser.add_argument("--tag", type=str, default="scalar_coherence_sweep")
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
        num_runs=2 * args.num_pairs,
        outer_seeds=parse_outer_seeds(args.outer_seeds),
        device=args.device,
        torch_threads=args.torch_threads,
        optimizer_name=args.optimizer,
        train_hidden_bias=True,
    )
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outer_records = []
    for outer_seed in cfg.outer_seeds or []:
        generator = torch.Generator()
        generator.manual_seed(10_000 + outer_seed)
        mask = torch.bernoulli(torch.full((cfg.width, ambient_dim(cfg.copies)), cfg.active_prob), generator=generator)
        mask_hist = pattern_histogram(mask)

        generator.manual_seed(20_000 + outer_seed)
        train_latent = torch.randn(cfg.n_train, 2, generator=generator)
        test_latent = torch.randn(cfg.n_test, 2, generator=generator)

        spread_records = []
        print(f"Outer seed {outer_seed}")
        for active_copies in range(1, cfg.copies + 1):
            data = make_dataset_from_latent(cfg, train_latent, test_latent, active_copies=active_copies)
            pair_records = []
            model_train_mses = []
            model_test_mses = []
            winner_records = []
            interp_losses = []
            for pair_idx in range(args.num_pairs):
                seed0 = 40_000 + 1_000 * outer_seed + 2 * pair_idx
                seed1 = seed0 + 1
                model_a = train_single_model(mask, cfg, data, seed=seed0)
                model_b = train_single_model(mask, cfg, data, seed=seed1)
                pair_matching = hidden_matching_metrics(model_a, model_b, data["test_x"].to(cfg.device))
                pair_interp = interpolation_barrier(mask, cfg, data, model_a, model_b)
                pair_records.append(
                    {
                        "pair_index": pair_idx,
                        "seeds": [seed0, seed1],
                        "matching": pair_matching,
                        "interpolation": pair_interp,
                    }
                )
                interp_losses.append(np.array(pair_interp["test_losses"], dtype=float))
                model_train_mses.extend(
                    [
                        mse(model_a, data["train_x"].to(cfg.device), data["train_y"].to(cfg.device)),
                        mse(model_b, data["train_x"].to(cfg.device), data["train_y"].to(cfg.device)),
                    ]
                )
                model_test_mses.extend(
                    [
                        mse(model_a, data["test_x"].to(cfg.device), data["test_y"].to(cfg.device)),
                        mse(model_b, data["test_x"].to(cfg.device), data["test_y"].to(cfg.device)),
                    ]
                )
                winner_records.extend(
                    [
                        winner_metrics(model_a, data["test_x"].to(cfg.device), data["teacher_test"].to(cfg.device)),
                        winner_metrics(model_b, data["test_x"].to(cfg.device), data["teacher_test"].to(cfg.device)),
                    ]
                )
            mean_matching = {
                "identity_obj": float(np.mean([p["matching"]["identity_obj"] for p in pair_records])),
                "optimal_obj": float(np.mean([p["matching"]["optimal_obj"] for p in pair_records])),
                "gap": float(np.mean([p["matching"]["gap"] for p in pair_records])),
            }
            mean_interp_losses = np.stack(interp_losses, axis=0).mean(axis=0)
            mean_interpolation = {
                "lambdas": pair_records[0]["interpolation"]["lambdas"],
                "test_losses": mean_interp_losses.tolist(),
                "peak_barrier": float(np.mean([p["interpolation"]["peak_barrier"] for p in pair_records])),
                "midpoint_barrier": float(np.mean([p["interpolation"]["midpoint_barrier"] for p in pair_records])),
            }
            record = {
                "active_copies": active_copies,
                "support_dims": 2 * active_copies,
                "coherence": 1.0 / active_copies,
                "num_pairs": args.num_pairs,
                "mean_train_mse": float(np.mean(model_train_mses)),
                "mean_test_mse": float(np.mean(model_test_mses)),
                "winner_metrics": winner_records,
                "pair_records": pair_records,
                "matching": mean_matching,
                "interpolation": mean_interpolation,
            }
            spread_records.append(record)
            print(
                f"  spread={2 * active_copies:2d} dims "
                f"coh={1.0 / active_copies:.3f} "
                f"test_mse={record['mean_test_mse']:.6f} "
                f"pair_gap={record['matching']['gap']:.6f} "
                f"mid_barrier={record['interpolation']['midpoint_barrier']:.6f}"
            )
        outer_records.append(
            {
                "outer_seed": outer_seed,
                "mask_seed": 10_000 + outer_seed,
                "dataset_seed": 20_000 + outer_seed,
                "num_pairs": args.num_pairs,
                "mask_histogram": mask_hist,
                "spread_records": spread_records,
            }
        )

    aggregates = []
    for active_copies in range(1, cfg.copies + 1):
        records = [outer["spread_records"][active_copies - 1] for outer in outer_records]
        aggregates.append(
            {
                "active_copies": active_copies,
                "support_dims": 2 * active_copies,
                "coherence": 1.0 / active_copies,
                "mean_test_mse": float(np.mean([r["mean_test_mse"] for r in records])),
                "mean_identity_gap": float(np.mean([r["matching"]["gap"] for r in records])),
                "mean_midpoint_barrier": float(np.mean([r["interpolation"]["midpoint_barrier"] for r in records])),
                "std_midpoint_barrier": float(np.std([r["interpolation"]["midpoint_barrier"] for r in records], ddof=1)) if len(records) > 1 else 0.0,
                "midpoint_barriers": [float(r["interpolation"]["midpoint_barrier"]) for r in records],
            }
        )

    output = {
        "config": asdict(cfg),
        "num_pairs_per_outer_seed": args.num_pairs,
        "outer_records": outer_records,
        "aggregate": aggregates,
    }
    output_path = RESULTS_DIR / f"run_{args.tag}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
