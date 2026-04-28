from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FC_CACHEDIR", "/tmp/fontconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import INTERPOLATION_HEIGHT_IN, INTERPOLATION_WIDTH_IN, apply_latex_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--joint-pairs", action="store_true")
    return parser.parse_args()


def endpoint_curves(
    payload: dict,
    support_dims: list[int],
    joint_pairs: bool,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for support_dim in support_dims:
        curves = []
        lambdas = None
        for outer in payload["outer_records"]:
            row = next(r for r in outer["spread_records"] if r["support_dims"] == support_dim)
            if joint_pairs:
                for pair in row["pair_records"]:
                    interp = pair["interpolation"]
                    lambdas = np.array(interp["lambdas"], dtype=float)
                    curves.append(np.array(interp["test_losses"], dtype=float))
            else:
                interp = row["interpolation"]
                lambdas = np.array(interp["lambdas"], dtype=float)
                curves.append(np.array(interp["test_losses"], dtype=float))
        out[support_dim] = (lambdas, np.stack(curves, axis=0))
    return out


def add_inline_curve_label(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    label: str,
    color: str,
    x_target: float,
    y_offset_points: float,
) -> None:
    y_target = float(np.interp(x_target, x_values, y_values))
    ax.annotate(
        label,
        xy=(x_target, y_target),
        xytext=(7, y_offset_points),
        textcoords="offset points",
        color=color,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
    )


def midpoint_index(x_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(x_values - 0.5)))


def infer_family_style(payload: dict) -> tuple[str, str]:
    family = payload.get("embedding_family", "exact_copy")
    if str(family).startswith("random"):
        return "Random frame", "C1"
    return "Exact copy", "C0"


def plot_endpoint_panel(
    ax: plt.Axes,
    payload: dict,
    confidence_level: float,
    joint_pairs: bool,
    panel_label: str | None = None,
    show_ylabel: bool = True,
    show_inline_labels: bool = True,
) -> None:
    family_name, family_color = infer_family_style(payload)
    support_dims = [row["support_dims"] for row in payload["aggregate"]]
    curves = endpoint_curves(
        payload,
        support_dims=[support_dims[0], support_dims[-1]],
        joint_pairs=joint_pairs,
    )
    styles = {
        support_dims[0]: {
            "label": r"$\nu(\mathcal{U})=1$",
            "marker": "^",
        },
        support_dims[-1]: {
            "label": rf"$\nu(\mathcal{{U}})={payload['aggregate'][-1]['coherence']:.1f}$",
            "marker": "s",
        },
    }
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    mean_curves: dict[int, tuple[np.ndarray, np.ndarray, str, str, str]] = {}
    for support_dim in [support_dims[0], support_dims[-1]]:
        lambdas, values = curves[support_dim]
        mean_curve = values.mean(axis=0)
        stderr = values.std(axis=0, ddof=1) / np.sqrt(max(values.shape[0], 1)) if values.shape[0] > 1 else np.zeros_like(mean_curve)
        ci = z * stderr
        style = styles[support_dim]
        ax.plot(
            lambdas,
            mean_curve,
            color=family_color,
        )
        mid_idx = midpoint_index(lambdas)
        ax.scatter(
            [lambdas[mid_idx]],
            [mean_curve[mid_idx] + (0.005 if family_name == "Exact copy" and support_dim == support_dims[0] else 0.0)],
            color=family_color,
            marker=style["marker"],
            s=26,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.fill_between(lambdas, mean_curve - ci, mean_curve + ci, color=family_color, alpha=0.18, linewidth=0.0)
        mean_curves[support_dim] = (lambdas, mean_curve, family_color, style["label"], style["marker"])

    ax.set_xlabel(r"$\lambda$")
    if show_ylabel:
        ax.set_ylabel("Test MSE")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(0.0, 0.3)
    if panel_label:
        ax.text(
            0.5,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
    if show_inline_labels:
        blue_dim = support_dims[0]
        orange_dim = support_dims[-1]
        blue_x, blue_y, blue_color, blue_label, _blue_marker = mean_curves[blue_dim]
        orange_x, orange_y, orange_color, orange_label, _orange_marker = mean_curves[orange_dim]
        add_inline_curve_label(
            ax,
            blue_x,
            blue_y,
            blue_label,
            blue_color,
            x_target=0.12,
            y_offset_points=6.0,
        )
        add_inline_curve_label(
            ax,
            orange_x,
            orange_y,
            orange_label,
            orange_color,
            x_target=0.58,
            y_offset_points=6.0,
        )


def main() -> None:
    args = parse_args()
    apply_latex_plot_style()
    payload = json.loads(args.input.read_text())

    fig, ax = plt.subplots(
        figsize=(INTERPOLATION_WIDTH_IN, INTERPOLATION_HEIGHT_IN),
        constrained_layout=True,
    )
    plot_endpoint_panel(
        ax,
        payload=payload,
        confidence_level=args.confidence_level,
        joint_pairs=args.joint_pairs,
        show_ylabel=True,
        show_inline_labels=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")



if __name__ == "__main__":
    main()
