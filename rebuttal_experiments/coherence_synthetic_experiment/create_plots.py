from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FC_CACHEDIR", "/tmp/fontconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


SERIES = {
    "exact_copy": {
        "label": "Exact copy",
        "color": "C0",
    },
    "random_tight_frame": {
        "label": "Random frame",
        "color": "C1",
    },
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.02,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-input",
        type=Path,
        default=RESULTS_DIR / "run_scalar_coherence_sweep_o5_pairs10.json",
    )
    parser.add_argument(
        "--random-input",
        type=Path,
        default=RESULTS_DIR / "run_scalar_coherence_sweep_random_frame_o5_pairs10.json",
    )
    parser.add_argument(
        "--midpoint-output",
        type=Path,
        default=FIGURES_DIR / "coherence_midpoint_barrier_comparison_o5_pairs10.pdf",
    )
    parser.add_argument(
        "--endpoint-output",
        type=Path,
        default=FIGURES_DIR / "coherence_endpoint_interpolation_combined_o5_pairs10.pdf",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def flatten_pair_midpoints(payload: dict) -> list[dict]:
    rows = []
    for aggregate_row in payload["aggregate"]:
        support_dim = aggregate_row["support_dims"]
        midpoint_barriers = []
        coherence = None
        for outer_record in payload["outer_records"]:
            spread_record = next(row for row in outer_record["spread_records"] if row["support_dims"] == support_dim)
            coherence = spread_record["coherence"]
            midpoint_barriers.extend(
                float(pair_record["interpolation"]["midpoint_barrier"])
                for pair_record in spread_record["pair_records"]
            )
        values = np.array(midpoint_barriers, dtype=float)
        rows.append(
            {
                "support_dims": support_dim,
                "coherence": coherence,
                "mean_midpoint_barrier": float(values.mean()),
                "std_midpoint_barrier": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return rows


def endpoint_curves(payload: dict, support_dims: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for support_dim in support_dims:
        curves = []
        lambdas = None
        for outer_record in payload["outer_records"]:
            spread_record = next(row for row in outer_record["spread_records"] if row["support_dims"] == support_dim)
            for pair_record in spread_record["pair_records"]:
                interpolation = pair_record["interpolation"]
                lambdas = np.array(interpolation["lambdas"], dtype=float)
                curves.append(np.array(interpolation["test_losses"], dtype=float))
        out[support_dim] = (lambdas, np.stack(curves, axis=0))
    return out


def midpoint_index(x_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(x_values - 0.5)))


def plot_midpoint_series(ax: plt.Axes, rows: list[dict], family: str) -> None:
    style = SERIES[family]
    coherences = np.array([row["coherence"] for row in rows], dtype=float)
    means = np.array([row["mean_midpoint_barrier"] for row in rows], dtype=float)
    stds = np.array([row["std_midpoint_barrier"] for row in rows], dtype=float)

    ax.plot(
        coherences,
        means,
        color=style["color"],
        marker="o",
        markersize=2.8,
    )
    ax.scatter(
        [coherences[0]],
        [means[0] + (0.005 if family == "exact_copy" else 0.0)],
        color=style["color"],
        marker="^",
        s=32,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax.scatter(
        [coherences[-1]],
        [means[-1]],
        color=style["color"],
        marker="s",
        s=24,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax.fill_between(
        coherences,
        means - stds,
        means + stds,
        color=style["color"],
        alpha=0.18,
        linewidth=0.0,
    )


def plot_midpoint_barrier_figure(exact_payload: dict, random_payload: dict, output_path: Path) -> None:
    exact_rows = flatten_pair_midpoints(exact_payload)
    random_rows = flatten_pair_midpoints(random_payload)

    fig, ax = plt.subplots(figsize=(2.0, 1.5), constrained_layout=True)
    plot_midpoint_series(ax, exact_rows, "exact_copy")
    plot_midpoint_series(ax, random_rows, "random_tight_frame")

    ax.set_xlabel(r"Coherence $\nu(\mathcal{U})$", labelpad=0.0)
    ax.set_ylabel("Midpoint barrier")
    ax.grid(alpha=0.25)
    ax.set_xlim(1.03, 0.07)
    ax.set_xticks([1.0, 0.5, 1.0 / 3.0, 0.2, 0.1])
    ax.set_xticklabels(["1", ".5", ".33", ".2", ".1"])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylim(-0.01, 0.31)

    handles = [
        Line2D([0], [0], color=SERIES[family]["color"], lw=1.8, marker="o", markersize=3.0, label=SERIES[family]["label"])
        for family in ("exact_copy", "random_tight_frame")
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_endpoint_family(ax: plt.Axes, payload: dict) -> None:
    family = payload.get("embedding_family", "exact_copy")
    color = SERIES[family]["color"]
    support_dims = [payload["aggregate"][0]["support_dims"], payload["aggregate"][-1]["support_dims"]]
    curves = endpoint_curves(payload, support_dims)
    styles = {
        support_dims[0]: "^",
        support_dims[-1]: "s",
    }

    for support_dim in support_dims:
        lambdas, values = curves[support_dim]
        mean_curve = values.mean(axis=0)
        std_band = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean_curve)
        ax.plot(lambdas, mean_curve, color=color)
        mid_idx = midpoint_index(lambdas)
        ax.scatter(
            [lambdas[mid_idx]],
            [mean_curve[mid_idx] + (0.005 if family == "exact_copy" and support_dim == support_dims[0] else 0.0)],
            color=color,
            marker=styles[support_dim],
            s=26,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.fill_between(
            lambdas,
            mean_curve - std_band,
            mean_curve + std_band,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )


def plot_endpoint_overlay_figure(exact_payload: dict, random_payload: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.0, 1.5), constrained_layout=True)
    plot_endpoint_family(ax, exact_payload)
    plot_endpoint_family(ax, random_payload)

    support_dims = [exact_payload["aggregate"][0]["support_dims"], exact_payload["aggregate"][-1]["support_dims"]]
    exact_curves = endpoint_curves(exact_payload, support_dims)
    random_curves = endpoint_curves(random_payload, support_dims)
    label_specs = {
        support_dims[0]: (r"$\nu(\mathcal{U})=1$", 0.12, 6.0),
        support_dims[-1]: (rf"$\nu(\mathcal{{U}})={exact_payload['aggregate'][-1]['coherence']:.1f}$", 0.55, 6.0),
    }
    for support_dim, (label, x_target, y_offset_points) in label_specs.items():
        exact_x, exact_values = exact_curves[support_dim]
        random_x, random_values = random_curves[support_dim]
        exact_mean = exact_values.mean(axis=0)
        random_mean = random_values.mean(axis=0)
        y_target = 0.5 * (
            float(np.interp(x_target, exact_x, exact_mean))
            + float(np.interp(x_target, random_x, random_mean))
        )
        ax.annotate(
            label,
            xy=(x_target, y_target),
            xytext=(7, y_offset_points),
            textcoords="offset points",
            color="black",
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
        )

    ax.set_xlabel(r"$\lambda$", labelpad=0.0)
    ax.set_ylabel("Test MSE")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_ylim(-0.01, 0.31)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    apply_plot_style()

    exact_payload = load_payload(args.exact_input)
    random_payload = load_payload(args.random_input)

    plot_midpoint_barrier_figure(exact_payload, random_payload, args.midpoint_output)
    plot_endpoint_overlay_figure(exact_payload, random_payload, args.endpoint_output)


if __name__ == "__main__":
    main()
