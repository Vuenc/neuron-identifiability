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
from matplotlib.lines import Line2D

from figure_style import COMPARISON_HEIGHT_IN, COMPARISON_WIDTH_IN, apply_latex_plot_style

SERIES = {
    "exact_copy": {
        "label": "Exact copy",
        "line_color": "C0",
        "fill_color": "C0",
    },
    "random_frame": {
        "label": "Random frame",
        "line_color": "C1",
        "fill_color": "C1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-input",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "run_scalar_coherence_sweep_fast.json",
    )
    parser.add_argument(
        "--random-input",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "run_scalar_coherence_sweep_random_frame_fast.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "coherence_midpoint_barrier_comparison.pdf",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--hide-samples", action="store_true")
    parser.add_argument("--joint-pairs", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def flatten_pair_midpoints(payload: dict) -> list[dict]:
    support_dims = [row["support_dims"] for row in payload["aggregate"]]
    out = []
    for support_dim in support_dims:
        pair_midpoints = []
        coherence = None
        for outer in payload["outer_records"]:
            row = next(r for r in outer["spread_records"] if r["support_dims"] == support_dim)
            coherence = row["coherence"]
            pair_midpoints.extend(
                float(pair["interpolation"]["midpoint_barrier"]) for pair in row["pair_records"]
            )
        values = np.array(pair_midpoints, dtype=float)
        out.append(
            {
                "support_dims": support_dim,
                "coherence": coherence,
                "mean_midpoint_barrier": float(values.mean()),
                "std_midpoint_barrier": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "midpoint_barriers": values.tolist(),
            }
        )
    return out


def plot_series(
    ax: plt.Axes,
    rows: list[dict],
    style: dict,
    show_samples: bool,
    confidence_level: float,
) -> None:
    coherences = np.array([row["coherence"] for row in rows], dtype=float)
    means = np.array([row["mean_midpoint_barrier"] for row in rows], dtype=float)
    stds = np.array([row["std_midpoint_barrier"] for row in rows], dtype=float)
    n = len(rows[0]["midpoint_barriers"]) if rows else 1
    stderr = stds / np.sqrt(max(n, 1))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    ci = z * stderr

    ax.plot(
        coherences,
        means,
        color=style["line_color"],
        marker="o",
        markersize=2.8,
    )
    ax.scatter(
        [coherences[0]],
        [means[0] + (0.005 if style["label"] == "Exact copy" else 0.0)],
        color=style["line_color"],
        marker="^",
        s=32,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax.scatter(
        [coherences[-1]],
        [means[-1]],
        color=style["line_color"],
        marker="s",
        s=24,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax.fill_between(
        coherences,
        means - ci,
        means + ci,
        color=style["fill_color"],
        alpha=0.18,
        linewidth=0.0,
    )
    if show_samples:
        for row in rows:
            x = row["coherence"]
            ys = np.array(row["midpoint_barriers"], dtype=float)
            ax.scatter(
                np.full_like(ys, x, dtype=float),
                ys,
                color=style["fill_color"],
                alpha=0.30,
                s=8,
                linewidths=0.0,
            )


def main() -> None:
    args = parse_args()
    apply_latex_plot_style()
    exact_payload = load_payload(args.exact_input)
    random_payload = load_payload(args.random_input)
    if args.joint_pairs:
        exact_rows = flatten_pair_midpoints(exact_payload)
        random_rows = flatten_pair_midpoints(random_payload)
    else:
        exact_rows = exact_payload["aggregate"]
        random_rows = random_payload["aggregate"]

    coherences = np.array([row["coherence"] for row in exact_rows], dtype=float)

    fig, ax = plt.subplots(
        figsize=(COMPARISON_WIDTH_IN, COMPARISON_HEIGHT_IN),
        constrained_layout=True,
    )
    plot_series(
        ax,
        exact_rows,
        SERIES["exact_copy"],
        show_samples=not args.hide_samples,
        confidence_level=args.confidence_level,
    )
    plot_series(
        ax,
        random_rows,
        SERIES["random_frame"],
        show_samples=not args.hide_samples,
        confidence_level=args.confidence_level,
    )

    ax.set_xlabel(r"Coherence $\nu(\mathcal{U})$")
    ax.set_ylabel("Midpoint barrier")
    ax.grid(alpha=0.25)
    ax.set_xlim(coherences.max() + 0.03, coherences.min() - 0.02)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([1.0, 0.5, 1.0 / 3.0, 0.2, 0.1])
    ax.set_xticklabels([r"$1.0$", r"$0.5$", r"$0.33$", r"$0.2$", r"$0.1$"])
    family_handles = [
        Line2D([0], [0], color=style["line_color"], lw=1.8, marker="o", markersize=3.0, label=style["label"])
        for style in SERIES.values()
    ]
    ax.legend(handles=family_handles, loc="upper left", frameon=False, handlelength=1.8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
