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

from figure_style import INTERPOLATION_HEIGHT_IN, INTERPOLATION_WIDTH_IN, apply_latex_plot_style
from plot_scalar_coherence_endpoints import plot_endpoint_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-input", type=Path, required=True)
    parser.add_argument("--random-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--joint-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_latex_plot_style()
    exact_payload = json.loads(args.exact_input.read_text())
    random_payload = json.loads(args.random_input.read_text())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(INTERPOLATION_WIDTH_IN, INTERPOLATION_HEIGHT_IN),
        constrained_layout=True,
        sharey=True,
    )
    plot_endpoint_panel(
        axes[0],
        payload=exact_payload,
        confidence_level=args.confidence_level,
        joint_pairs=args.joint_pairs,
        panel_label="Exact copy",
        show_ylabel=True,
        show_inline_labels=True,
    )
    plot_endpoint_panel(
        axes[1],
        payload=random_payload,
        confidence_level=args.confidence_level,
        joint_pairs=args.joint_pairs,
        panel_label="Random frame",
        show_ylabel=False,
        show_inline_labels=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
