from __future__ import annotations

import argparse
from collections import namedtuple
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List
import polars as pl
import hydra

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib-asymmetric-networks"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("FC_CACHEDIR", tempfile.gettempdir())

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import matplotlib.colors
import numpy as np
import pandas as pd
import checkpoint_directories


DEFAULT_OUTPUTS_PATH = (
    "outputs"
)
FILE_EXTENSION = "pdf"


# Helper functions style_figure and plot_grid for falling back to plotly in interactive notebooks. Don't delete!
def style_figure(fig):
    fig.update_layout(
        template="simple_white",
        font=dict(color="black", family="Deja Vu Math TeX Gyre"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
        ticklen=5,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="black",
        ticks="outside",
        ticklen=5,
    )


def plot_grid(figures, width=500, height=300):
    """
    Display figures in a grid with complete separation.
    """
    import plotly.graph_objects as go
    from IPython.display import display
    import ipywidgets as widgets
    rows = []

    for row_figs in figures:
        fig_widgets = []
        for fig in row_figs:
            if fig is None:
                continue

            # Create FigureWidget from existing figure's data and layout
            if not isinstance(fig, widgets.Widget):
                fw = go.FigureWidget(data=fig.data, layout=fig.layout)
                fw.update_layout(width=width, height=height)
                fig_widgets.append(fw)
            else:
                fig_widgets.append(fig)

        rows.append(widgets.HBox(fig_widgets))

    grid = widgets.VBox(rows)
    display(grid)


def _apply_plot_style() -> None:
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


_apply_plot_style()

_COLOR_SEQUENCE = ["C0", "C1", "C2", "C3", "C4", "C5"]
_LINESTYLE_SEQUENCE = ["-", "-", ":", "--"]

_RUN_LABELS = {
    "mlp_symmetry0": r"$\mathrm{MLP}$",
    "mlp_symmetry1_kappa0": r"$\boldsymbol{W}$-$\mathrm{MLP}$, $\kappa=0$",
    "mlp_symmetry1_kappa1": r"$\boldsymbol{W}$-$\mathrm{MLP}$, $\kappa=1$",
    "mlp_symmetry3_kappa1": r"$\mathcal{N}$-$\mathrm{MLP}$, $\kappa=1$",
    "resnet_symmetry0": r"$\mathrm{ResNet}$",
    "resnet_symmetry1_kappa0": r"$\boldsymbol{W}$-$\mathrm{ResNet}$, $\kappa=0$",
    "resnet_symmetry1_kappa2": r"$\boldsymbol{W}$-$\mathrm{ResNet}$, $\kappa=2$",
    "resnet_symmetry3_kappa2": r"$\mathcal{N}$-$\mathrm{ResNet}$, $\kappa=2$",
}


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _format_compact_decimal(value: float) -> str:
    text = _format_number(value)
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _extract_kappa_value(run_key: str) -> float | None:
    match = re.search(r"kappa([\d.]+)", run_key)
    return float(match.group(1)) if match else None

def _extract_sparsity_value(run_key: str) -> float | None:
    match = re.search(r"sparsity([\d.]+)", run_key)
    return float(match.group(1)) if match else None


def _plot_dir() -> Path:
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, format=FILE_EXTENSION, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, *, grid_axis: str | None = "y", grid_alpha: float = 0.25) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=grid_alpha, linewidth=0.55)
    ax.tick_params(
        axis="both",
        direction="out",
        length=plt.rcParams["xtick.major.size"],
        width=plt.rcParams["xtick.major.width"],
        colors="black",
        labelsize=plt.rcParams["xtick.labelsize"],
    )
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(plt.rcParams["axes.linewidth"])
        ax.spines[spine].set_color("black")


def _set_zero_aligned_ylim(
    ax: plt.Axes,
    data_min: float,
    *,
    zero_line_fraction: float = 0.035,
    pad_fraction: float = 0.005,
) -> None:
    _, current_top = ax.get_ylim()
    data_pad = pad_fraction * max(current_top - data_min, 1e-6)
    zero_aligned_bottom = -(zero_line_fraction * current_top) / (1.0 - zero_line_fraction)
    ax.set_ylim(bottom=min(data_min - data_pad, zero_aligned_bottom), top=current_top)


def _ordered_run_labels(run_keys: pd.Series) -> list[tuple[str, str]]:
    unique_run_keys = set(run_keys.tolist())
    return [(run_key, label) for run_key, label in _RUN_LABELS.items() if run_key in unique_run_keys]


def _safe_name(value: str | None) -> str:
    if value is None:
        return "alllayers"
    return str(value).replace("/", "_").replace(" ", "_")


####################################################################
#                                                                  #
#                 -------------------------------                  #
#                      Standard LMC Experiments                    #
#                 -------------------------------                  #
#                                                                  #
####################################################################


def load_lmc_df(path: str | Path) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(
        [
            {
                "run_key": entry["run_key"],
                "lambda": interpolation_lambda,
                "train_accuracy": entry[mode]["train_accuracy"][i],
                "mode": mode,
            }
            for entry in data
            for mode in [
                "interpolation_results_unaligned",
                "interpolation_results_activation_aligned",
                "interpolation_results_weight_aligned",
            ]
            for i, interpolation_lambda in enumerate(entry[mode]["lambdas"])
        ]
    )


def lmc_plot(
    df: pd.DataFrame,
    alignment_mode: str,
    range_y: list[float],
) -> plt.Figure:
    df = df[df["mode"] == alignment_mode]
    agg = (
        df.groupby(["run_key", "mode", "lambda"])
        .agg(train_accuracy_mean=("train_accuracy", "mean"))
        .reset_index()
        .sort_values(["run_key", "mode", "lambda"])
    )

    ordered_labels = _ordered_run_labels(agg["run_key"])
    draw_order = {run_key: idx for idx, (run_key, _) in enumerate(ordered_labels)}
    for red_run_key, green_run_key in [
        ("mlp_symmetry3_kappa1", "mlp_symmetry1_kappa1"),
        ("resnet_symmetry3_kappa2", "resnet_symmetry1_kappa2"),
    ]:
        if red_run_key in draw_order and green_run_key in draw_order:
            draw_order[red_run_key], draw_order[green_run_key] = (
                draw_order[green_run_key],
                draw_order[red_run_key],
            )
    plot_order = sorted(ordered_labels, key=lambda item: draw_order[item[0]])
    style_index = {run_key: idx for idx, (run_key, _) in enumerate(ordered_labels)}
    line_handles: dict[str, plt.Line2D] = {}
    fig, ax = plt.subplots(figsize=(2.0, 1.2))

    for zorder, (run_key, label) in enumerate(plot_order, start=2):
        idx = style_index[run_key]
        run_df = agg[agg["run_key"] == run_key].sort_values("lambda")
        line, = ax.plot(
            run_df["lambda"],
            run_df["train_accuracy_mean"],
            color=_COLOR_SEQUENCE[idx],
            linestyle=_LINESTYLE_SEQUENCE[idx],
            linewidth=1.5,
            label=label,
            zorder=zorder,
        )
        line_handles[run_key] = line

    ax.set_ylim(range_y)
    ax.set_xlim(agg["lambda"].min(), agg["lambda"].max())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: _format_number(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{_format_number(y)}%"))
    ax.set_xlabel(r"$\lambda$")

    _style_axes(ax, grid_axis="both")
    if alignment_mode == "interpolation_results_activation_aligned":
        ax.legend(
            handles=[line_handles[run_key] for run_key, _ in ordered_labels],
            labels=[label for _, label in ordered_labels],
            loc="lower left",
            frameon=False,
            handlelength=1.8,
        )

    return fig


def make_lmc_plots_mlp(outputs_path: str = DEFAULT_OUTPUTS_PATH) -> dict[str, plt.Figure]:
    df = load_lmc_df(f"{outputs_path}/lmc-mlp-aligned-unaligned-37steps-8modelpairs.json")
    important_runs = [
        "mlp_symmetry0",
        "mlp_symmetry1_kappa0",
        "mlp_symmetry1_kappa1",
        "mlp_symmetry3_kappa1",
    ]
    df = df[df["run_key"].isin(important_runs)]
    plots = {
        mode: lmc_plot(df, mode, range_y=[93, 100.1])
        for mode in df["mode"].unique()
    }
    for mode, fig in plots.items():
        _save_figure(fig, _plot_dir() / f"lmc-mlp-{mode}.{FILE_EXTENSION}")
    return plots


def make_lmc_plots_resnet(outputs_path: str = DEFAULT_OUTPUTS_PATH) -> dict[str, plt.Figure]:
    df = load_lmc_df(f"{outputs_path}/lmc-resnet-aligned-unaligned-37steps-8modelpairs.json")
    plots = {
        mode: lmc_plot(df, mode, range_y=[0, 101])
        for mode in df["mode"].unique()
    }
    for mode, fig in plots.items():
        _save_figure(fig, _plot_dir() / f"lmc-resnet-{mode}.{FILE_EXTENSION}")
    return plots


####################################################################
#                                                                  #
#                 -------------------------------                  #
#                 Activation Matching Experiments                  #
#                 -------------------------------                  #
#                                                                  #
####################################################################


def activation_matching_kappa_sweep_plot(
    df: pd.DataFrame,
    random_objectives: pd.DataFrame,
    epoch: int = 100,
    y_tick_step: float | None = None,
    **kwargs,
) -> dict[str, plt.Figure]:
    df = df.copy()
    random_objectives = random_objectives.copy()
    df["kappa"] = df["run_key"].apply(_extract_kappa_value)
    random_objectives["kappa"] = random_objectives["run_key"].apply(_extract_kappa_value)

    epoch_df = df[df["epoch"] == epoch].copy()
    layers = sorted(epoch_df["layer"].unique()) + [None]

    return activation_matching_sweep_plot(random_objectives, epoch, "kappa", r"$\kappa$", y_tick_step, epoch_df, layers, **kwargs)


def activation_matching_sparsity_sweep_plot(
    df: pd.DataFrame,
    random_objectives: pd.DataFrame,
    kappa_filter=None,
    epoch: int = 100,
    y_tick_step: float | None = None,
) -> dict[str, plt.Figure]:
    df = df.copy()
    random_objectives = random_objectives.copy()
    df["sparsity"] = df["run_key"].apply(_extract_sparsity_value)
    random_objectives["sparsity"] = random_objectives["run_key"].apply(_extract_sparsity_value)

    if kappa_filter is not None:
        df["kappa"] = df["run_key"].apply(_extract_kappa_value)
        random_objectives["kappa"] = random_objectives["run_key"].apply(_extract_kappa_value)
        df = df[df["kappa"] == kappa_filter]
        random_objectives = random_objectives[random_objectives["kappa"] == kappa_filter]

    epoch_df = df[df["epoch"] == epoch].copy()
    layers = sorted(epoch_df["layer"].unique()) + [None]

    return activation_matching_sweep_plot(random_objectives, epoch, "sparsity", rf"Proportion of fixed weights ($\kappa$={kappa_filter})", y_tick_step, epoch_df, layers)


def activation_matching_sweep_plot(random_objectives, epoch, x_axis_column: str, x_axis_label: str, y_tick_step, epoch_df, layers, y_range=None, figsize=None):
    out_plots: dict[str, plt.Figure] = {}
    for layer in layers:
        layer_df = epoch_df[epoch_df["layer"] == layer].copy() if layer is not None else epoch_df.copy()
        random_layer_df = random_objectives[random_objectives["epoch"] == epoch].copy()
        if layer is not None:
            random_layer_df = random_layer_df[random_layer_df["layer"] == layer].copy()

        agg_df = (
            layer_df.groupby(x_axis_column)
            .agg(
                objective_optimal=("objective_optimal", "mean"),
                objective_identity=("objective_identity", "mean"),
            )
            .reset_index()
            .sort_values(x_axis_column)
        )
        agg_random = (
            random_layer_df.groupby(x_axis_column)["objective_random"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values(x_axis_column)
        )

        x_positions = np.arange(len(agg_df), dtype=float)
        fig, ax = plt.subplots(figsize=figsize or (3.0, 1.7))
        ax.plot(
            x_positions,
            agg_df["objective_optimal"],
            color="C0",
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Optimal",
        )
        ax.plot(
            x_positions,
            agg_df["objective_identity"],
            color="C1",
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Identity",
        )
        ax.plot(
            x_positions,
            agg_random["mean"],
            color="C2",
            linestyle=":",
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Random",
        )
        ax.fill_between(
            x_positions,
            agg_random["min"],
            agg_random["max"],
            color="C2",
            alpha=0.18,
            linewidth=0,
        )

        xticks = agg_df[x_axis_column].tolist()
        ax.set_xticks(x_positions)
        ax.set_xticklabels([_format_compact_decimal(kappa) for kappa in xticks])
        ax.minorticks_off()
        ax.set_xlabel(x_axis_label)
        ax.set_xlim(-0.18, len(x_positions) - 1 + 0.18)
        _set_zero_aligned_ylim(
            ax,
            data_min=min(
                float(agg_df["objective_optimal"].min()),
                float(agg_df["objective_identity"].min()),
                float(agg_random["min"].min()),
            ),
        )
        if y_tick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
        if y_range is not None:
            plt.ylim(bottom=y_range[0], top=y_range[1])

        _style_axes(ax, grid_axis="both")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 0.98),
            ncol=1,
            frameon=False,
            handlelength=1.8,
        )

        out_plots[_safe_name(layer)] = fig

    return out_plots


def activation_matching_over_epochs_plot(
    df: pd.DataFrame,
    random_objectives: pd.DataFrame,
    run_key: str,
) -> plt.Figure:
    d = df[df["run_key"] == run_key]
    random_d = random_objectives[random_objectives["run_key"] == run_key]
    agg = (
        d.groupby("epoch")
        .agg(
            optimal_mean=("objective_optimal", "mean"),
            identity_mean=("objective_identity", "mean"),
        )
        .reset_index()
        .sort_values("epoch")
    )
    agg_random = (
        random_d.groupby("epoch")["objective_random"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("epoch")
    )

    fig, ax = plt.subplots(figsize=(1.5, 1.0))
    x = agg["epoch"].to_numpy()
    ax.plot(x, agg["optimal_mean"], color="C0", linestyle="-", linewidth=1.8)
    ax.plot(x, agg["identity_mean"], color="C1", linestyle="--", linewidth=1.8)
    ax.plot(x, agg_random["mean"], color="C2", linestyle=":", linewidth=1.8)
    ax.fill_between(
        x,
        agg_random["min"],
        agg_random["max"],
        color="C2",
        alpha=0.18,
        linewidth=0,
    )

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.minorticks_off()
    kappa = _extract_kappa_value(run_key)
    if kappa is not None:
        if np.isclose(kappa, 0.0):
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
        elif np.isclose(kappa, 1.0) or np.isclose(kappa, 2.0):
            ax.set_ylim(top=1.1)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
    _style_axes(ax, grid_axis="both")

    return fig


def load_activation_matching_data(
    path: str | Path,
    matching_mode: str = "post_activation_function",
    correlation_mode: str = "pearson_correlation_with_zero_for_constant",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(
        [
            {
                "run_key": d["run_key"],
                "epoch": d["epoch"],
                "layer": r["layer"],
                "objective_optimal": r["objectives"]["optimal"],
                "objective_identity": r["objectives"]["identity"],
            }
            for d in data
            for r in d["matching_results"]
            if r["matching_mode"] == matching_mode
            and r["correlation_mode"] == correlation_mode
        ]
    )
    random_objectives = pd.DataFrame(
        [
            {
                "run_key": d["run_key"],
                "epoch": d["epoch"],
                "layer": r["layer"],
                "objective_random": objective,
            }
            for d in data
            for r in d["matching_results"]
            if r["matching_mode"] == matching_mode
            and r["correlation_mode"] == correlation_mode
            for objective in r["objectives"]["random"]
        ]
    )
    return df, random_objectives


def make_activation_matching_plots_kappa_sweep(
    architecture: str,
    layers,
    outputs_path: str = DEFAULT_OUTPUTS_PATH,
    **kwargs,
) -> dict[str, plt.Figure]:
    df, random_objectives = load_activation_matching_data(
        f"{outputs_path}/activation-matching-results-{architecture}-kappa-sweep.json",
        matching_mode="post_activation_function",
        correlation_mode="pearson_correlation_with_zero_for_constant",
    )
    y_tick_step = 0.1 if architecture == "resnet" else None
    plots_by_layer = activation_matching_kappa_sweep_plot(
        df,
        random_objectives,
        y_tick_step=y_tick_step,
        **kwargs
    )
    for layer, fig in plots_by_layer.items():
        if layers is None or layer in layers:
            _save_figure(
                fig,
                _plot_dir() / f"activation-matching-kappas-{architecture}-{layer}.{FILE_EXTENSION}",
            )
    return plots_by_layer

def make_activation_matching_plots_sparsity_sweep(
    architecture: str,
    layers,
    kappa_filter=None,
    outputs_path: str = DEFAULT_OUTPUTS_PATH,
    filename_suffix="sparsity-sweep"
) -> dict[str, plt.Figure]:
    df, random_objectives = load_activation_matching_data(
        f"{outputs_path}/activation-matching-results-{architecture}-{filename_suffix}.json",
        matching_mode="post_activation_function",
        correlation_mode="pearson_correlation_with_zero_for_constant",
    )
    y_tick_step = 0.1 if architecture == "resnet" else None
    plots_by_layer = activation_matching_sparsity_sweep_plot(
        df,
        random_objectives,
        kappa_filter=kappa_filter,
        y_tick_step=y_tick_step,
    )
    for layer, fig in plots_by_layer.items():
        if layers is None or layer in layers:
            _save_figure(
                fig,
                _plot_dir() / f"activation-matching-sparsity-{architecture}-kappa={kappa_filter if kappa_filter is not None else "all"}-{layer}.{FILE_EXTENSION}",
            )
    return plots_by_layer


def make_activation_matching_plots_epoch_sweep(
    architecture: str,
    outputs_path: str = DEFAULT_OUTPUTS_PATH,
    matching_mode: str = "post_activation_function",
    correlation_mode: str = "pearson_correlation_with_zero_for_constant",
) -> None:
    df, random_objectives = load_activation_matching_data(
        f"{outputs_path}/activation-matching-results-{architecture}.json",
        matching_mode=matching_mode,
        correlation_mode=correlation_mode,
    )

    for run_key in df["run_key"].unique():
        for layer_name in sorted(df[df["run_key"] == run_key]["layer"].unique().tolist()) + [None]:
            fig = activation_matching_over_epochs_plot(
                df[df["layer"] == layer_name] if layer_name is not None else df,
                random_objectives[random_objectives["layer"] == layer_name]
                if layer_name is not None
                else random_objectives,
                run_key=run_key,
            )
            _save_figure(
                fig,
                _plot_dir()
                / f"activation-matching-epochs-{run_key}-{_safe_name(layer_name)}.{FILE_EXTENSION}",
            )


############################################################################################
#                                                                                          #
#              --------------------------------------------------------------              #
#              Pairwise realization costs between neurons (Mahalanobis/ridge)              #
#              --------------------------------------------------------------              #
#                                                                                          #
############################################################################################


# Utility function
def load_config(directory_or_checkpoint_path: str):
    path = Path(directory_or_checkpoint_path)
    config_dir = path.parent if path.is_file() else path
    
    with hydra.initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        return hydra.compose(config_name="config")


def load_df_ridge_mahalanobis(PARQUET_PATH_RIDGE, PARQUET_PATH_MAHALANOBIS, MAP_PATH_TO_RUN_KEY):
    df_ridge = (
        pl.read_parquet(PARQUET_PATH_RIDGE)
        .unnest("results")
        .unpivot(index="num_inner_descent_iterations", variable_name="run_key", value_name="results")
        .explode("results").unnest("results")
        .explode("neuron_realization_costs").unnest("neuron_realization_costs")
        .drop("objective_mse") # TODO fix MSE objective
        .with_columns(mse=(pl.col("objective_full") - pl.col("objective_weight_regularization"))*pl.col("beta"))
        # Adding in transposition costs
        .sort(*(groups := ["layer", "run_key", "beta"]), "id1", "id2")  # Sort the entire dataset into stacked matrices
        .with_columns(
            transposition_cost=(
                (c := pl.col(cost := "objective_weight_regularization")) 
                + c.sort_by(*groups, "id2", "id1")  # Transpose within each group
                - (d := pl.when(pl.col("id1") == pl.col("id2")).then(c)).max().over(*groups, "id1") # Broadcast diag across row per group
                - d.max().over(*groups, "id2") # Broadcast diag across col per group
            )
        )
        .rename({"transposition_cost": "transposition_cost_squared_ridge"})
        .rename({"objective_weight_regularization": "realization_cost_squared_ridge"})
        .with_columns(
            realization_cost_sqrt_ridge=pl.col("realization_cost_squared_ridge") ** 0.5,
            transposition_cost_sqrt_ridge=(abs(pl.col("transposition_cost_squared_ridge")) ** 0.5) * pl.col("transposition_cost_squared_ridge").sign(),
        )
    )
    df_mahalanobis = (
        pl.read_parquet(PARQUET_PATH_MAHALANOBIS)
        .with_columns(run_key=pl.col("run_directory").replace_strict(MAP_PATH_TO_RUN_KEY))
        .explode("realization_cost_results")
        .unnest("realization_cost_results")
        .with_columns(i=pl.int_ranges(pl.col("proj_center_distances").list.len()))
        .explode(["i"] + (["proj_center_distances", "transposition_cost_squared", "mahalanobis_distance_squared"]))
        .with_columns(j=pl.int_ranges(pl.col("proj_center_distances").list.len()))
        .explode(["j"] + (["proj_center_distances", "transposition_cost_squared", "mahalanobis_distance_squared"]))
        .with_columns(is_diag=(pl.col(name="i") == pl.col("j")), id1=pl.col("i"), id2=pl.col("j"))
        .rename({"transposition_cost_squared": "transposition_cost_squared_mahalanobis"})
        .rename({"mahalanobis_distance_squared": "realization_cost_squared_mahalanobis"})
        .with_columns(
            realization_cost_sqrt_mahalanobis=pl.col("realization_cost_squared_mahalanobis") ** 0.5,
            transposition_cost_sqrt_mahalanobis=(abs(pl.col("transposition_cost_squared_mahalanobis")) ** 0.5) * pl.col("transposition_cost_squared_mahalanobis").sign(),
        )
    )
    df = df_mahalanobis.join(df_ridge, on=["run_key", "id1", "id2", "layer"])
    return df


def load_data_gmm():
    Key = namedtuple("Key", field_names=["dataset_name", "symmetry", "kappa", "d_intrinsic", "hidden_dim",])
    gmm_paths_with_properties = {
        Key(
            f"{cfg.dataset.name}{'-' + str(cfg.dataset.parity_subspace_dataset.modulo_base) if cfg.dataset.name == "parity-subspace-dataset" else ''}",
            cfg.model.symmetry,
            cfg.model.mask_params.default.mask_constant if cfg.model.symmetry == 1 else None,
            (cfg.dataset.parity_subspace_dataset if cfg.dataset.name == "parity-subspace-dataset" else cfg.dataset.gaussian_subspace_dataset if cfg.dataset.name == "gaussian-subspace-dataset" else None).d_intrinsic,
            cfg.model.hidden_dim
        ): path
        for path in checkpoint_directories.checkpoint_directories_by_architecture["mlp-gaussian-mixture-symmetry1-untrained"].values()
        for cfg in [load_config(directory_or_checkpoint_path=path)]
    }

    df_gmm: pl.DataFrame = load_df_ridge_mahalanobis(
        "outputs/ridge-regression-realization-costs-gaussian-mixture-dim64.parquet",
        "outputs/realization-costs-mahalanobis-gaussian-mixture-dim64.parquet",
        {val.removesuffix("/"): key for key, val in checkpoint_directories.checkpoint_directories_by_architecture["mlp-gaussian-mixture-symmetry1-untrained"].items()}
    ).join(
        pl.DataFrame([
            {**properties._asdict(), "run_directory": path}
            for properties, path in gmm_paths_with_properties.items()
        ]),
        on="run_directory"
    )
    consec_pairs_df_gmm = df_gmm.filter(pl.col("id2") == pl.col(name="id1") + 1, pl.col(name="id1") % 2 == 0)
    return df_gmm, consec_pairs_df_gmm

def load_data_mnist():
    df_mnist: pl.DataFrame = load_df_ridge_mahalanobis(
        "outputs/ridge-regression-realization-costs-mnist-batchnorm.parquet",
        "outputs/realization-costs-mahalanobis-mnist-batchnorm.parquet",
        {val.removesuffix("/"): key for key, val in checkpoint_directories.checkpoint_directories_by_architecture["mlp-batchnorm"].items()}
    )
    consec_pairs_df_mnist = df_mnist.filter(pl.col("id2") == pl.col(name="id1") + 1, pl.col(name="id1") % 2 == 0)
    return df_mnist, consec_pairs_df_mnist

def filter_neuron_cost_df(df, run_key, layer, beta=None):
    return (
        (df_tmp := df.filter(
            *([pl.col("run_key") == run_key, pl.col("layer") == layer] + ([pl.col("beta") == beta] if beta is not None else []))
        )).sort("id1", "id2")
    )

def extract_cost_matrix_from_df(df, run_key, layer, beta=None, cost_name="transposition_cost_squared_mahalanobis"):
    df_tmp = filter_neuron_cost_df(df, run_key, layer, beta)
    return df_tmp[cost_name].to_numpy().reshape(m := (df_tmp["id1"].max() + 1), m)


def plot_neuron_pair_costs(data, out_filename=None, show_y_axis=True, show_x_axis=True, show_colorbar=True, colorbar_ticker_base=None, red_threshold=None):
    vmax = max(1, np.nanmax(data))
    vmin = min(-vmax*0.05, np.nanmin(data))
    
    # Fallback in case the data is completely constant
    if vmax <= vmin:
        vmax = vmin + 0.1
        
    total = vmax - vmin

    # Safe normalizer: applies the boundary offset FIRST, then strictly clips.
    # This guarantees we never spit out a value like 1.00001.
    def get_x(val, offset=0.0):
        norm = (val - vmin) / total
        return np.clip(norm + offset, 0.0, 1.0)

    yellow_start = min(0-(vmax-vmin)/1000, vmin*0.05)

    red_threshold = red_threshold if red_threshold is not None else 0.3
    red_threshold_default = 0.3
    factor = red_threshold/red_threshold_default
    print(red_threshold, get_x(-10*factor), get_x(50.0*factor))

    # The exact sequence of colors mapped to data milestones
    raw_stops = [
        (min(0, get_x(-10*factor)),            "#A2EF09"), # Green-yellow base
        (get_x(yellow_start, -1e-5),    "#C9D009"),
        (get_x(yellow_start),           "#ff0000"), # Transition to red
        (get_x(0.0),             "#ff0000"), # Hold pure red up to exactly 0.0
        (get_x(0.0, 1e-5),       "#ff0000"), # Immediate jump to purple > 0.0
        (get_x(3.0/100*factor),             "#bb0044"), # Dark Red
        (get_x(4.0/100*factor),             "#3b528b"), # Blue
        (get_x(25.0*factor),             "#2b3260"), # Darker blue at 25
        (max(1, get_x(50.0*factor)),  "#12124b")  # Dark blue at 50 
    ]

    # Enforce strict monotonicity (x1 < x2 < x3) without exceeding 1.0.
    # This acts as a firewall to prevent Matplotlib from defaulting to yellow.
    stops = []
    last_x = -1.0
    for x, color in raw_stops:
        if x <= last_x:
            x = last_x + 1e-9 # Micro-bump to keep Matplotlib's interpolator happy
        
        # Drop any points that the offset pushed beyond the colormap limits
        if x <= 1.0:
            stops.append((x, color))
            last_x = x
            
    # Matplotlib colormaps must end exactly at 1.0
    if stops[-1][0] < 1.0:
        stops.append((1.0, stops[-1][1]))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", stops)

    import ipywidgets as widgets
    plot_output = widgets.Output()
    with plot_output:
        fig, ax = plt.subplots(figsize=(1.8+0.6*show_colorbar+0.1*show_y_axis, 1.8-0.137*(not show_x_axis)), dpi=300)
        im = ax.imshow(data, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), aspect="equal", interpolation='none')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("neuron j")
        ax.set_ylabel("neuron i")
        ax.xaxis.set_label_position("top")

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            if colorbar_ticker_base is not None:
                cbar.ax.yaxis.set_major_locator(MultipleLocator(colorbar_ticker_base))
        if not show_y_axis:
            ax.get_yaxis().set_visible(False)
        if not show_x_axis:
            ax.get_xaxis().set_visible(False)
        
        plt.tight_layout()
        if out_filename is not None:
            fig.savefig(out_filename, format="pdf", bbox_inches="tight")
        plt.show()
    return plot_output


def plot_and_save_neuron_pair_cost(df, run_key, layer, beta, cost_name, **kwargs):
    output_file = lambda run_key, layer, beta, cost_name: f"plots/neuron-pair-cost_{run_key}_{layer}_{beta}_{cost_name}.pdf"
    red_threshold = filter_neuron_cost_df(df, run_key, layer, beta).filter(pl.col("i") == pl.col("j"))[cost_name.replace("transposition", "realization")].median()
    return plot_neuron_pair_costs(
        extract_cost_matrix_from_df(df, run_key, layer, beta, cost_name),
        output_file(run_key, layer, beta, cost_name),
        red_threshold=red_threshold,
        **kwargs
    )


def make_neuron_pair_cost_matrix_plots(outputs_path: str = DEFAULT_OUTPUTS_PATH) -> List[List]:
    df_gmm, _ = load_data_gmm()
    all_plots = [
        [
            plot_and_save_neuron_pair_cost(df_gmm, f"d-intr2_hidden-dim64_symmetry1_kappa{kappa}", "lins.0", 0.01, cost_name, show_colorbar=False, show_x_axis=(show_x_axis := (kappa=="0.0"))),
            plot_and_save_neuron_pair_cost(df_gmm, f"d-intr8_hidden-dim64_symmetry1_kappa{kappa}", "lins.0", 0.01, cost_name, show_y_axis=False, show_colorbar=False, show_x_axis=show_x_axis),
            plot_and_save_neuron_pair_cost(df_gmm, f"d-intr32_hidden-dim64_symmetry1_kappa{kappa}", "lins.0", 0.01, cost_name, show_y_axis=False, show_x_axis=show_x_axis,
                                           colorbar_ticker_base=(0.4 if kappa == "0.0" else None) # manual adjustment of colorbar ticker spacing
            )
        ]
        for kappa in ["0.0", "1.0"]
        for cost_name in ["transposition_cost_squared_mahalanobis", "transposition_cost_squared_ridge"]
    ]

    
    df_mnist, _ = load_data_mnist()
    all_plots.extend([
        [
            plot_and_save_neuron_pair_cost(
                df_mnist, run_key, layer, 0.01, cost_name,
                show_y_axis=(layer == "lins.0"), show_colorbar=(layer == "lins.2"),
                show_x_axis = ("mahalanobis" in cost_name)
                # range_y=([-0.2, 0.2] if run_key == "mlp_symmetry0" else None)
            )
            for layer in ["lins.0", "lins.1", "lins.2"] #, "lins.3"]
        ]
        for run_key in ["mlp_symmetry0", "mlp_symmetry1_kappa0", "mlp_symmetry1_kappa1",  "mlp_symmetry3_kappa1"]
        for cost_name in [
            "transposition_cost_squared_mahalanobis",
            # "transposition_cost_sqrt_mahalanobis",
            "transposition_cost_squared_ridge"
        ]
    ])

    return all_plots


####################################################################
#                                                                  #
#              -----------------------------------                 #
#              Hausdorff distances between classes                 #
#              -----------------------------------                 #
#                                                                  #
####################################################################


def load_hausdorff_distance_data(path: str | Path) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(
        [
            {
                "run_key": run_key,
                "layer": layer_result["layer"],
                "neuron1": int(neuron_pair.split(", ")[0]),
                "neuron2": int(neuron_pair.split(", ")[1]),
                "distance": distance,
            }
            for run_key, results in data["results"].items()
            for layer_result in results
            for neuron_pair, (distance, _, _) in layer_result["neuron_hausdorff_distances"].items()
        ]
    )


def hausdorff_distance_box_plot(df: pd.DataFrame) -> plt.Figure:
    plot_df = df[df["neuron1"] != df["neuron2"]].copy()
    ordered_labels = _ordered_run_labels(plot_df["run_key"])
    ordered_run_keys = [run_key for run_key, _ in ordered_labels]
    data = [
        plot_df[plot_df["run_key"] == current_run_key]["distance"].dropna().to_numpy()
        for current_run_key in ordered_run_keys
    ]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    boxplot = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for idx, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(_COLOR_SEQUENCE[idx])
        box.set_alpha(0.82)
        box.set_edgecolor("black")
        box.set_linewidth(0.75)

    for artist in boxplot["whiskers"] + boxplot["caps"] + boxplot["medians"]:
        artist.set_color("black")
        artist.set_linewidth(0.75)

    ax.set_xticks(range(1, len(ordered_labels) + 1))
    ax.set_xticklabels(
        [label for _, label in ordered_labels],
        rotation=18,
        ha="right",
    )
    ax.set_yscale("log")
    _style_axes(ax, grid_axis="y")
    ax.set_ylabel("Distance")

    return fig


def make_hausdorff_distance_plots(outputs_path: str = DEFAULT_OUTPUTS_PATH) -> dict[str, plt.Figure]:
    df = load_hausdorff_distance_data(f"{outputs_path}/function-classes.json")
    plots_by_layer = {
        layer_name: hausdorff_distance_box_plot(df[df["layer"] == layer_name] if layer_name is not None else df)
        for layer_name in sorted(df["layer"].unique()) + [None]
    }
    for layer_name, fig in plots_by_layer.items():
        _save_figure(
            fig,
            _plot_dir() / f"hausdorff-distance-epochs-{_safe_name(layer_name)}.{FILE_EXTENSION}",
        )
    return plots_by_layer


def main() -> None:
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-path", type=str, default=DEFAULT_OUTPUTS_PATH)
    args = parser.parse_args()

    make_lmc_plots_mlp(outputs_path=args.outputs_path)
    make_lmc_plots_resnet(outputs_path=args.outputs_path)

    make_activation_matching_plots_kappa_sweep(
        architecture="mlp",
        layers=None,
        outputs_path=args.outputs_path,
    )
    make_activation_matching_plots_kappa_sweep(
        architecture="resnet",
        layers=None,
        outputs_path=args.outputs_path,
    )

    make_activation_matching_plots_epoch_sweep(
        architecture="mlp",
        outputs_path=args.outputs_path,
        correlation_mode="pearson_correlation",
    )
    make_activation_matching_plots_epoch_sweep(
        architecture="resnet",
        outputs_path=args.outputs_path,
    )

    make_hausdorff_distance_plots(outputs_path=args.outputs_path)


if __name__ == "__main__":
    main()
