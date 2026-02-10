from __future__ import annotations
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import plotly.colors
import plotly.graph_objects as go

from IPython.display import display
import ipywidgets as widgets
import plotly.graph_objects as go
import argparse


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
    rows = []

    for row_figs in figures:
        fig_widgets = []
        for fig in row_figs:
            if fig is None:
                continue

            # Create FigureWidget from existing figure's data and layout
            fw = go.FigureWidget(data=fig.data, layout=fig.layout)
            fw.update_layout(width=width, height=height)
            fig_widgets.append(fw)

        rows.append(widgets.HBox(fig_widgets))

    grid = widgets.VBox(rows)
    display(grid)


####################################################################
#                                                                  #
#                 -------------------------------                  #
#                      Standard LMC Experiments                    #
#                 -------------------------------                  #
#                                                                  #
####################################################################


def load_lmc_df(path):
    with open(path, "r") as f:
        # with open("outputs/lmc-resnet-aligned-unaligned-37steps-8modelpairs.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(
        [
            {
                "run_key": entry["run_key"],
                "model1_index": entry["model1_index"],
                "lambda": interpolation_lambda,
                "model2_index": entry["model2_index"],
                "train_loss": entry[mode]["train_loss"][i],
                "val_loss": entry[mode]["val_loss"][i],
                "test_loss": entry[mode]["test_loss"][i],
                "train_accuracy": entry[mode]["train_accuracy"][i],
                "val_accuracy": entry[mode]["val_accuracy"][i],
                "test_accuracy": entry[mode]["test_accuracy"][i],
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
    return df


def lmc_plot(df, alignment_mode, criterion, range_y, dataset_name):
    df = df[df["mode"] == alignment_mode]
    agg = (
        df.groupby(["run_key", "mode", "lambda"])
        .agg(
            **{
                f"{crit}_mean": (crit, "mean")
                for crit in [
                    "train_loss",
                    "val_loss",
                    "test_loss",
                    "train_accuracy",
                    "val_accuracy",
                    "test_accuracy",
                ]
            }
        )
        .reset_index()
        .sort_values(["run_key", "mode", "lambda"])
    )

    mode_to_description = {
        "interpolation_results_activation_aligned": "Aligned LMC (activation matching)",
        "interpolation_results_weight_aligned": "Aligned LMC (weight matching)",
        "interpolation_results_unaligned": "Unaligned LMC",
    }
    architecture_to_description = {
        "mlp_symmetry0": "MLP",
        "mlp_symmetry1_kappa0": "<i>W</i>-MLP/κ=0",
        "mlp_symmetry1_kappa1": "<i>W</i>-MLP/κ=1",
        "mlp_symmetry3_kappa1": "<i>N</i>-MLP/κ=1",
        "resnet_symmetry0": "ResNet",
        "resnet_symmetry1_kappa0": "<i>W</i>-ResNet/κ=0",
        "resnet_symmetry1_kappa2": "<i>W</i>-ResNet/κ=2",
        "resnet_symmetry3_kappa2": "<i>N</i>-ResNet/κ=2",
        # "resnet_symmetry0": "ResNet20 (8×)",
        # "resnet_symmetry1_kappa0": "W-asym. ResNet20 (8×), κ=0",
        # "resnet_symmetry1_kappa2": "W-asym. ResNet20 (8×), κ=2",
        # "resnet_symmetry3_kappa2": "Noise-asym. ResNet20 (8×), κ=2",
    }
    architecture_to_description = {
        k: v
        for k, v in architecture_to_description.items()
        if k in agg["run_key"].tolist()
    }
    print(architecture_to_description)
    criterion_to_description = {
        "train_loss": "Train loss",
        "train_accuracy": "Train accuracy",
    }

    agg["run_key"] = agg["run_key"].replace(architecture_to_description)
    # color_map = [
    #     '#EE9B00',
    #     '#06A77D',
    #     '#56D77D',
    #     # '#005F73',  # related to line1
    #     '#BB3E03'
    # ]
    color_scheme = plotly.colors.qualitative.Set1
    color_map = [color_scheme[1], color_scheme[0], color_scheme[4], color_scheme[2]]
    color_map = {
        architecture: color
        for architecture, color in zip(architecture_to_description.values(), color_map)
    }
    line_dash_map = ["solid", "solid", "dot", "dash"]
    line_dash_map = {
        architecture: line_dash
        for architecture, line_dash in zip(
            architecture_to_description.values(), line_dash_map
        )
    }

    fig = px.line(
        agg,
        x="lambda",
        y=f"{criterion}_mean",
        color="run_key",
        line_dash="run_key",
        range_y=range_y,  # , title=f"{mode_to_description[alignment_mode]} / {dataset_name}",
        labels={"run_key": "", **architecture_to_description},
        color_discrete_map=color_map,
        line_dash_map=line_dash_map,
    )
    fig.update_layout(
        dict(
            xaxis=dict(title=None, tickprefix="λ=", tickfont=dict(size=19)),
            # yaxis=dict(title=criterion_to_description[criterion],ticksuffix='%'),
            yaxis=dict(title=None, ticksuffix="%", tickfont=dict(size=19)),
            legend=dict(
                x=0.0, y=0.35, bgcolor="rgba(255, 255, 255, 0.9)", orientation="v"
            ),
            margin=dict(l=0, r=20, t=10, b=65),
        )
    )
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        font=dict(family="CMU Serif"),
        legend_font_size=14,
        width=400,
        height=220,
    )

    fig.update_traces(line=dict(width=3.0))

    style_figure(fig)
    return fig


def make_lmc_plots_mlp(outputs_path="outputs", save_plots=True, display=True):
    # MLP plots
    df = load_lmc_df(
        f"{outputs_path}/lmc-mlp-aligned-unaligned-37steps-8modelpairs.json"
    )
    important_runs = [
        "mlp_symmetry0",
        "mlp_symmetry1_kappa0",
        "mlp_symmetry1_kappa1",
        "mlp_symmetry3_kappa1",
    ]
    df = df[df["run_key"].isin(important_runs)]
    plots = {
        mode: lmc_plot(
            df, mode, "train_accuracy", range_y=[93, 100.1], dataset_name="MNIST"
        )
        for mode in df["mode"].unique()
    }
    if save_plots:
        for mode, fig in plots.items():
            fig.write_image(f"plots/lmc-mlp-{mode}.{FILE_EXTENSION}")
            fig.write_html(f"plots/lmc-mlp-{mode}.html")

    if display:
        next(iter(plots.values())).show()
        return plot_grid([[p] for p in plots.values()])
    return plots


def make_lmc_plots_resnet(outputs_path="outputs", save_plots=True, display=True):
    # ResNet plots
    df = load_lmc_df(
        f"{outputs_path}/lmc-resnet-aligned-unaligned-37steps-8modelpairs.json"
    )
    plots = {
        mode: lmc_plot(
            df, mode, "train_accuracy", range_y=[0, 101], dataset_name="CIFAR-10"
        )
        for mode in df["mode"].unique()
    }
    if save_plots:
        for mode, fig in plots.items():
            fig.write_image(f"plots/lmc-resnet-{mode}.{FILE_EXTENSION}")
            fig.write_html(f"plots/lmc-resnet-{mode}.html")

    if display:
        return plot_grid([[p] for p in plots.values()])
    return plots


####################################################################
#                                                                  #
#                 -------------------------------                  #
#                 Activation Matching Experiments                  #
#                 -------------------------------                  #
#                                                                  #
####################################################################


def activation_matching_kappa_sweep_plot(df, random_objectives, epoch=100):
    import pandas as pd
    import plotly.graph_objects as go
    import re

    # Assuming df and random_objectives DataFrames are available
    # df has columns: run_key, model1_index, model2_index, epoch, layer, objective_optimal, objective_identity, ...
    # random_objectives has columns: run_key, model1_index, model2_index, matching_mode, correlation_mode, objective_random

    # Extract kappa from run_key
    def extract_kappa(run_key):
        match = re.search(r"kappa([\d.]+)", run_key)
        return float(match.group(1)) if match else None

    df["kappa"] = df["run_key"].apply(extract_kappa)
    random_objectives["kappa"] = random_objectives["run_key"].apply(extract_kappa)

    # Filter to epoch epoch
    epoch_df = df[df["epoch"] == epoch].copy()

    # Get unique layer values
    layers = sorted(epoch_df["layer"].unique()) + [None]

    out_plots = {}
    # Create separate plots for each layer
    for layer in layers:
        layer_df = (
            epoch_df[epoch_df["layer"] == layer] if layer is not None else epoch_df
        ).copy()
        random_objectives_df = random_objectives[(random_objectives["epoch"] == epoch)]
        if layer is not None:
            random_objectives[(random_objectives["layer"] == layer)]

        # Aggregate by kappa (average over model index pairs)
        agg_df = (
            layer_df.groupby("kappa")
            .agg({"objective_optimal": "mean", "objective_identity": "mean"})
            .reset_index()
        )

        # Aggregate random objectives by kappa (get mean, min, max)
        agg_random = (
            random_objectives_df.groupby("kappa")["objective_random"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        # Create the plot
        fig = go.Figure()

        # Add optimal line
        fig.add_trace(
            go.Scatter(
                x=agg_df["kappa"],
                y=agg_df["objective_optimal"],
                mode="lines+markers",
                name="Optimal",
                line=dict(color="rgb(31, 119, 180)", dash="solid", width=2),
                marker=dict(size=8),
            )
        )

        # Add random mean line
        fig.add_trace(
            go.Scatter(
                x=agg_random["kappa"],
                y=agg_random["mean"],
                mode="lines+markers",
                name="Random",
                line=dict(color="rgb(44, 160, 44)", dash="dot", width=2),
                marker=dict(size=8),
            )
        )

        # Add shaded region for random min/max
        fig.add_trace(
            go.Scatter(
                x=agg_random["kappa"].tolist() + agg_random["kappa"].tolist()[::-1],
                y=agg_random["max"].tolist() + agg_random["min"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(44, 160, 44, 0.2)",
                line=dict(color="rgba(44, 160, 44,.5)"),
                showlegend=False,
                name="Random (min/max)",
                hoverinfo="skip",
            )
        )

        # Add identity line
        fig.add_trace(
            go.Scatter(
                x=agg_df["kappa"],
                y=agg_df["objective_identity"],
                mode="lines+markers",
                name="Identity",
                line=dict(color="rgb(255, 127, 14)", dash="dash", width=3),
                marker=dict(size=8),
            )
        )

        # Update layout
        fig.update_layout(
            # title=f'Objective Values vs Kappa (Layer: {layer}, Epoch: {epoch})',
            xaxis_title=None,
            # yaxis_title='Activation matching obj.',
            yaxis_title=None,
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(
                type="log",
                tickmode="array",
                tickvals=agg_df["kappa"].tolist(),
                ticktext=[
                    f"κ={int(k) if int(k) == k else k}"
                    for k in agg_df["kappa"].tolist()
                ],
            ),
            width=300,
            height=240,
            legend=dict(
                x=0.0, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)", orientation="h"
            ),
            margin=dict(l=0, r=20, t=0, b=65),
        )

        style_figure(fig)
        out_plots[layer or "alllayers"] = fig
        # fig.show()
        # Or save: fig.write_html(f'objectives_kappa_layer_{layer}.html')
    return out_plots


def activation_matching_over_epochs_plot(
    df, run_key, log_y_axis=True, show_min_max=False
):
    """
    For a single run_key:
    - Average over (model1_index, model2_index) per epoch
    - Plot mean ± std for each objective
    """
    d = df[df["run_key"] == run_key]

    # Aggregate per epoch
    agg = (
        d.groupby("epoch")
        .agg(
            optimal_mean=("objective_optimal", "mean"),
            optimal_min=("objective_optimal", "min"),
            optimal_max=("objective_optimal", "max"),
            optimal_std=("objective_optimal", "std"),
            identity_mean=("objective_identity", "mean"),
            identity_min=("objective_identity", "min"),
            identity_max=("objective_identity", "max"),
            identity_std=("objective_identity", "std"),
            random_mean=("objective_random_mean", "mean"),
            random_min=("objective_random_mean", "min"),
            random_max=("objective_random_mean", "max"),
            random_std=("objective_random_mean", "std"),
        )
        .reset_index()
        .sort_values("epoch")
    )
    # print(agg)

    fig = go.Figure()

    def add_mean_min_max(x, mean, min, max, name, color, dash, line_width):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash, width=line_width),
            )
        )

        if show_min_max:
            # fig.add_trace(go.Scatter(
            #     x=list(x) + list(x[::-1]),
            #     y=list(max) + list(min)[::-1],
            #     fill="toself",
            #     fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.2)"),
            #     line=dict(color="rgba(255,255,255,0)"),
            #     hoverinfo="skip",
            #     showlegend=False,
            # ))
            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(max),
                    line=dict(
                        color=color[:-1].replace("rgb", "rgba") + ",10)", width=0.5
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(min),
                    line=dict(
                        color=color[:-1].replace("rgb", "rgba") + ",10)", width=0.5
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    add_mean_min_max(
        agg["epoch"],
        agg["optimal_mean"],
        agg["optimal_min"],
        agg["optimal_max"],
        "Optimal",
        "rgb(31, 119, 180)",
        "solid",
        4,
    )

    add_mean_min_max(
        agg["epoch"],
        agg["random_mean"],
        agg["random_min"],
        agg["random_max"],
        "Random",
        "rgb(44, 160, 44)",
        "dot",
        4,
    )

    add_mean_min_max(
        agg["epoch"],
        agg["identity_mean"],
        agg["identity_min"],
        agg["identity_max"],
        "Identity",
        "rgb(255, 127, 14)",
        "dash",
        6,
    )

    fig.add_trace(
        go.Scatter(
            x=list(agg["epoch"]) + list(agg["epoch"])[::-1],
            y=list(agg["random_mean"] + agg["random_std"])
            + list(agg["random_mean"] - agg["random_std"])[::-1],
            line=dict(color="rgba(44, 160, 44, 20)", width=0.5),
            fill="toself",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        # title=f"{run_key} / {layer_name}: Matching, averaged over 8 model pairs",
        title="",
        # xaxis_title="Epoch",
        # yaxis_title="Activation match. objective",
        # yaxis_title="$\\mathcal{L}_{\\text{act}}$",
        template="plotly_white",
        legend_title="Objective",
        width=250,
        height=160,
        # legend=dict(x=0.3,y=0.98,bgcolor='rgba(255, 255, 255, 0.8)',orientation="h"),
        showlegend=False,
        margin=dict(l=0, r=50, t=20, b=50),
    )
    fig.update_xaxes(automargin=True)
    if log_y_axis:
        fig.update_layout(yaxis_type="log")

    style_figure(fig)

    return fig


def load_activation_matching_data(
    path,
    matching_mode="post_activation_function",
    correlation_mode="pearson_correlation_with_zero_for_constant",
):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(
        [
            {
                "run_key": d["run_key"],
                "model1_index": d["model1_index"],
                "model2_index": d["model2_index"],
                "epoch": d["epoch"],
                "layer": r["layer"],
                "objective_optimal": r["objectives"]["optimal"],
                "objective_identity": r["objectives"]["identity"],
                "objective_random_mean": r["objectives"]["random_mean"],
                "objective_random_std": r["objectives"]["random_std"],
                "matching_mode": r["matching_mode"],
                "correlation_mode": r["correlation_mode"],
            }
            for d in data
            for r in d["matching_results"]
        ]
    )
    random_objectives = pd.DataFrame(
        [
            {
                "run_key": d["run_key"],
                "model1_index": d["model1_index"],
                "model2_index": d["model2_index"],
                "matching_mode": r["matching_mode"],
                "correlation_mode": r["correlation_mode"],
                "epoch": d["epoch"],
                "layer": r["layer"],
                "objective_random": o,
            }
            for d in data
            for r in d["matching_results"]
            for o in r["objectives"]["random"]
        ]
    )

    print("Available layers:", df["layer"].unique())
    df = df[
        (df["matching_mode"] == matching_mode)
        & (df["correlation_mode"] == correlation_mode)
    ]
    random_objectives = random_objectives[
        (random_objectives["matching_mode"] == matching_mode)
        & (random_objectives["correlation_mode"] == correlation_mode)
    ]
    print("Selected layers:", df["layer"].unique())
    print("Available runs:", df["run_key"].unique())
    return df, random_objectives


def make_activation_matching_plots_kappa_sweep(
    architecture, layers, outputs_path="outputs", save_plots=True, display=True
):
    df, random_objectives = load_activation_matching_data(
        f"{outputs_path}/activation-matching-results-{architecture}-kappa-sweep.json",
        matching_mode="post_activation_function",
        correlation_mode="pearson_correlation_with_zero_for_constant",
    )

    kappa_sweep_plots_by_layer = activation_matching_kappa_sweep_plot(
        df, random_objectives
    )
    if save_plots:
        for layer, fig in kappa_sweep_plots_by_layer.items():
            if layers is None or layer in layers:
                fig.write_image(
                    f"plots/activation-matching-kappas-{architecture}-{layer}.{FILE_EXTENSION}"
                )
                # fig.write_html(f"plots/activation-matching-kappas-{architecture}-{layer}.html")

    if display:
        next(iter(kappa_sweep_plots_by_layer.values())).show()
        return plot_grid([[p] for p in kappa_sweep_plots_by_layer.values()])


def make_activation_matching_plots_epoch_sweep(
    architecture,
    outputs_path="outputs",
    save_plots=True,
    display=True,
    matching_mode="post_activation_function",
    correlation_mode="pearson_correlation_with_zero_for_constant",
):
    df, _ = load_activation_matching_data(
        f"{outputs_path}/activation-matching-results-{architecture}.json",
        matching_mode=matching_mode,
        correlation_mode=correlation_mode,
    )

    epoch_sweep_plots_by_layer_and_run = {
        (run_key, layer_name): activation_matching_over_epochs_plot(
            df[df["layer"] == layer_name] if layer_name is not None else df,
            run_key=run_key,
            log_y_axis=False,
            show_min_max=False,
        )
        for run_key in df["run_key"].unique()
        for layer_name in sorted(
            df[df["run_key"] == run_key]["layer"].unique().tolist()
        )
        + [None]
    }

    if save_plots:
        for (run_key, layer_name), fig in epoch_sweep_plots_by_layer_and_run.items():
            # if layers is None or layer in layers:
            fig.write_image(
                f"plots/activation-matching-epochs-{run_key}-{layer_name if layer_name is not None else "alllayers"}.{FILE_EXTENSION}"
            )
            # fig.write_html(f"plots/activation-matching-kappas-{architecture}-{layer}.html")

    if display:
        next(iter(epoch_sweep_plots_by_layer_and_run.values())).show()
        return plot_grid(
            [[p] for _, p in zip(range(2), epoch_sweep_plots_by_layer_and_run.values())]
        )


####################################################################
#                                                                  #
#              -----------------------------------                 #
#              Hausdorff distances between classes                 #
#              -----------------------------------                 #
#                                                                  #
####################################################################


def load_hausdorff_distance_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(
        [
            {
                "run_key": run_key,
                "layer": layer_result["layer"],
                "neuron1": int(neuron_pair.split(", ")[0]),
                "neuron2": int(neuron_pair.split(", ")[1]),
                "neuron_pair": neuron_pair,
                "distance": distance,
                "scale1": scale1,
                "scale2": scale2,
            }
            for run_key, results in data["results"].items()
            for layer_result in results
            for neuron_pair, [distance, scale1, scale2] in layer_result[
                "neuron_hausdorff_distances"
            ].items()
        ]
    )
    return df


def hausdorff_distance_box_plot(df, run_key, layer_name, log_y_axis=True):
    run_df = df[(df["layer"] == layer_name)] if layer_name is not None else df.copy()
    run_df = run_df[run_df["neuron1"] != run_df["neuron2"]]
    if run_key is not None:
        run_df = df[(df["run_key"] == run_key)]

    architecture_to_description = {
        "mlp_symmetry0": "MLP",
        "mlp_symmetry1_kappa0": "<i>W</i>-MLP/κ=0",
        "mlp_symmetry1_kappa1": "<i>W</i>-MLP/κ=1",
        "mlp_symmetry3_kappa1": "<i>N</i>-MLP/κ=1",
    }
    run_df["run_key"] = run_df["run_key"].replace(architecture_to_description)
    color_scheme = plotly.colors.qualitative.Set1
    color_map = [color_scheme[1], color_scheme[0], color_scheme[4], color_scheme[2]]
    color_map = {
        architecture: color
        for architecture, color in zip(architecture_to_description.values(), color_map)
    }

    fig = px.box(run_df, y="distance", color="run_key", color_discrete_map=color_map)

    fig.update_layout(
        # title=f"{run_key if run_key is not None else 'Hausdorff distances between function classes'}: {layer_name}",
        yaxis_type="log" if log_y_axis else None,
        margin=dict(l=0, r=20, t=70, b=0),
    )

    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        yaxis=dict(
            title=None,
            tickfont=dict(size=17.5),
        ),
        font=dict(family="CMU Serif"),
        legend_font_size=14,
        width=300,
        height=300,
        legend=dict(
            x=0.0, y=1.0, bgcolor="rgba(255, 255, 255, 0.9)", orientation="v", title=""
        ),
    )

    # fig.update_traces(line=dict(width=3.))

    style_figure(fig)

    return fig


def make_hausdorff_distance_plots(
    outputs_path="outputs", save_plots=True, display=True
):
    df = load_hausdorff_distance_data(f"{outputs_path}/function-classes.json")

    hausdorff_distance_boxplots_by_layer = {
        layer_name: hausdorff_distance_box_plot(
            df[df["layer"] == layer_name] if layer_name is not None else df,
            layer_name=layer_name,
            run_key=None,
            log_y_axis=True,
        )
        for layer_name in sorted(df["layer"].unique()) + [None]
    }

    if save_plots:
        for layer_name, fig in hausdorff_distance_boxplots_by_layer.items():
            # if layers is None or layer in layers:
            fig.write_image(
                f"plots/hausdorff-distance-epochs-{layer_name if layer_name is not None else "alllayers"}.{FILE_EXTENSION}"
            )
            # fig.write_html(f"plots/activation-matching-kappas-{architecture}-{layer}.html")

    if display:
        next(iter(hausdorff_distance_boxplots_by_layer.values())).show()
        return plot_grid(
            [
                [p]
                for _, p in zip(range(2), hausdorff_distance_boxplots_by_layer.values())
            ]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-path", type=str, default="outputs/")
    args = parser.parse_args()
    make_lmc_plots_mlp(outputs_path=args.outputs_path, display=False)
    make_lmc_plots_resnet(outputs_path=args.outputs_path, display=False)

    make_activation_matching_plots_kappa_sweep(
        architecture="mlp", layers=None, outputs_path=args.outputs_path, display=False
    )
    make_activation_matching_plots_kappa_sweep(
        architecture="resnet",
        layers=None,
        outputs_path=args.outputs_path,
        display=False,
    )

    make_activation_matching_plots_epoch_sweep(
        architecture="mlp",
        outputs_path=args.outputs_path,
        display=False,
        correlation_mode="pearson_correlation",
    )
    make_activation_matching_plots_epoch_sweep(
        architecture="resnet", outputs_path=args.outputs_path, display=False
    )

    make_hausdorff_distance_plots(
        outputs_path=args.outputs_path, save_plots=True, display=False
    )


FILE_EXTENSION = "pdf"

if __name__ == "__main__":
    main()
