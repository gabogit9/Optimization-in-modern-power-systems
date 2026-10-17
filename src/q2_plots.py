"""Figures for Q2: load vs reference, sweep panels, heatmap, hourly response, model comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import _finish

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED, PV_FILL, GRID = "#52514e", "#f6d58a", "#e4e3df"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def plot_loads_vs_reference(loads: dict[str, np.ndarray], reference: np.ndarray, pv_available: np.ndarray,
                            title: str, save_to: Path) -> plt.Figure:
    """Hourly load of one or several runs (``label -> 24 values``) against l_ref and the available PV."""
    h = np.arange(len(reference))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(h, pv_available, step="mid", color=PV_FILL, alpha=0.6, lw=0, label="PV available")
    ax.step(h, reference, where="mid", color=MUTED, ls="--", lw=1.5, label="reference $l^{ref}$")
    for i, (label, load) in enumerate(loads.items()):
        ax.step(h, load, where="mid", color=SERIES[i % len(SERIES)], lw=2, label=label)
    ax.set(xlabel="hour", ylabel="kWh/h", title=title, xticks=h[::2])
    _style(ax)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    return _finish(fig, save_to)


def plot_sweep_metrics(df: pd.DataFrame, x: str, xlabel: str, title: str, thresholds: list[float],
                       save_to: Path) -> plt.Figure:
    """2x2 panels of the sweep metrics against c^L. Between two consecutive ``thresholds`` the metrics are
    constant, so each run is drawn as a flat segment over its regime (faint vertical lines = thresholds)."""
    df = df.sort_values(x)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    panels = [(axes[0, 0], [("energy_kWh", "daily energy")], "kWh"),
              (axes[0, 1], [("abs_deviation_kWh", "total |deviation|")], "kWh"),
              (axes[1, 0], [("procurement_cost", "procurement cost"), ("disutility", "disutility"),
                            ("total", "total")], "DKK"),
              (axes[1, 1], [("hours_deviating", "hours deviating"),
                            ("hours_dev_bound_binding", "hours deviation bound binding")], "hours")]
    edges = np.r_[df[x].min(), thresholds, df[x].max()]
    for ax, cols, unit in panels:
        for i, (col, label) in enumerate(cols):
            for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                v = df[col][(df[x] >= a) & (df[x] <= b)]
                if len(v):
                    ax.hlines(v.iloc[0], a, b, color=SERIES[i], lw=2, label=label if j == 0 else None)
            ax.plot(df[x], df[col], "o", ms=4, color=SERIES[i])
        for th in thresholds:
            ax.axvline(th, color=MUTED, lw=0.5, alpha=0.4)
        ax.set_ylabel(unit)
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11)
    return _finish(fig, save_to)


def plot_load_heatmap(loads: pd.DataFrame, reference: np.ndarray, xlabel: str, title: str,
                      save_to: Path) -> plt.Figure:
    """Load as a share of the reference (hours x swept values). Hours with l_ref = 0 are left blank."""
    with np.errstate(divide="ignore", invalid="ignore"):
        share = loads.to_numpy() / reference[:, None]
    share[reference <= 1e-9, :] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(share, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(loads.shape[1]), [f"{c:.3g}" for c in loads.columns], rotation=90, fontsize=7)
    ax.set(xlabel=xlabel, ylabel="hour", title=title)
    fig.colorbar(im, ax=ax, label="load / reference [-]")
    return _finish(fig, save_to)


def plot_hourly_response(curves: dict[str, pd.DataFrame], hours: list[int], save_to: Path) -> plt.Figure:
    """2.(d): load of selected hours vs the disutility coefficient, one panel per model.

    ``curves`` maps "panel title|x-axis label" to a DataFrame indexed by the coefficient, one column per hour.
    """
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8), sharey=True)
    for ax, (key, df) in zip(axes, curves.items()):
        title, xlabel = key.split("|")
        for i, t in enumerate(hours):
            ax.plot(df.index, df[t], "o-", ms=3, lw=2, color=SERIES[i], label=f"hour {t}")
        ax.set(xscale="log", xlabel=xlabel, title=title)
        _style(ax)
    axes[0].set_ylabel("load [kWh/h]")
    axes[0].legend(fontsize=8, frameon=False)
    return _finish(fig, save_to)


def plot_model_metrics(df: pd.DataFrame, save_to: Path) -> plt.Figure:
    """2.(d): horizontal bars of energy, total deviation and procurement cost per scenario."""
    metrics = [("energy_kWh", "daily energy [kWh]"), ("abs_deviation_kWh", "total |deviation| [kWh]"),
               ("procurement_cost", "procurement cost [DKK]")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.45 * len(df) + 1.6), sharey=True)
    y = np.arange(len(df))
    colors = [SERIES[{"linear": 0, "quadratic": 1, "Q1 utility": 2}[m]] for m in df["model"]]
    for ax, (col, label) in zip(axes, metrics):
        ax.barh(y, df[col], color=colors, height=0.7)
        for yi, v in zip(y, df[col]):
            ax.text(max(v, 0), yi, f" {v:.1f}", va="center", fontsize=7, color=MUTED)
        ax.set(title=label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, df["scenario"], fontsize=8)
    axes[0].invert_yaxis()
    return _finish(fig, save_to)
