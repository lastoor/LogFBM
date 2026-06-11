#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.colors import LogNorm
import cmcrameri.cm as cmc
import numpy as np


COLORS = {
    "fp": "white",
    "lrp": "white",
    "source_target": "#00F0FF",
}

PATH_EFFECTS = [pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()]

def configure_style():
    import glob
    import matplotlib.font_manager as fm

    family: str = "CMU Sans Serif"
    mathtext_fontset: str = "cm"
    unicode_minus: bool = False
    cmu_font_glob: list = ["/home1/binhaoli/.local/share/fonts/cmun*.ttf", "/Users/binhaoli/Library/Fonts/cmun*.ttf"]

    for paths in cmu_font_glob:
        for path in glob.glob(paths):
            fm.fontManager.addfont(path)
            
    plt.rcParams["font.family"] = family
    plt.rcParams["mathtext.fontset"] = mathtext_fontset
    plt.rcParams["axes.unicode_minus"] = unicode_minus

    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 12


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_npz_dir(path: Path) -> list[tuple[Path, dict[str, np.ndarray]]]:
    npz_paths = sorted(child for child in path.iterdir() if child.is_file() and child.suffix == ".npz")
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in input directory: {path}")
    return [(npz_path, _load_npz(npz_path)) for npz_path in npz_paths]


_COMPARISON_KEYS = (
    "fp_travel_time",
    "lrp_pseudo_travel_time",
    "dh",
    "fp_compute_time_sec",
    "lrp_compute_time_sec",
    "fp_total_time_sec",
    "lrp_total_time_sec",
)


def _load_npz_keys(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in keys if key in data.files}


def _load_comparison_dataset(path: Path) -> list[tuple[Path, dict[str, np.ndarray]]]:
    npz_paths = sorted(child for child in path.iterdir() if child.is_file() and child.suffix == ".npz")
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in input directory: {path}")
    return [(npz_path, _load_npz_keys(npz_path, _COMPARISON_KEYS)) for npz_path in npz_paths]


def _require_keys(data: dict[str, np.ndarray], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required keys for {context}: {missing_str}")


def _scalar(data: dict[str, np.ndarray], key: str):
    value = data[key]
    if np.ndim(value) == 0:
        return value.item()
    if np.size(value) == 1:
        return np.ravel(value)[0].item()
    raise ValueError(f"Expected scalar value for {key}, got shape {np.shape(value)}")


def _build_geometry(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    source_x = float(_scalar(data, "source_x"))
    target_x = float(_scalar(data, "target_x"))
    source_y_start = int(_scalar(data, "source_y_start"))
    source_y_end = int(_scalar(data, "source_y_end"))

    source_rows = np.arange(source_y_start, source_y_end, dtype=float)
    source_xy = np.column_stack([np.full(source_rows.size, source_x, dtype=float), source_rows])

    y_coords = np.asarray(data["y"], dtype=float)
    target_rows = y_coords.astype(float)
    target_xy = np.column_stack([np.full(target_rows.size, target_x, dtype=float), target_rows])
    return source_xy, target_xy


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("x and y must contain at least two coordinates")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return (float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), float(y[0] - 0.5 * dy), float(y[-1] + 0.5 * dy))


def _validate_path(path: np.ndarray, name: str) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2); got {path.shape}")
    if path.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 points; got {path.shape[0]}")
    return path


def _add_path_overlays(
    ax: plt.Axes,
    *,
    fp: np.ndarray,
    lrp: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    linewidth: float,
) -> None:
    line, = ax.plot(
        source_xy[:, 0],
        source_xy[:, 1],
        color=COLORS["source_target"],
        linewidth=linewidth,
        linestyle="--",
        label="Source",
        zorder=4,
        alpha=1,
    )
    line.set_path_effects(PATH_EFFECTS)
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")

    line, = ax.plot(
        target_xy[:, 0],
        target_xy[:, 1],
        color=COLORS["source_target"],
        linewidth=linewidth,
        linestyle="-.",
        label="Target",
        zorder=4,
        alpha=1,
    )
    line.set_path_effects(PATH_EFFECTS)
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")

    line, = ax.plot(
        fp[:, 0],
        fp[:, 1],
        color=COLORS["fp"],
        linewidth=linewidth,
        linestyle="-",
        label="FP",
        zorder=5,
    )
    line.set_path_effects(PATH_EFFECTS)
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")

    line, = ax.plot(
        lrp[:, 0],
        lrp[:, 1],
        color=COLORS["lrp"],
        linewidth=linewidth,
        linestyle=":",
        label="LRP",
        path_effects=PATH_EFFECTS,
        zorder=5,
    )
    line.set_path_effects(PATH_EFFECTS)
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")


def plot_logk_paths(data: dict[str, np.ndarray], *, path_linewidth: float) -> plt.Figure:
    _require_keys(data, ["x", "y", "logk", "fp", "lrp", "source_x", "target_x", "source_y_start", "source_y_end"], "logk_paths")
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    logk = np.asarray(data["logk"], dtype=float)
    fp = _validate_path(data["fp"], "fp")
    lrp = _validate_path(data["lrp"], "lrp")
    source_xy, target_xy = _build_geometry(data)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(logk, origin="lower", cmap=cmc.batlowW_r, interpolation="nearest", extent=_extent(x, y))
    _add_path_overlays(ax, fp=fp, lrp=lrp, source_xy=source_xy, target_xy=target_xy, linewidth=path_linewidth)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log K")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Log-Permeability With FP and LRP")
    ax.legend(loc="best", facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    return fig


def plot_head_velocity(data: dict[str, np.ndarray], *, path_linewidth: float, stream_density: float) -> plt.Figure:
    _require_keys(data, ["x", "y", "vx", "vy", "fp", "lrp", "source_x", "target_x", "source_y_start", "source_y_end"], "head_velocity")
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    vx = np.asarray(data["vx"], dtype=float)
    vy = np.asarray(data["vy"], dtype=float)
    speed = np.hypot(vx, vy)
    fp = _validate_path(data["fp"], "fp")
    lrp = _validate_path(data["lrp"], "lrp")
    source_xy, target_xy = _build_geometry(data)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(speed, origin="lower", cmap=cmc.batlowW_r, interpolation="nearest", extent=_extent(x, y))
    ax.streamplot(x, y, vx, vy, color=(0.0, 0.0, 0.0, 0.35), linewidth=0.8, density=stream_density, arrowsize=0.7)
    _add_path_overlays(ax, fp=fp, lrp=lrp, source_xy=source_xy, target_xy=target_xy, linewidth=path_linewidth)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\mathbf{v}|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Magnitude and Streamlines")
    ax.legend(loc="best", facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    return fig


def plot_travel_time_profiles(data: dict[str, np.ndarray]) -> plt.Figure:
    _require_keys(
        data,
        ["fp_arc_length", "fp_travel_time", "lrp_arc_length", "lrp_pseudo_travel_time"],
        "travel_time_profiles",
    )
    fp_s = np.asarray(data["fp_arc_length"], dtype=float)
    fp_t = np.asarray(data["fp_travel_time"], dtype=float)
    lrp_s = np.asarray(data["lrp_arc_length"], dtype=float)
    lrp_t = np.asarray(data["lrp_pseudo_travel_time"], dtype=float)
    if fp_s.ndim != 1 or fp_t.ndim != 1 or lrp_s.ndim != 1 or lrp_t.ndim != 1:
        raise ValueError("Travel-time arrays must be one-dimensional")
    if fp_s.size < 2 or lrp_s.size < 2:
        raise ValueError("Travel-time arrays must contain at least 2 samples")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fp_s, fp_t, color=COLORS["fp"], linewidth=2.0, label=f"FP final = {fp_t[-1]:.4g}")
    ax.plot(lrp_s, lrp_t, color=COLORS["lrp"], linewidth=2.0, linestyle="-.", label=f"LRP pseudo final = {lrp_t[-1]:.4g}")
    ax.set_xlabel("Arc length")
    ax.set_ylabel("Cumulative travel time")
    ax.set_title("Travel-Time Profiles Along Paths")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    return fig


def plot_travel_time_comparison(dataset: list[tuple[Path, dict[str, np.ndarray]]]) -> plt.Figure:
    x_values: list[float] = []
    y_values: list[float] = []
    skipped = 0
    for npz_path, data in dataset:
        _require_keys(data, ["fp_travel_time", "lrp_pseudo_travel_time", "dh"], f"travel_time_comparison ({npz_path.name})")
        fp_t = np.asarray(data["fp_travel_time"], dtype=float)
        lrp_t = np.asarray(data["lrp_pseudo_travel_time"], dtype=float)
        if fp_t.ndim != 1 or lrp_t.ndim != 1 or fp_t.size < 1 or lrp_t.size < 1:
            raise ValueError(f"Travel-time arrays must be one-dimensional and non-empty in {npz_path}")
        t_ref = abs(float(_scalar(data, "dh")))
        if not np.isfinite(t_ref) or t_ref <= 0.0:
            raise ValueError(f"Dimensionless travel-time reference |dh| must be finite and positive in {npz_path}")
        x_val = float(fp_t[-1]) / t_ref
        y_val = float(lrp_t[-1]) / t_ref
        if not np.isfinite(x_val) or not np.isfinite(y_val) or x_val <= 0.0 or y_val <= 0.0:
            skipped += 1
            continue
        x_values.append(x_val)
        y_values.append(y_val)

    if skipped:
        print(
            f"[warn] skipped {skipped} non-finite/non-positive travel-time points out of {len(dataset)} files",
            file=sys.stderr,
        )
    if not x_values:
        raise ValueError("No finite positive travel-time points available for comparison plot.")

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    xy_min = float(min(np.min(x_arr), np.min(y_arr)))
    xy_max = float(max(np.max(x_arr), np.max(y_arr)))
    lim_min = xy_min * 0.9
    lim_max = xy_max * 1.1

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x_arr, y_arr, color="#444444", s=36, alpha=0.5, linewidths=0)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="0.4", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$t_{\min}^{\text{FP}} / t_{\text{ref}}$")
    ax.set_ylabel(r"$t_{\min}^{\text{LRP}} / t_{\text{ref}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="major", alpha=0.8)
    # ax.grid(True, which="minor", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_computation_time_profiles(data: dict[str, np.ndarray]) -> plt.Figure:
    _require_keys(
        data,
        [
            "head_solve_time_sec",
            "velocity_solve_time_sec",
            "fp_compute_time_sec",
            "lrp_compute_time_sec",
            "fp_total_time_sec",
            "lrp_total_time_sec",
        ],
        "computation_time_profiles",
    )
    head_solve_time_sec = float(_scalar(data, "head_solve_time_sec"))
    velocity_solve_time_sec = float(_scalar(data, "velocity_solve_time_sec"))
    fp_compute_time_sec = float(_scalar(data, "fp_compute_time_sec"))
    lrp_compute_time_sec = float(_scalar(data, "lrp_compute_time_sec"))
    fp_total_time_sec = float(_scalar(data, "fp_total_time_sec"))
    lrp_total_time_sec = float(_scalar(data, "lrp_total_time_sec"))

    labels = [
        "FP only",
        "FP + head + velocity",
        "LRP only",
        "LRP + head + velocity",
    ]
    values = [
        fp_compute_time_sec,
        fp_total_time_sec,
        lrp_compute_time_sec,
        lrp_total_time_sec,
    ]
    colors = [
        COLORS["fp"],
        "#6E6E6E",
        COLORS["lrp"],
        "#D98A00",
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4g}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Computation time (s)")
    ax.set_title("Computation Time Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(
        0.02,
        0.98,
        f"head = {head_solve_time_sec:.4g}s, velocity = {velocity_solve_time_sec:.4g}s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )
    fig.tight_layout()
    return fig


def _collect_computation_times(
    dataset: list[tuple[Path, dict[str, np.ndarray]]],
    components: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (x_self, y_self, x_total, y_total) based on the requested components."""
    required: list[str] = []
    if components in ("self", "both"):
        required += ["fp_compute_time_sec", "lrp_compute_time_sec"]
    if components in ("total", "both"):
        required += ["fp_total_time_sec", "lrp_total_time_sec"]
    x_self: list[float] = []
    y_self: list[float] = []
    x_total: list[float] = []
    y_total: list[float] = []
    for npz_path, data in dataset:
        _require_keys(data, required, f"computation_time_comparison ({npz_path.name})")
        if components in ("self", "both"):
            x_self.append(float(_scalar(data, "fp_compute_time_sec")))
            y_self.append(float(_scalar(data, "lrp_compute_time_sec")))
        if components in ("total", "both"):
            x_total.append(float(_scalar(data, "fp_total_time_sec")))
            y_total.append(float(_scalar(data, "lrp_total_time_sec")))
    return x_self, y_self, x_total, y_total


def plot_computation_time_comparison_scatter(
    dataset: list[tuple[Path, dict[str, np.ndarray]]],
    *,
    components: str = "both",
) -> plt.Figure:
    x_self, y_self, x_total, y_total = _collect_computation_times(dataset, components)

    all_values = x_self + y_self + x_total + y_total
    line_max = max(max(all_values) * 1.05, 1e-12)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if x_self:
        mean_ratio_self = float(np.mean(y_self) / np.mean(x_self))
        ax.scatter(x_self, y_self, color=COLORS["fp"], s=40, alpha=0.8,
                   label="Self: " + rf"$\mathbb{{E}}[t_{{\text{{LRP}}}}] / \mathbb{{E}}[t_{{\text{{FP}}}}] = {mean_ratio_self:.4g}$")
    if x_total:
        mean_ratio_total = float(np.mean(y_total) / np.mean(x_total))
        ax.scatter(x_total, y_total, color=COLORS["lrp"], s=40, marker="s", alpha=0.8,
                   label="With velocity field solving: " + rf"$\mathbb{{E}}[t_{{\text{{LRP}}}}] / \mathbb{{E}}[t_{{\text{{FP}}}}] = {mean_ratio_total:.4g}$")
    ax.plot([0.0, line_max], [0.0, line_max], color="0.4", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlabel(r"$t_{\text{compute}}^{\text{FP}}$ (s)")
    ax.set_ylabel(r"$t_{\text{compute}}^{\text{LRP}}$ (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", facecolor="white", edgecolor="black", framealpha=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_computation_time_comparison_histogram(
    dataset: list[tuple[Path, dict[str, np.ndarray]]],
    *,
    components: str = "both",
) -> plt.Figure:
    x_self, y_self, x_total, y_total = _collect_computation_times(dataset, components)

    series: list[tuple[np.ndarray, str, str]] = []
    if x_self:
        ratios = np.asarray(y_self, dtype=float) / np.asarray(x_self, dtype=float)
        series.append((ratios, "#444444", rf"Self ($\mu={np.mean(ratios):.3g}$)"))
    if x_total:
        ratios = np.asarray(y_total, dtype=float) / np.asarray(x_total, dtype=float)
        series.append((ratios, "#111111", rf"With velocity field solving ($\mu={np.mean(ratios):.3g}$)"))

    all_ratios = np.concatenate([s[0] for s in series])
    bins = np.logspace(np.log10(np.nanmin(all_ratios) * 0.9), np.log10(np.nanmax(all_ratios) * 1.1), 30)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for ratios_arr, color, label in series:
        weights = np.ones_like(ratios_arr, dtype=float) / ratios_arr.size
        ax.hist(
            ratios_arr,
            bins=bins,
            weights=weights,
            color=color,
            alpha=0.6,
            edgecolor=color,
            linewidth=0,
        ) #, label=label)
        mean_ratio = np.mean(ratios_arr)
        ax.axvline(mean_ratio, color="#000000", linestyle="--", linewidth=2.0, label=f"Mean: {mean_ratio:.3g}")
    ax.set_xlabel(r"$t_{\text{compute}}^{\text{LRP}} / t_{\text{compute}}^{\text{FP}}$")
    ax.set_ylabel("Frequency")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="major", alpha=0.8)
    ax.grid(True, which="minor", alpha=0.6)
    ax.legend(loc="best", facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    return fig


def plot_computation_time_comparison(
    dataset: list[tuple[Path, dict[str, np.ndarray]]],
    *,
    style: str = "scatter",
    components: str = "both",
) -> list[tuple[str, plt.Figure]]:
    """Return a list of (stem_suffix, figure) pairs (one per style requested)."""
    styles = ("scatter", "histogram") if style == "both" else (style,)
    results: list[tuple[str, plt.Figure]] = []
    for s in styles:
        suffix = f"_{s}" if style == "both" else ""
        if s == "histogram":
            fig = plot_computation_time_comparison_histogram(dataset, components=components)
        else:
            fig = plot_computation_time_comparison_scatter(dataset, components=components)
        results.append((suffix, fig))
    return results


def plot_first_arrival_plume(
    data: dict[str, np.ndarray], *, path_linewidth: float, stream_density: float
) -> plt.Figure:
    _require_keys(
        data,
        ["x", "y", "logk", "vx", "vy", "fp", "lrp", "plume_x", "plume_y",
         "source_x", "target_x", "source_y_start", "source_y_end"],
        "first_arrival_plume",
    )
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    logk = np.asarray(data["logk"], dtype=float)
    vx = np.asarray(data["vx"], dtype=float)
    vy = np.asarray(data["vy"], dtype=float)
    plume_x = np.asarray(data["plume_x"], dtype=float).ravel()
    plume_y = np.asarray(data["plume_y"], dtype=float).ravel()
    fp = _validate_path(data["fp"], "fp")
    lrp = _validate_path(data["lrp"], "lrp")
    source_xy, target_xy = _build_geometry(data)
    mask = np.isfinite(plume_x) & np.isfinite(plume_y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.exp(logk), norm=LogNorm(), origin="lower", cmap=cmc.batlowW_r, interpolation="nearest",
              extent=_extent(x, y))
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$K$")
    cbar.locator = LogLocator(base=10)
    cbar.formatter = LogFormatterMathtext(base=10)
    cbar.update_ticks()

    ax.streamplot(
        x,
        y,
        vx,
        vy,
        color='#222222',
        linewidth=0.8,
        density=stream_density,
        arrowsize=0.5,
        zorder=2
    )
    line, = ax.plot(plume_x[mask], plume_y[mask], color="#00F0FF",
               linewidth=2, label="Plume", zorder=3, alpha=1)
    line.set_path_effects([pe.Stroke(linewidth=3.0, foreground="black"), pe.Normal()])
    line.set_solid_capstyle("round")
    line.set_dash_capstyle("round")
    _add_path_overlays(ax, fp=fp, lrp=lrp, source_xy=source_xy,
                       target_xy=target_xy, linewidth=path_linewidth)
    ax.set_xlabel(r"$x / \delta$")
    ax.set_ylabel(r"$y / \delta$")
    # ax.set_title("FP First Arrival Plume")
    ax.legend(loc="lower left", facecolor="white", edgecolor="black", framealpha=0.7)
    fig.tight_layout()
    return fig


def _default_output_dir(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_figures")


def _default_output_dir_for_dir(input_dir: Path) -> Path:
    return input_dir


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _flag_was_explicitly_set(argv: list[str], name: str) -> bool:
    return f"--{name}" in argv or f"--no-{name}" in argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize figures from a generate_lrp_fp.py output .npz file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Input .npz path from generate_lrp_fp.py for per-file figures.")
    group.add_argument("--input-dir", help="Input directory containing many .npz files for aggregate comparison figures.")
    parser.add_argument("--output-dir", default=None, help="Output directory for saved figures.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--stream-density", type=float, default=1.0)
    parser.add_argument("--path-linewidth", type=float, default=2.0)
    parser.add_argument("--save-logk-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-head-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-travel-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-computation-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-first-arrival-plume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-travel-time-comparison", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-computation-time-comparison", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--computation-time-comparison-style",
        choices=("scatter", "histogram", "both"),
        default="scatter",
        help="Plot style for computation-time comparison: scatter (default), histogram of LRP/FP ratios, or both (saves two separate figures).",
    )
    parser.add_argument(
        "--computation-time-comparison-components",
        choices=("self", "total", "both"),
        default="both",
        help="Which timing components to include: self (path-finding only), total (including velocity-field solving), or both (default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    argv = sys.argv[1:]
    per_file_flags = [
        "save-logk-paths",
        "save-head-velocity",
        "save-travel-time",
        "save-computation-time",
        "save-first-arrival-plume",
    ]
    aggregate_flags = [
        "save-travel-time-comparison",
        "save-computation-time-comparison",
    ]
    saved_paths: list[Path] = []
    if args.input is not None:
        for flag in aggregate_flags:
            if not _flag_was_explicitly_set(argv, flag):
                setattr(args, flag.replace("-", "_"), False)
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        if args.save_travel_time_comparison or args.save_computation_time_comparison:
            raise ValueError("travel_time_comparison and computation_time_comparison require --input-dir.")
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(input_path)
        data = _load_npz(input_path)

        if not (args.save_logk_paths or args.save_head_velocity or args.save_travel_time
                or args.save_computation_time or args.save_first_arrival_plume):
            raise ValueError("At least one per-file figure must be enabled for --input mode.")

        if args.save_logk_paths:
            fig = plot_logk_paths(data, path_linewidth=args.path_linewidth)
            path = output_dir / f"logk_paths.{args.format}"
            _save_figure(fig, path, args.dpi)
            saved_paths.append(path)

        if args.save_head_velocity:
            fig = plot_head_velocity(data, path_linewidth=args.path_linewidth, stream_density=args.stream_density)
            path = output_dir / f"head_velocity.{args.format}"
            _save_figure(fig, path, args.dpi)
            saved_paths.append(path)

        if args.save_travel_time:
            fig = plot_travel_time_profiles(data)
            path = output_dir / f"travel_time_profiles.{args.format}"
            _save_figure(fig, path, args.dpi)
            saved_paths.append(path)

        if args.save_computation_time:
            fig = plot_computation_time_profiles(data)
            path = output_dir / f"computation_time_profiles.{args.format}"
            _save_figure(fig, path, args.dpi)
            saved_paths.append(path)

        if args.save_first_arrival_plume:
            if "plume_x" not in data:
                print("[warn] plume_x not found in data; skipping first-arrival-plume figure")
            else:
                fig = plot_first_arrival_plume(
                    data,
                    path_linewidth=args.path_linewidth,
                    stream_density=args.stream_density,
                )
                path = output_dir / f"first_arrival_plume.{args.format}"
                _save_figure(fig, path, args.dpi)
                saved_paths.append(path)
    else:
        for flag in per_file_flags:
            if not _flag_was_explicitly_set(argv, flag):
                setattr(args, flag.replace("-", "_"), False)
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if args.save_logk_paths or args.save_head_velocity or args.save_travel_time or args.save_computation_time or args.save_first_arrival_plume:
            raise ValueError("Per-file figures require --input. Use only comparison flags with --input-dir.")
        if not (args.save_travel_time_comparison or args.save_computation_time_comparison):
            raise ValueError("At least one aggregate comparison figure must be enabled for --input-dir mode.")

        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir_for_dir(input_dir)
        dataset = _load_comparison_dataset(input_dir)

        if args.save_travel_time_comparison:
            fig = plot_travel_time_comparison(dataset)
            path = output_dir / f"travel_time_comparison.{args.format}"
            _save_figure(fig, path, args.dpi)
            saved_paths.append(path)

        if args.save_computation_time_comparison:
            for suffix, fig in plot_computation_time_comparison(
                dataset,
                style=args.computation_time_comparison_style,
                components=args.computation_time_comparison_components,
            ):
                path = output_dir / f"computation_time_comparison{suffix}.{args.format}"
                _save_figure(fig, path, args.dpi)
                saved_paths.append(path)

    print(f"Saved {len(saved_paths)} figure(s) to {output_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
