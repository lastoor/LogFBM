#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


COLORS = {
    "fp": "black",
    "lrp": "#F9B43F",
    "source_target": "purple",
}

PATH_EFFECTS = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


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
    ax.plot(
        source_xy[:, 0],
        source_xy[:, 1],
        color=COLORS["source_target"],
        linewidth=linewidth,
        linestyle="-",
        label="Source",
        path_effects=PATH_EFFECTS,
        zorder=4,
    )
    ax.plot(
        target_xy[:, 0],
        target_xy[:, 1],
        color=COLORS["source_target"],
        linewidth=linewidth,
        linestyle="-",
        label="Target",
        path_effects=PATH_EFFECTS,
        zorder=4,
    )
    ax.plot(
        fp[:, 0],
        fp[:, 1],
        color=COLORS["fp"],
        linewidth=linewidth,
        linestyle="-",
        label="FP",
        path_effects=PATH_EFFECTS,
        zorder=5,
    )
    ax.plot(
        lrp[:, 0],
        lrp[:, 1],
        color=COLORS["lrp"],
        linewidth=linewidth,
        linestyle="-.",
        label="LRP",
        path_effects=PATH_EFFECTS,
        zorder=5,
    )


def plot_logk_paths(data: dict[str, np.ndarray], *, path_linewidth: float) -> plt.Figure:
    _require_keys(data, ["x", "y", "logk", "fp", "lrp", "source_x", "target_x", "source_y_start", "source_y_end"], "logk_paths")
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    logk = np.asarray(data["logk"], dtype=float)
    fp = _validate_path(data["fp"], "fp")
    lrp = _validate_path(data["lrp"], "lrp")
    source_xy, target_xy = _build_geometry(data)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(logk, origin="lower", cmap="viridis", interpolation="nearest", extent=_extent(x, y))
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
    im = ax.imshow(speed, origin="lower", cmap="magma", interpolation="nearest", extent=_extent(x, y))
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


def _default_output_dir(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_figures")


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize figures from a generate_lrp_fp.py output .npz file.")
    parser.add_argument("--input", required=True, help="Input .npz path from generate_lrp_fp.py.")
    parser.add_argument("--output-dir", default=None, help="Output directory for saved figures. Defaults to <input_stem>_figures next to the input file.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--stream-density", type=float, default=1.0)
    parser.add_argument("--path-linewidth", type=float, default=2.0)
    parser.add_argument("--save-logk-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-head-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-travel-time", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(input_path)
    data = _load_npz(input_path)

    if not (args.save_logk_paths or args.save_head_velocity or args.save_travel_time):
        raise ValueError("At least one figure must be enabled for saving.")

    saved_paths: list[Path] = []
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

    print(f"Saved {len(saved_paths)} figure(s) to {output_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
