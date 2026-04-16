#!/usr/bin/env python3
"""
Visualize outputs from tip_fbm.py simulations.

Handles two .npz types (auto-detected by key inspection):

  Single-run  (keys: state, elastic_grid, transport_grid, ...)
    → state.png      : FBM field | invasion state | overlaid backbones
    → backbone.png   : elastic vs transport backbone side-by-side

  Scaling study  (keys: L_values, mean_cluster_mass, ...)
    → scaling.png          : log-log M vs L for all four structures with fits
    → backbone_fraction.png: backbone/cluster and elastic/cluster mass ratios vs L

When multiple scaling-study files are passed:
    → h_comparison.png : fractal dimension Df vs H for each structure

Usage:
  # Single run:
  python scripts/visualize_tip_fbm.py --input outputs/tip_fbm/tip_single_L64_H0.7_modesite_seed42.npz

  # Scaling study:
  python scripts/visualize_tip_fbm.py --input outputs/tip_fbm/H0.7_modesite_std1.0/tip_fbm_H0.7_modesite.npz

  # H comparison (multiple scaling files):
  python scripts/visualize_tip_fbm.py \\
      --input outputs/tip_fbm/*/tip_fbm_H*_modesite.npz \\
      --save-h-comparison
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRUCTURE_LABELS = {
    'cluster_mass':       'SSC (cluster)',
    'elastic_mass':       'Elastic backbone',
    'transport_mass':     'Transport backbone',
    'minimal_path_length': 'Minimal path',
}

STRUCTURE_COLORS = {
    'cluster_mass':       '#1f77b4',
    'elastic_mass':       '#d62728',
    'transport_mass':     '#ff7f0e',
    'minimal_path_length': '#2ca02c',
}


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def _scalar(data: dict, key: str):
    v = data[key]
    return v.item() if hasattr(v, 'item') else v


def _is_single_run(data: dict) -> bool:
    return 'state' in data


def _is_scaling_study(data: dict) -> bool:
    return 'L_values' in data and 'mean_cluster_mass' in data


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def _fit_loglog(L: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    """Returns (Df, intercept_log) from log-log linear regression M ~ L^Df."""
    valid = np.isfinite(L) & np.isfinite(M) & (L > 0) & (M > 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    coeffs = np.polyfit(np.log(L[valid]), np.log(M[valid]), 1)
    return float(coeffs[0]), float(coeffs[1])


def _fit_fss(L: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    """
    Finite-size scaling fit (Eq. 10 of Knackstedt et al. 2000):
      c1 + Df * M^w = c2 * L^(w * Df)
    Rearranged: M = [(c2 * L^(w*Df) - c1) / Df]^(1/w)
    Returns (Df, w).
    """
    valid = np.isfinite(L) & np.isfinite(M) & (L > 0) & (M > 0)
    if valid.sum() < 3:
        return np.nan, np.nan

    def model(L_, c1, c2, Df, w):
        inner = (c2 * L_ ** (w * Df) - c1) / Df
        return np.maximum(inner, 1e-10) ** (1.0 / w)

    try:
        popt, _ = curve_fit(
            model, L[valid], M[valid],
            p0=[1.0, 1.0, 1.89, 1.0],
            bounds=([-np.inf, -np.inf, 1.0, 0.01], [np.inf, np.inf, 2.0, 5.0]),
            maxfev=10000,
        )
        return float(popt[2]), float(popt[3])
    except RuntimeError:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Single-run figures
# ---------------------------------------------------------------------------

def plot_state(data: dict) -> plt.Figure:
    """
    3-panel: FBM field / invasion state / backbones overlaid.
    Requires keys: state, elastic_grid, transport_grid, H, mode.
    """
    state = np.asarray(data['state'], dtype=int)
    elastic = np.asarray(data['elastic_grid'], dtype=int)
    transport = np.asarray(data['transport_grid'], dtype=int)
    H = _scalar(data, 'H')
    mode = str(_scalar(data, 'mode'))
    L = state.shape[0]

    cluster_mass = int(_scalar(data['cluster_mass'])) if 'cluster_mass' in data else np.sum(state == 1)
    elastic_mass = int(np.sum(elastic))
    transport_mass = int(np.sum(transport))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"TIP  H={H}  mode={mode}  L={L}", fontsize=13)

    # Panel 1: invasion state
    cmap_state = mcolors.ListedColormap(['#cce5ff', '#ff6b6b', '#2b2d42'])
    im0 = axes[0].imshow(state.T, cmap=cmap_state, vmin=0, vmax=2, origin='lower')
    axes[0].set_title(f"Invasion State\ncluster={cluster_mass}, trapped={np.sum(state==2)}")
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar0.ax.set_yticklabels(['Defender', 'Invader', 'Trapped'])
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    # Panel 2: elastic backbone on cluster background
    bg = np.where(state == 1, 0.4, 0.0)  # cluster in grey
    axes[1].imshow(bg.T, cmap='Greys', vmin=0, vmax=1, origin='lower', alpha=0.5)
    axes[1].imshow(
        np.ma.masked_where(elastic.T == 0, elastic.T),
        cmap='Blues', vmin=0, vmax=1, origin='lower',
    )
    axes[1].set_title(f"Elastic Backbone\n(minimal path, {elastic_mass} sites)")
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    # Panel 3: transport backbone on cluster background
    axes[2].imshow(bg.T, cmap='Greys', vmin=0, vmax=1, origin='lower', alpha=0.5)
    axes[2].imshow(
        np.ma.masked_where(transport.T == 0, transport.T),
        cmap='Reds', vmin=0, vmax=1, origin='lower',
    )
    axes[2].set_title(f"Transport Backbone\n(with loops, {transport_mass} sites)")
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

    fig.tight_layout()
    return fig


def plot_backbone_comparison(data: dict) -> plt.Figure:
    """
    Side-by-side elastic vs transport backbone with mass annotations.
    """
    elastic = np.asarray(data['elastic_grid'], dtype=int)
    transport = np.asarray(data['transport_grid'], dtype=int)
    state = np.asarray(data['state'], dtype=int)

    loop_only = (transport - elastic).clip(0)  # sites in transport but not elastic

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Backbone Structures", fontsize=13)

    for ax, grid, title, cmap in [
        (axes[0], elastic,   f"Elastic (shortest path)\n{int(np.sum(elastic))} sites", 'Blues'),
        (axes[1], transport, f"Transport (includes loops)\n{int(np.sum(transport))} sites", 'Reds'),
    ]:
        bg = np.where(state == 1, 0.3, 0.0)
        ax.imshow(bg.T, cmap='Greys', vmin=0, vmax=1, origin='lower', alpha=0.4)
        ax.imshow(np.ma.masked_where(grid.T == 0, grid.T), cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scaling-study figures
# ---------------------------------------------------------------------------

def plot_scaling(data: dict) -> plt.Figure:
    """
    2×2 log-log M vs L plots for cluster, elastic, transport, minimal path.
    Includes error bars (std), standard log-log fit line, and FSS fit annotation.
    """
    L = np.asarray(data['L_values'], dtype=float)
    H = _scalar(data, 'H')
    mode = str(_scalar(data, 'mode'))
    scale_mode = str(_scalar(data['scale_mode'])) if 'scale_mode' in data else 'std'
    scale_val = float(_scalar(data['scale_value'])) if 'scale_value' in data else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        f"TIP Finite-Size Scaling  H={H}  mode={mode}  "
        f"scale={scale_mode}={scale_val}",
        fontsize=12,
    )

    keys = list(STRUCTURE_LABELS.keys())
    for ax, key in zip(axes.flat, keys):
        mean = np.asarray(data[f'mean_{key}'], dtype=float)
        std  = np.asarray(data[f'std_{key}'], dtype=float)

        valid = np.isfinite(mean) & (mean > 0)
        color = STRUCTURE_COLORS[key]

        ax.errorbar(
            L[valid], mean[valid], yerr=std[valid],
            fmt='o', color=color, capsize=4, label='data', zorder=3,
        )

        # Log-log fit line
        Df_std, intercept = _fit_loglog(L, mean)
        if np.isfinite(Df_std):
            L_fit = np.logspace(np.log10(L[valid].min()), np.log10(L[valid].max()), 100)
            M_fit = np.exp(intercept) * L_fit ** Df_std
            ax.plot(L_fit, M_fit, '--', color=color, alpha=0.7,
                    label=f'fit $D_f$={Df_std:.3f}')

        # FSS fit annotation
        Df_fss, w = _fit_fss(L, mean)
        if np.isfinite(Df_fss):
            ax.text(0.05, 0.95,
                    f"FSS: $D_f$={Df_fss:.3f}, $\\omega$={w:.3f}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$L$'); ax.set_ylabel('$M$')
        ax.set_title(STRUCTURE_LABELS[key])
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    return fig


def plot_backbone_fraction(data: dict) -> plt.Figure:
    """
    Backbone/cluster and elastic/cluster mass ratios vs L.
    For H > 0.5 (first-order transition) these ratios should approach 1.
    """
    L = np.asarray(data['L_values'], dtype=float)
    H = _scalar(data, 'H')
    mode = str(_scalar(data, 'mode'))

    cluster = np.asarray(data['mean_cluster_mass'], dtype=float)
    elastic = np.asarray(data['mean_elastic_mass'], dtype=float)
    transport = np.asarray(data['mean_transport_mass'], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(L, transport / np.maximum(cluster, 1), 'o-',
            color=STRUCTURE_COLORS['transport_mass'], label='Transport bb / cluster')
    ax.plot(L, elastic / np.maximum(cluster, 1), 's--',
            color=STRUCTURE_COLORS['elastic_mass'], label='Elastic bb / cluster')
    ax.axhline(1.0, color='grey', linestyle=':', alpha=0.6, label='compact limit')

    ax.set_xscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('Backbone / Cluster mass ratio')
    ax.set_title(
        f"Backbone Fraction  H={H}  mode={mode}\n"
        r"$\to 1$: compact (first-order, $H>0.5$)  "
        r"$< 1$: fractal ($H<0.5$)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.2)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-file H comparison
# ---------------------------------------------------------------------------

def plot_h_comparison(datasets: list[tuple[float, dict]]) -> plt.Figure:
    """
    Fractal dimension Df vs H for each structure.
    datasets: list of (H_value, data_dict) from scaling study files.
    """
    datasets = sorted(datasets, key=lambda x: x[0])
    H_vals = np.array([h for h, _ in datasets])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fractal Dimension vs Hurst Exponent", fontsize=13)

    for key, label, color in [
        ('cluster_mass',       STRUCTURE_LABELS['cluster_mass'],       STRUCTURE_COLORS['cluster_mass']),
        ('elastic_mass',       STRUCTURE_LABELS['elastic_mass'],       STRUCTURE_COLORS['elastic_mass']),
        ('transport_mass',     STRUCTURE_LABELS['transport_mass'],     STRUCTURE_COLORS['transport_mass']),
        ('minimal_path_length', STRUCTURE_LABELS['minimal_path_length'], STRUCTURE_COLORS['minimal_path_length']),
    ]:
        Df_std_vals, Df_fss_vals = [], []
        for _, data in datasets:
            L = np.asarray(data['L_values'], dtype=float)
            mean = np.asarray(data[f'mean_{key}'], dtype=float)
            Df_s, _ = _fit_loglog(L, mean)
            Df_f, _ = _fit_fss(L, mean)
            Df_std_vals.append(Df_s)
            Df_fss_vals.append(Df_f)

        Df_std_vals = np.array(Df_std_vals)
        Df_fss_vals = np.array(Df_fss_vals)

        axes[0].plot(H_vals, Df_std_vals, 'o-', color=color, label=label)
        axes[1].plot(H_vals, Df_fss_vals, 'o-', color=color, label=label)

    for ax, title in [
        (axes[0], 'Standard log-log fit'),
        (axes[1], 'Finite-size scaling fit'),
    ]:
        ax.axvline(0.5, color='grey', linestyle=':', alpha=0.6, label='H=0.5 (uncorrelated)')
        ax.axhline(2.0, color='lightgrey', linestyle='--', alpha=0.6, label='$D_f=2$ (compact)')
        ax.set_xlabel('Hurst exponent $H$')
        ax.set_ylabel('Fractal dimension $D_f$')
        ax.set_title(title)
        ax.set_ylim(0.8, 2.2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize tip_fbm.py .npz output files."
    )
    p.add_argument(
        '--input', nargs='+', required=True,
        help='One or more .npz files produced by tip_fbm.py.',
    )
    p.add_argument(
        '--output-dir', default=None,
        help='Output directory for figures. Defaults to <input_stem>_figures/ next to the input file.',
    )
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--format', choices=('png', 'pdf', 'svg'), default='png', dest='fmt')

    # Single-run figure toggles
    p.add_argument('--save-state', action=argparse.BooleanOptionalAction, default=True,
                   help='Save state + backbone overlay figure (single-run .npz only).')
    p.add_argument('--save-backbone', action=argparse.BooleanOptionalAction, default=True,
                   help='Save elastic vs transport backbone figure (single-run .npz only).')

    # Scaling-study figure toggles
    p.add_argument('--save-scaling', action=argparse.BooleanOptionalAction, default=True,
                   help='Save log-log scaling plot (scaling-study .npz only).')
    p.add_argument('--save-backbone-fraction', action=argparse.BooleanOptionalAction, default=True,
                   help='Save backbone/cluster fraction plot (scaling-study .npz only).')

    # Multi-file option
    p.add_argument('--save-h-comparison', action='store_true',
                   help='Save Df vs H comparison plot (requires multiple scaling-study files).')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in args.input]
    for p in input_paths:
        if not p.is_file():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    single_runs   = []  # (path, data)
    scaling_studies = []  # (H, path, data)

    for path in input_paths:
        data = _load(path)
        if _is_single_run(data):
            single_runs.append((path, data))
        elif _is_scaling_study(data):
            H = float(_scalar(data, 'H'))
            scaling_studies.append((H, path, data))
        else:
            print(f"WARNING: {path.name}: unrecognised .npz format, skipping.", file=sys.stderr)

    # --- single-run files ---
    for path, data in single_runs:
        out_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else path.with_name(f"{path.stem}_figures")
        )
        print(f"\nSingle-run: {path.name}  →  {out_dir}/")

        if args.save_state:
            _save(plot_state(data), out_dir / f"state.{args.fmt}", args.dpi)
        if args.save_backbone:
            _save(plot_backbone_comparison(data), out_dir / f"backbone.{args.fmt}", args.dpi)

    # --- scaling study files ---
    for H, path, data in scaling_studies:
        out_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else path.with_name(f"{path.stem}_figures")
        )
        print(f"\nScaling study H={H}: {path.name}  →  {out_dir}/")

        if args.save_scaling:
            _save(plot_scaling(data), out_dir / f"scaling.{args.fmt}", args.dpi)
        if args.save_backbone_fraction:
            _save(plot_backbone_fraction(data), out_dir / f"backbone_fraction.{args.fmt}", args.dpi)

    # --- multi-file H comparison ---
    if args.save_h_comparison:
        if len(scaling_studies) < 2:
            print("WARNING: --save-h-comparison requires at least 2 scaling-study files.", file=sys.stderr)
        else:
            datasets = [(H, data) for H, _, data in scaling_studies]
            # Use output-dir if given, else put next to the first file
            ref_path = scaling_studies[0][1]
            out_dir = (
                Path(args.output_dir).expanduser().resolve()
                if args.output_dir
                else ref_path.parent
            )
            print(f"\nH comparison ({len(datasets)} files)  →  {out_dir}/")
            _save(plot_h_comparison(datasets), out_dir / f"h_comparison.{args.fmt}", args.dpi)


if __name__ == '__main__':
    main()
