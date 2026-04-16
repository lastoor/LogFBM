#!/usr/bin/env python3
"""
TIP simulation with FBM long-range correlations.

Implements Knackstedt, Sahimi & Sheppard (2000, Phys. Rev. E):
  - Trapping Invasion Percolation on correlated FBM landscapes
  - Site TIP (imbibition) and Bond TIP (drainage)
  - Elastic and transport backbone extraction
  - Finite-size scaling to estimate fractal dimensions

Usage examples:
  # Single run for testing:
  python scripts/tip_fbm.py --single-run --L 64 --H 0.7 --mode site --visualize

  # Scaling study, fix std across L values (default):
  python scripts/tip_fbm.py --L 64 128 256 --H 0.7 --mode site --n-real 20 --n-workers 8

  # Scaling study, fix c directly (same c for all L):
  python scripts/tip_fbm.py --L 64 128 256 --H 0.7 --mode site --fix-c 0.5 --n-workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _add_search_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_flow_modules():
    """Import grf_generation from FlowSimulation (respects FLOWSIM_PATH env var)."""
    workspace_root = Path(__file__).resolve().parent.parent
    flowsim_path = Path(
        os.environ.get("FLOWSIM_PATH", str(workspace_root.parent / "FlowSimulation"))
    ).expanduser().resolve()

    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )

    os.environ["LD_LIBRARY_PATH"] = (
        "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    _add_search_path(flowsim_path)

    try:
        from simulation import grf_generation as sim_grf
    except Exception as exc:
        raise RuntimeError(
            "Failed to import FlowSimulation/simulation/grf_generation. "
            "Ensure FlowSimulation is available and its dependencies are installed."
        ) from exc

    return sim_grf


def _import_tip_modules():
    """Import TIPEngine, BackboneExtractor, DataAnalyzer from src/tip.py."""
    src_path = Path(__file__).resolve().parent.parent / "src"
    _add_search_path(src_path)

    try:
        from tip import TIPEngine, BackboneExtractor, DataAnalyzer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import from src/tip.py. Ensure {src_path} is correct."
        ) from exc

    return TIPEngine, BackboneExtractor, DataAnalyzer


# ---------------------------------------------------------------------------
# Field generation
# ---------------------------------------------------------------------------

class FieldGenerator:
    """
    Generates FBM correlated random fields for TIP simulations.

    The field values are used directly as site/bond threshold values.
    Long-range correlations are controlled by the Hurst exponent H:
      H > 0.5  -> persistent (positive) correlations -> compact clusters
      H < 0.5  -> antipersistent (negative) correlations -> fractal clusters
      H = 0.5  -> uncorrelated (standard IP universality class)

    Scale control (mutually exclusive):
      fix_c=None   -> compute c from std_target via compute_c(L, H, std_target).
                      Keeps the marginal std consistent across all L values.
      fix_c=float  -> use this c directly for all L values (L-independent).
                      Useful when comparing runs at fixed intrinsic correlation
                      strength rather than fixed output variance.
    """

    def __init__(
        self,
        L: int,
        H: float,
        std_target: float = 1.0,
        fix_c: float | None = None,
        minpadding: int = 0,
        sim_grf=None,
    ):
        if not (0 < H < 1):
            raise ValueError(f"Hurst exponent H must be in (0, 1), got {H}")

        self.L = L
        self.H = H
        self.minpadding = minpadding

        if sim_grf is None:
            sim_grf = _import_flow_modules()
        self._sim_grf = sim_grf

        alpha = 2.0 * H
        if fix_c is not None:
            c = float(fix_c)
            self.scale_mode = 'fix_c'
            self.scale_value = c
        else:
            c = sim_grf.compute_c(L, H=H, std_target=std_target)
            self.scale_mode = 'std'
            self.scale_value = std_target

        self._gen = sim_grf.fBmGenerator(
            c=c, alpha=alpha, num_pt=L, minpadding=minpadding
        )

    def generate(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        return np.asarray(self._gen.sample(), dtype=float)


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def run_single_simulation(
    L: int,
    H: float,
    mode: str = 'site',
    seed: int = 42,
    std_target: float = 1.0,
    fix_c: float | None = None,
    minpadding: int = 0,
    sim_grf=None,
    verbose: bool = True,
) -> dict:
    """
    Run one TIP simulation and return statistics.

    Parameters
    ----------
    fix_c : float or None
        If given, pass directly to FieldGenerator (bypasses compute_c).
        If None, FieldGenerator uses compute_c(L, H, std_target).

    Returns
    -------
    dict with keys:
        cluster_mass, trapped_mass, elastic_mass, transport_mass,
        minimal_path_length, rg_cluster, breakthrough,
        state (np.ndarray), elastic_bb (set), transport_bb (set)
    """
    TIPEngine, BackboneExtractor, DataAnalyzer = _import_tip_modules()

    fg = FieldGenerator(
        L=L, H=H, std_target=std_target, fix_c=fix_c,
        minpadding=minpadding, sim_grf=sim_grf,
    )
    field = fg.generate(seed)

    engine = TIPEngine(field, mode=mode)
    if not verbose:
        import builtins
        _orig_print = builtins.print
        builtins.print = lambda *a, **k: None
        final_state = engine.run_until_breakthrough()
        builtins.print = _orig_print
    else:
        final_state = engine.run_until_breakthrough()

    if not engine.breakthrough:
        return {
            'cluster_mass': 0, 'trapped_mass': 0,
            'elastic_mass': 0, 'transport_mass': 0,
            'minimal_path_length': 0, 'rg_cluster': 0.0,
            'state': final_state, 'elastic_bb': set(), 'transport_bb': set(),
            'breakthrough': False,
        }

    extractor = BackboneExtractor(final_state)
    elastic_bb = extractor.extract_elastic_backbone()
    transport_bb = extractor.extract_transport_backbone()

    if verbose:
        print(f"  Elastic backbone: {len(elastic_bb)} sites")
        print(f"  Transport backbone: {len(transport_bb)} sites")

    analyzer = DataAnalyzer(final_state, elastic_bb, transport_bb)

    return {
        'cluster_mass': int(analyzer.calculate_mass('cluster')),
        'trapped_mass': int(analyzer.calculate_mass('trapped')),
        'elastic_mass': int(analyzer.calculate_mass('elastic')),
        'transport_mass': int(analyzer.calculate_mass('transport')),
        'minimal_path_length': analyzer.calculate_minimal_path_length(),
        'rg_cluster': float(analyzer.calculate_radius_of_gyration('cluster')),
        'state': final_state,
        'elastic_bb': elastic_bb,
        'transport_bb': transport_bb,
        'breakthrough': True,
    }


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _simulation_worker(task: tuple) -> dict:
    """
    Standalone worker called by Pool.map. Re-imports modules in each process
    (safe because Python caches imports after the first call).
    Returns a lightweight dict (no large arrays) for aggregation.
    """
    L, H, mode, seed, std_target, fix_c, minpadding = task
    sim_grf = _import_flow_modules()
    stats = run_single_simulation(
        L=L, H=H, mode=mode, seed=seed,
        std_target=std_target, fix_c=fix_c,
        minpadding=minpadding, sim_grf=sim_grf, verbose=False,
    )
    # Drop large arrays before returning across process boundary
    return {k: v for k, v in stats.items() if k not in ('state', 'elastic_bb', 'transport_bb')}


# ---------------------------------------------------------------------------
# Scaling study
# ---------------------------------------------------------------------------

def run_scaling_study(
    L_values: list[int],
    H: float,
    mode: str = 'site',
    n_realizations: int = 50,
    base_seed: int = 42,
    std_target: float = 1.0,
    fix_c: float | None = None,
    minpadding: int = 0,
    n_workers: int = 1,
) -> dict:
    """
    Run TIP at multiple L values with multiple realizations each.
    Realizations are parallelized across n_workers processes.

    Returns
    -------
    dict with L_values and per-metric mean/std arrays.
    """
    metric_keys = [
        'cluster_mass', 'trapped_mass', 'elastic_mass',
        'transport_mass', 'minimal_path_length', 'rg_cluster',
    ]

    # Build flat task list: all (L, realization) pairs
    tasks = [
        (L, H, mode, base_seed + i, std_target, fix_c, minpadding)
        for L in L_values
        for i in range(n_realizations)
    ]

    print(f"Submitting {len(tasks)} tasks to {n_workers} worker(s)...")

    if n_workers > 1:
        with Pool(n_workers) as pool:
            all_stats = pool.map(_simulation_worker, tasks)
    else:
        all_stats = [_simulation_worker(t) for t in tasks]

    # Group by L
    results = {k: {'mean': [], 'std': []} for k in metric_keys}
    results['L_values'] = L_values

    for L in L_values:
        L_stats = [
            s for s, t in zip(all_stats, tasks)
            if t[0] == L and s['breakthrough']
        ]
        n_ok = len(L_stats)
        n_total = n_realizations
        print(f"  L={L}: breakthrough in {n_ok}/{n_total} runs")

        for k in metric_keys:
            arr = np.array([s[k] for s in L_stats], dtype=float)
            results[k]['mean'].append(float(np.mean(arr)) if n_ok > 0 else np.nan)
            results[k]['std'].append(float(np.std(arr)) if n_ok > 0 else np.nan)

    for k in metric_keys:
        results[k]['mean'] = np.array(results[k]['mean'])
        results[k]['std'] = np.array(results[k]['std'])

    return results


# ---------------------------------------------------------------------------
# Fractal dimension fitting
# ---------------------------------------------------------------------------

def fit_fractal_dimensions(L_values, results) -> dict:
    """
    Fit fractal dimensions from scaling study results using both standard
    log-log regression and the paper's finite-size correction formula.
    """
    _, _, DataAnalyzer = _import_tip_modules()
    dummy_analyzer = DataAnalyzer(np.zeros((2, 2), dtype=int), set(), set())

    L_arr = np.array(L_values, dtype=float)
    fit_targets = ['cluster_mass', 'elastic_mass', 'transport_mass', 'minimal_path_length']

    fits = {}
    for k in fit_targets:
        mean_arr = results[k]['mean']
        valid = np.isfinite(mean_arr) & (mean_arr > 0)
        if valid.sum() < 2:
            fits[k] = {'Df_standard': np.nan, 'Df_advanced': np.nan, 'w': np.nan}
            continue

        Df_std = dummy_analyzer.fit_standard_scaling(L_arr[valid], mean_arr[valid])
        result_adv = dummy_analyzer.fit_advanced_finite_size_scaling(
            L_arr[valid], mean_arr[valid]
        )
        Df_adv, w = result_adv if isinstance(result_adv, tuple) else (result_adv, None)

        fits[k] = {
            'Df_standard': float(Df_std),
            'Df_advanced': float(Df_adv) if Df_adv is not None else np.nan,
            'w': float(w) if w is not None else np.nan,
        }

    return fits


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(final_state, extractor):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cmap_state = plt.matplotlib.colors.ListedColormap(['#a0c4ff', '#ffadad', '#2b2d42'])
    axes[0].imshow(final_state.T, cmap=cmap_state, origin='lower')
    axes[0].set_title("1. Final State\n(Blue=Defender, Red=Invader, Dark=Trapped)")
    axes[0].axis('off')

    elastic_grid, transport_grid = extractor.get_backbone_grids()

    axes[1].imshow(elastic_grid.T, cmap='Blues', origin='lower')
    axes[1].set_title("2. Elastic Backbone\n(Shortest Path / Minimal Path)")
    axes[1].axis('off')

    axes[2].imshow(transport_grid.T, cmap='Reds', origin='lower')
    axes[2].set_title("3. Transport Backbone\n(Includes Loops)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TIP simulation with FBM long-range correlations (Knackstedt et al. 2000)"
    )
    p.add_argument(
        '--L', type=int, nargs='+', default=[64, 128, 256],
        help='Grid size(s). For --single-run only the first value is used.',
    )
    p.add_argument('--H', type=float, default=0.7,
                   help='Hurst exponent (0 < H < 1).')
    p.add_argument('--mode', choices=['site', 'bond'], default='site',
                   help='site = imbibition; bond = drainage.')
    p.add_argument('--n-real', type=int, default=50,
                   help='Number of independent realizations per L.')
    p.add_argument('--seed', type=int, default=42,
                   help='Base random seed.')
    p.add_argument('--minpadding', type=int, default=0,
                   help='Circulant-embedding padding.')
    p.add_argument('--n-workers', type=int, default=1,
                   help='Number of parallel worker processes for realizations.')

    # Mutually exclusive: fix std or fix c
    scale_group = p.add_mutually_exclusive_group()
    scale_group.add_argument(
        '--std-target', type=float, default=None,
        help=(
            'Fix the marginal std of the FBM field (default mode, default value 1.0). '
            'c is computed per-L via compute_c(L, H, std_target), so variance is '
            'consistent across grid sizes.'
        ),
    )
    scale_group.add_argument(
        '--fix-c', type=float, default=None,
        help=(
            'Fix the FBM scale parameter c directly (bypasses compute_c). '
            'c is L-independent, so the effective std will vary across grid sizes — '
            'use this when you want to hold the intrinsic correlation strength fixed.'
        ),
    )

    p.add_argument('--output-dir', type=str, default='outputs/tip_fbm',
                   help='Directory for .npz output files.')
    p.add_argument('--visualize', action='store_true',
                   help='Show 3-panel plot (single-run only).')
    p.add_argument('--single-run', action='store_true',
                   help='Run one simulation (first L, given seed) for testing.')
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve scale parameters
    fix_c = args.fix_c
    std_target = args.std_target if args.std_target is not None else 1.0
    scale_desc = f"fix_c={fix_c}" if fix_c is not None else f"std_target={std_target}"

    sim_grf = _import_flow_modules()

    if args.single_run:
        L = args.L[0]
        print(f"\n--- Single TIP Run: L={L}, H={args.H}, mode={args.mode}, "
              f"seed={args.seed}, {scale_desc} ---")
        stats = run_single_simulation(
            L=L, H=args.H, mode=args.mode, seed=args.seed,
            std_target=std_target, fix_c=fix_c,
            minpadding=args.minpadding, sim_grf=sim_grf, verbose=True,
        )

        print("\n--- Statistics ---")
        print(f"  Cluster mass (SSC):     {stats['cluster_mass']}")
        print(f"  Trapped mass:           {stats['trapped_mass']}")
        print(f"  Elastic backbone:       {stats['elastic_mass']}")
        print(f"  Transport backbone:     {stats['transport_mass']}")
        print(f"  Minimal path length:    {stats['minimal_path_length']}")
        print(f"  Cluster Rg:             {stats['rg_cluster']:.2f}")
        print(f"  Backbone/cluster ratio: "
              f"{stats['transport_mass'] / max(stats['cluster_mass'], 1):.3f}")

        if args.visualize and stats['breakthrough']:
            _, BackboneExtractor, _ = _import_tip_modules()
            extractor = BackboneExtractor(stats['state'])
            extractor.elastic_bb = stats['elastic_bb']
            extractor.transport_bb = stats['transport_bb']
            extractor._calculate_chemical_distances()
            plot_results(stats['state'], extractor)

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"tip_single_L{L}_H{args.H}_mode{args.mode}_seed{args.seed}.npz"
            elastic_grid = np.zeros((L, L), dtype=int)
            transport_grid = np.zeros((L, L), dtype=int)
            for x, y in stats['elastic_bb']:
                elastic_grid[x, y] = 1
            for x, y in stats['transport_bb']:
                transport_grid[x, y] = 1
            np.savez_compressed(
                fname,
                state=stats['state'],
                elastic_grid=elastic_grid,
                transport_grid=transport_grid,
                L=np.array(L), H=np.array(args.H), mode=np.array(args.mode),
                seed=np.array(args.seed),
                cluster_mass=np.array(stats['cluster_mass']),
                trapped_mass=np.array(stats['trapped_mass']),
                elastic_mass=np.array(stats['elastic_mass']),
                transport_mass=np.array(stats['transport_mass']),
                minimal_path_length=np.array(stats['minimal_path_length']),
                rg_cluster=np.array(stats['rg_cluster']),
            )
            print(f"\nSaved to {fname}")

    else:
        print(f"\n=== TIP Scaling Study: H={args.H}, mode={args.mode}, {scale_desc} ===")
        print(f"L values: {args.L},  realizations per L: {args.n_real},  "
              f"workers: {args.n_workers}")

        results = run_scaling_study(
            L_values=args.L,
            H=args.H,
            mode=args.mode,
            n_realizations=args.n_real,
            base_seed=args.seed,
            std_target=std_target,
            fix_c=fix_c,
            minpadding=args.minpadding,
            n_workers=args.n_workers,
        )

        print("\n--- Scaling Results ---")
        header = (f"{'L':>6}  {'M_cluster':>12}  {'M_elastic':>12}  "
                  f"{'M_transport':>12}  {'M_min_path':>12}  {'Rg':>8}")
        print(header)
        print('-' * len(header))
        for i, L in enumerate(args.L):
            print(
                f"{L:>6}  "
                f"{results['cluster_mass']['mean'][i]:>12.1f}  "
                f"{results['elastic_mass']['mean'][i]:>12.1f}  "
                f"{results['transport_mass']['mean'][i]:>12.1f}  "
                f"{results['minimal_path_length']['mean'][i]:>12.1f}  "
                f"{results['rg_cluster']['mean'][i]:>8.2f}"
            )

        if len(args.L) >= 3:
            print("\n--- Fractal Dimension Estimates ---")
            fits = fit_fractal_dimensions(args.L, results)
            labels = {
                'cluster_mass': 'SSC (cluster)',
                'elastic_mass': 'Elastic backbone',
                'transport_mass': 'Transport backbone',
                'minimal_path_length': 'Minimal path',
            }
            print(f"{'Structure':<22}  {'Df (log-log)':>14}  {'Df (FSS)':>10}  {'omega':>8}")
            print('-' * 60)
            for k, label in labels.items():
                f = fits.get(k, {})
                print(
                    f"{label:<22}  "
                    f"{f.get('Df_standard', float('nan')):>14.4f}  "
                    f"{f.get('Df_advanced', float('nan')):>10.4f}  "
                    f"{f.get('w', float('nan')):>8.4f}"
                )

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"tip_fbm_H{args.H}_mode{args.mode}.npz"
            save_dict = {
                'L_values': np.array(args.L),
                'H': np.array(args.H),
                'mode': np.array(args.mode),
                'n_realizations': np.array(args.n_real),
                'base_seed': np.array(args.seed),
                'scale_mode': np.array('fix_c' if fix_c is not None else 'std'),
                'scale_value': np.array(fix_c if fix_c is not None else std_target),
            }
            for k in ['cluster_mass', 'trapped_mass', 'elastic_mass',
                      'transport_mass', 'minimal_path_length', 'rg_cluster']:
                save_dict[f'mean_{k}'] = results[k]['mean']
                save_dict[f'std_{k}'] = results[k]['std']
            np.savez_compressed(fname, **save_dict)
            print(f"\nSaved scaling data to {fname}")


if __name__ == "__main__":
    main()
