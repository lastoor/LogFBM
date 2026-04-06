#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dctn, fftn, next_fast_len
from scipy.interpolate import RegularGridInterpolator


def _add_search_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_solver_modules() -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parent.parent
    repo_root = Path(os.environ.get("FMM_FASTEST_PATH", str(workspace_root / "FMM_FastestPath"))).expanduser().resolve()
    algo_src = repo_root / "Algorithm" / "src"
    flowsim_path = Path(os.environ.get("FLOWSIM_PATH", str(workspace_root / "FlowSimulation"))).expanduser().resolve()

    if not algo_src.is_dir():
        raise RuntimeError(f"Algorithm src path does not exist: {algo_src}")
    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )

    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    _add_search_path(algo_src)
    _add_search_path(flowsim_path)

    try:
        from fmm_core.fmm import extract_path_continuous, extract_path_source_to_target, fast_marching_2d_regions
        from simulation import hydraulic as sim_hyd
        from simulation import particle_tracking as sim_pt
        from simulation import path_finding as sim_path
    except Exception as exc:
        raise RuntimeError(
            "Failed to import solver dependencies. "
            "Ensure FMM_FastestPath/Algorithm/src and FlowSimulation are available."
        ) from exc

    return {
        "extract_path_continuous": extract_path_continuous,
        "extract_path_source_to_target": extract_path_source_to_target,
        "fast_marching_2d_regions": fast_marching_2d_regions,
        "sim_hyd": sim_hyd,
        "sim_pt": sim_pt,
        "sim_path": sim_path,
    }


class GaussianRandomFieldCircEmbed:
    def __init__(self, mean, cov, pts, minpadding=0):
        self.mean = mean
        self.cov = cov
        self.pts = pts

        normedpts = [p - p[0] for p in pts]
        pad = minpadding
        if np.isscalar(pad):
            pads = [pad] * len(pts)
        else:
            pads = pad
        dims = [
            self.circulant_minsize(cov, normedpts[i], pads[i])
            for i in range(len(pts))
        ]

        lambdas = self.circulant_eigvals(cov, normedpts, dims)

        total_modes = np.prod(dims)
        neg_mask = lambdas < 0
        n_neg = int(neg_mask.sum())
        lambda_min = None
        if n_neg:
            lambda_min = float(lambdas[neg_mask].min())
            lambdas[neg_mask] = 0.0

        pos_mask = ~neg_mask
        lambdas[pos_mask] = np.sqrt(lambdas[pos_mask] / total_modes)

        if n_neg:
            warnings.warn(
                f"{n_neg} negative eigenvalues >= {lambda_min:.3e} were set to zero; "
                "consider increasing padding.",
                stacklevel=2,
            )

        self.data = (lambdas, tuple(dims))

    def sample(self):
        lambdas, shape_emb = self.data
        xi = np.random.randn(*shape_emb)
        w = lambdas * xi
        w = fftn(w)
        orig_shape = tuple(len(p) for p in self.pts)
        slices = tuple(slice(0, n) for n in orig_shape)
        w_crop = w[slices]
        return self.mean + (w_crop.real + w_crop.imag)

    @staticmethod
    def circulant_minsize(cov, pts, minpadding):
        n = len(pts)
        base = n + minpadding - (1 if cov.is_even else 0)
        return 2 * next_fast_len(base)

    @staticmethod
    def circulant_eigvals(cov, pts, dims):
        pts = [np.asarray(p) for p in pts]
        starts = [p[0] for p in pts]
        steps = [p[1] - p[0] for p in pts]
        ndim = len(dims)
        is_even = getattr(cov, "is_even", False)
        cov_vec = np.vectorize(cov, signature="(n)->()")

        if is_even:
            half_shape = tuple(d // 2 + 1 for d in dims)
            coords = [starts[d] + np.arange(half_shape[d]) * steps[d] for d in range(ndim)]
            mesh = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
            cov_eval = cov_vec(mesh)
            cov_eval = dctn(cov_eval, type=1, norm=None)
            idx = [np.minimum(np.arange(d), d - np.arange(d)) for d in dims]
            mesh_idx = np.ix_(*idx)
            return cov_eval[mesh_idx]

        cov_eval = np.zeros(dims, dtype=float)
        dims2 = [d + 2 for d in dims]
        mids = [d2 // 2 for d2 in dims2]
        grid = [np.arange(d) for d in dims]
        coords = []
        for d in range(ndim):
            idx = grid[d]
            pos = np.where(idx < mids[d], idx, idx - dims[d])
            coords.append(starts[d] + pos * steps[d])
        mesh = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
        mask = np.ones(dims, dtype=bool)
        for d in range(ndim):
            shape = tuple(dims[j] if j == d else 1 for j in range(ndim))
            mask &= ((grid[d] + 1) != mids[d]).reshape(shape)
        cov_eval[mask] = cov_vec(mesh[mask])
        return np.real(np.fft.fftn(cov_eval))


class PrefbmCov:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_even = True
        self.c0 = None
        self.c2 = None

    def __call__(self, x):
        r = np.linalg.norm(x)
        alpha = self.alpha
        if 0 < alpha <= 1.5:
            self.c0 = 1 - alpha / 2
            self.c2 = alpha / 2
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0 - r**alpha + self.c2 * r**2
            return 0.0
        if 1.5 < alpha < 2:
            self.c0 = 1 - alpha / 6 - alpha**2 / 6
            self.c2 = alpha * (5 + 2 * alpha) / 18
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0 - r**alpha + self.c2 * r**2
            if r <= 2:
                return alpha * (2 - alpha) / 18 * (2 - r) ** 3 / r
            return 0.0
        raise ValueError("alpha must lie in (0, 2)")


class ExpCov:
    def __init__(self, sigma=1.0, length_scale=1.0):
        self.sigma = sigma
        self.length_scale = length_scale
        self.is_even = True

    def __call__(self, x):
        r = np.linalg.norm(x)
        return self.sigma**2 * np.exp(-np.abs(r) / self.length_scale)


class fBmGenerator:
    def __init__(self, *, c, alpha, num_pt, dim=2, minpadding=0):
        self.c = c
        self.alpha = alpha
        self.num_pt = num_pt
        self.pts = (np.arange(num_pt) / num_pt,) * dim
        self.minpadding = minpadding if alpha <= 1.5 else minpadding + num_pt
        self.cov_c0 = None
        self.cov_c2 = None

        mean = np.zeros(tuple(len(p) for p in self.pts))
        cov = PrefbmCov(alpha)
        self._prefbm_gen = GaussianRandomFieldCircEmbed(mean, cov, self.pts, minpadding=self.minpadding)
        self.cov_c0 = self._prefbm_gen.cov.c0
        self.cov_c2 = self._prefbm_gen.cov.c2
        self._sigma_trend = np.sqrt(2 * self.cov_c2)

    def sample(self):
        prefbm = self._prefbm_gen.sample()
        coords = np.meshgrid(*self.pts, indexing="ij")
        coeffs = np.random.normal(scale=self._sigma_trend, size=len(coords))
        trend = sum(coord * coeff for coord, coeff in zip(coords, coeffs))
        fbm_c0_unit = prefbm + trend
        fbm = np.sqrt(self.c) * self.num_pt ** (self.alpha / 2) * (fbm_c0_unit - fbm_c0_unit[0, 0])
        return fbm


def compute_c(l, H, std_target):
    coords = np.arange(l)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="ij")
    r = np.sqrt(x_grid**2 + y_grid**2)
    avg_r2h = np.mean(r ** (2 * H))
    return std_target**2 / (2 * avg_r2h)


def generate_logk_expcov(field_size, std, length_scale, seed, minpadding=0):
    np.random.seed(seed)
    cov = ExpCov(sigma=std, length_scale=length_scale)
    pts = (np.linspace(0, 1, field_size),) * 2
    mean = np.zeros((field_size, field_size))
    grf = GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=minpadding)
    return np.asarray(grf.sample(), dtype=float)


def generate_logk_fbm(field_size, alpha, std_target, seed, minpadding=0):
    np.random.seed(seed)
    c = compute_c(field_size, H=alpha / 2, std_target=std_target)
    gen = fBmGenerator(c=c, alpha=alpha, num_pt=field_size, minpadding=minpadding)
    return np.asarray(gen.sample(), dtype=float)


@dataclass(slots=True)
class Geometry:
    x: np.ndarray
    y: np.ndarray
    source_x: int
    target_x: int
    source_y_start: int
    source_y_end: int
    target_y_start: int
    target_y_end: int

    @property
    def source_rows(self) -> np.ndarray:
        return np.arange(self.source_y_start, self.source_y_end, dtype=int)

    @property
    def target_rows(self) -> np.ndarray:
        return np.arange(self.target_y_start, self.target_y_end, dtype=int)


def default_geometry(field_size: int) -> Geometry:
    source_x = field_size // 20 - 1
    target_x = field_size - field_size // 20 - 1
    source_y_start = field_size // 20 - 1
    source_y_end = field_size - field_size // 20 - 1
    target_y_start = 0
    target_y_end = field_size
    return Geometry(
        x=np.arange(0, field_size, 1.0),
        y=np.arange(0, field_size, 1.0),
        source_x=int(source_x),
        target_x=int(target_x),
        source_y_start=int(source_y_start),
        source_y_end=int(source_y_end),
        target_y_start=int(target_y_start),
        target_y_end=int(target_y_end),
    )


def build_face_coordinates_from_centers(coord_centers: np.ndarray) -> np.ndarray:
    coord_centers = np.asarray(coord_centers, dtype=float).ravel()
    if coord_centers.size < 2:
        raise ValueError("coord_centers must contain at least 2 values")
    faces = np.empty(coord_centers.size + 1, dtype=float)
    faces[1:-1] = 0.5 * (coord_centers[:-1] + coord_centers[1:])
    faces[0] = coord_centers[0] - 0.5 * (coord_centers[1] - coord_centers[0])
    faces[-1] = coord_centers[-1] + 0.5 * (coord_centers[-1] - coord_centers[-2])
    return faces


def _resolve_particle_count(*, num_particles: int | None, density_particle: int, y0: int, y1: int) -> int:
    if num_particles is not None:
        return max(2, int(num_particles))
    return max(2, int(density_particle * abs(y1 - y0)))


def _build_particle_seed_y(*, y_start: int, y_end: int, num_particles: int) -> np.ndarray:
    return np.linspace(float(y_start), float(y_end), int(num_particles), endpoint=False, dtype=np.float32)


def _validate_particle_tracking_options(*, solver: str, velocity_location: str, backend: str = "cpu") -> None:
    if solver not in {"pollock", "rk45"}:
        raise ValueError(f"Unsupported particle-tracking solver: {solver}")
    if velocity_location not in {"face", "cell"}:
        raise ValueError(f"Unsupported velocity_location={velocity_location}; expected 'face' or 'cell'.")
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported backend={backend}; expected 'cpu' or 'gpu'.")
    if solver == "pollock" and velocity_location != "face":
        raise ValueError("Pollock solver requires velocity_location='face'.")
    if solver != "pollock" and backend == "gpu":
        raise ValueError("GPU backend is currently supported only with solver='pollock'.")


def _build_source_target_masks(
    *,
    field_shape: tuple[int, int],
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    x0: int,
    x1: int,
    traversable_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_y, n_x = field_shape
    source_mask = np.zeros((n_y, n_x), dtype=bool)
    target_mask = np.zeros((n_y, n_x), dtype=bool)
    source_mask[source_rows, x0] = True
    target_mask[target_rows, x1] = True
    if traversable_mask is None:
        traversable_mask = np.ones((n_y, n_x), dtype=bool)
    else:
        traversable_mask = np.asarray(traversable_mask, dtype=bool)
        if traversable_mask.shape != (n_y, n_x):
            raise ValueError(f"Traversable-mask shape {traversable_mask.shape} does not match field shape {(n_y, n_x)}.")
    source_mask = source_mask & traversable_mask
    target_mask = target_mask & traversable_mask
    return source_mask, target_mask, traversable_mask


def _extract_fmm_path_from_t_field(
    t_field: np.ndarray,
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    backtracking: str,
    continuous_step_size: float,
    traversable_mask: np.ndarray,
) -> np.ndarray:
    if backtracking == "continuous":
        path_ij, _chosen_target = extract_path_continuous(
            t_field,
            source_mask,
            target_mask,
            step_size=continuous_step_size,
            traversable_mask=traversable_mask,
        )
    elif backtracking in {"discrete", "zigzag"}:
        path_ij, _chosen_target = extract_path_source_to_target(
            t_field,
            source_mask,
            target_mask,
            traversable_mask=traversable_mask,
        )
    else:
        raise ValueError(f"Unsupported fmm backtracking method {backtracking!r}; expected 'discrete' or 'continuous'.")
    if len(path_ij) < 2:
        raise RuntimeError("FMM path extraction failed.")
    return np.asarray(path_ij, dtype=float)[:, [1, 0]][::-1]


def _compute_fmm_path(
    field: np.ndarray,
    *,
    geometry: Geometry,
    grid_spacing: float,
    fast_marching_2d_regions: Any,
    extract_path_source_to_target: Any,
    extract_path_continuous: Any,
    backtracking: str,
    continuous_step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    traversable_mask = np.isfinite(field) & (field > 0.0)
    source_mask, target_mask, traversable_mask = _build_source_target_masks(
        field_shape=field.shape,
        source_rows=geometry.source_rows,
        target_rows=geometry.target_rows,
        x0=geometry.source_x,
        x1=geometry.target_x,
        traversable_mask=traversable_mask,
    )
    if not np.any(source_mask) or not np.any(target_mask):
        raise RuntimeError("No traversable source/target cells for FMM.")

    t_field, _state = fast_marching_2d_regions(field, source_mask, h=grid_spacing, traversable_mask=traversable_mask)
    path_xy = _extract_fmm_path_from_t_field(
        t_field,
        source_mask=source_mask,
        target_mask=target_mask,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=backtracking,
        continuous_step_size=continuous_step_size,
        traversable_mask=traversable_mask,
    )
    return path_xy, np.asarray(t_field, dtype=float)


def _run_particle_tracking(
    *,
    sim_hyd: Any,
    sim_pt: Any,
    k_field: np.ndarray,
    h_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: Geometry,
    solver: str,
    backend: str,
    velocity_location: str,
    align_with_modflow_velocity: bool,
    density_particle: int,
    num_particles: int | None,
    eps_speed: float,
    ds_min: float,
    ds_max: float,
    tol: float,
    y_policy: str,
    max_points: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_particle_tracking_options(solver=solver, velocity_location=velocity_location, backend=backend)

    n_particles = _resolve_particle_count(
        num_particles=num_particles,
        density_particle=density_particle,
        y0=geometry.source_y_start,
        y1=geometry.source_y_end,
    )
    seeds_x = np.full(n_particles, geometry.source_x, np.float32)
    seeds_y = _build_particle_seed_y(
        y_start=geometry.source_y_start,
        y_end=geometry.source_y_end,
        num_particles=n_particles,
    )

    summary_params: dict[str, Any] = {
        "backend": backend,
        "method": solver,
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "arrival_fraction": 1.0,
        "save_trajectories": False,
        "return_fastest": True,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": geometry.target_x,
        "max_points": max_points,
        "max_iters": max_iters,
        "threads_per_block": 256,
        "dtype": np.float64,
    }

    if solver == "pollock":
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h_field,
            output_location="face",
            align_with_modflow=align_with_modflow_velocity,
            face_output="full",
        )
        summary_params.update(
            {
                "x_faces": build_face_coordinates_from_centers(x).astype(np.float32),
                "y_faces": build_face_coordinates_from_centers(y).astype(np.float32),
                "vx": vx,
                "vy": vy,
                "max_steps": max_iters,
            }
        )
    else:
        vx, vy = sim_hyd.compute_velocity(
            k_field,
            x,
            y,
            h_field,
            output_location=velocity_location,
            align_with_modflow=align_with_modflow_velocity,
        )
        summary_params.update(
            {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "velocity_location": velocity_location,
                "eps_speed": eps_speed,
                "ds_min": ds_min,
                "ds_max": ds_max,
                "tol": tol,
                "y_policy": y_policy,
            }
        )

    summary = sim_pt.particle_tracking_summary(summary_params)
    result = summary.get("fastest")
    if result is None:
        raise RuntimeError("Particle tracking summary did not return the fastest trajectory.")
    fp = np.vstack((result["x"], result["y"])).T
    return np.asarray(fp, dtype=float), np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)


def compute_cumulative_travel_time(path_xy, vx_field, vy_field, x_coords, y_coords):
    path_xy = np.asarray(path_xy, dtype=float)
    vx_field = np.asarray(vx_field, dtype=float)
    vy_field = np.asarray(vy_field, dtype=float)
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)

    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError(f"path_xy must have shape (N, 2); got {path_xy.shape}")
    if path_xy.shape[0] == 0:
        raise ValueError("path_xy must contain at least one point")

    vx_interp = RegularGridInterpolator(
        (y_coords, x_coords),
        vx_field,
        bounds_error=False,
        fill_value=np.nan,
    )
    vy_interp = RegularGridInterpolator(
        (y_coords, x_coords),
        vy_field,
        bounds_error=False,
        fill_value=np.nan,
    )

    if path_xy.shape[0] == 1:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)

    delta = np.diff(path_xy, axis=0)
    ds = np.hypot(delta[:, 0], delta[:, 1])
    arc_lengths = np.concatenate(([0.0], np.cumsum(ds)))

    tx = np.divide(delta[:, 0], ds, out=np.zeros_like(ds), where=ds > 0.0)
    ty = np.divide(delta[:, 1], ds, out=np.zeros_like(ds), where=ds > 0.0)

    midpoints = 0.5 * (path_xy[:-1] + path_xy[1:])
    midpoints_yx = np.column_stack([midpoints[:, 1], midpoints[:, 0]])
    vx_mid = vx_interp(midpoints_yx)
    vy_mid = vy_interp(midpoints_yx)
    v_proj = vx_mid * tx + vy_mid * ty

    global_speed = np.hypot(vx_field, vy_field)
    global_max_speed = float(np.nanmax(global_speed))
    if not np.isfinite(global_max_speed) or global_max_speed <= 0.0:
        global_max_speed = 1.0
    v_proj_safe = np.maximum(v_proj, 1e-10 * global_max_speed)
    travel_increments = np.divide(ds, v_proj_safe, out=np.full_like(ds, np.inf), where=np.isfinite(v_proj_safe))
    travel_times = np.concatenate(([0.0], np.cumsum(travel_increments)))
    return arc_lengths, travel_times


def _solve_head_field(sim_hyd: Any, k_field: np.ndarray, x: np.ndarray, y: np.ndarray, dh: float) -> np.ndarray:
    result = sim_hyd.solve_head(k_field, x, y, dh=dh)
    if isinstance(result, tuple):
        if len(result) < 1:
            raise RuntimeError("solve_head returned an empty tuple.")
        h_field = result[0]
        msg = result[1] if len(result) > 1 else ""
        is_success = result[2] if len(result) > 2 else True
        if not bool(is_success):
            raise RuntimeError(f"Hydraulic head solve failed: {msg}")
        return np.asarray(h_field, dtype=float)
    return np.asarray(result, dtype=float)


def _run_visualization(
    *,
    npz_path: Path,
    output_dir: str | None,
    dpi: int,
    figure_format: str,
    stream_density: float,
    path_linewidth: float,
    save_logk_paths: bool,
    save_head_velocity: bool,
    save_travel_time: bool,
) -> None:
    viz_script = Path(__file__).resolve().parent / "visualize_lrp_fp.py"
    if not viz_script.is_file():
        raise RuntimeError(f"Visualization script does not exist: {viz_script}")

    cmd = [
        sys.executable,
        str(viz_script),
        "--input",
        str(npz_path),
        "--dpi",
        str(dpi),
        "--format",
        figure_format,
        "--stream-density",
        str(stream_density),
        "--path-linewidth",
        str(path_linewidth),
    ]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    cmd.append("--save-logk-paths" if save_logk_paths else "--no-save-logk-paths")
    cmd.append("--save-head-velocity" if save_head_velocity else "--no-save-head-velocity")
    cmd.append("--save-travel-time" if save_travel_time else "--no-save-travel-time")
    subprocess.run(cmd, check=True)


def run(
    field_type,
    field_size,
    std,
    length_scale,
    alpha,
    seed,
    dh,
    solver,
    backend,
    velocity_location,
    align_with_modflow,
    density_particle,
    num_particles,
    eps_speed,
    ds_min,
    ds_max,
    tol,
    y_policy,
    max_points,
    max_iters,
    fmm_backtracking,
    fmm_continuous_step_size,
    output_path,
    visualize,
    visualize_output_dir,
    visualize_dpi,
    visualize_format,
    visualize_stream_density,
    visualize_path_linewidth,
    visualize_logk_paths,
    visualize_head_velocity,
    visualize_travel_time,
):
    modules = _import_solver_modules()
    sim_hyd = modules["sim_hyd"]
    sim_pt = modules["sim_pt"]
    fast_marching_2d_regions = modules["fast_marching_2d_regions"]
    extract_path_source_to_target = modules["extract_path_source_to_target"]
    extract_path_continuous = modules["extract_path_continuous"]

    geometry = default_geometry(field_size)
    x = geometry.x
    y = geometry.y

    if field_type == "expcov":
        logk = generate_logk_expcov(
            field_size=field_size,
            std=std,
            length_scale=length_scale,
            seed=seed,
        )
        grf_length_scale = float(length_scale)
        fbm_alpha = np.nan
    elif field_type == "fbm":
        logk = generate_logk_fbm(
            field_size=field_size,
            alpha=alpha,
            std_target=std,
            seed=seed,
        )
        grf_length_scale = np.nan
        fbm_alpha = float(alpha)
    else:
        raise ValueError(f"Unsupported field_type={field_type!r}")

    k_field = np.exp(logk)
    h = _solve_head_field(sim_hyd, k_field, x, y, dh)
    grad_h_y, grad_h_x = np.gradient(h, y, x, edge_order=2)
    vx_cell = -k_field * grad_h_x
    vy_cell = -k_field * grad_h_y

    dh_mean = abs(float(dh))
    if dh_mean <= 0.0 or not np.isfinite(dh_mean):
        raise ValueError(f"Invalid dh={dh}; expected a positive finite magnitude.")
    lrp_speed = k_field * dh_mean
    lrp, _t_lrp = _compute_fmm_path(
        lrp_speed,
        geometry=geometry,
        grid_spacing=1.0,
        fast_marching_2d_regions=fast_marching_2d_regions,
        extract_path_source_to_target=extract_path_source_to_target,
        extract_path_continuous=extract_path_continuous,
        backtracking=fmm_backtracking,
        continuous_step_size=fmm_continuous_step_size,
    )

    fp, _vx_track, _vy_track = _run_particle_tracking(
        sim_hyd=sim_hyd,
        sim_pt=sim_pt,
        k_field=k_field,
        h_field=h,
        x=x,
        y=y,
        geometry=geometry,
        solver=solver,
        backend=backend,
        velocity_location=velocity_location,
        align_with_modflow_velocity=align_with_modflow,
        density_particle=density_particle,
        num_particles=num_particles,
        eps_speed=eps_speed,
        ds_min=ds_min,
        ds_max=ds_max,
        tol=tol,
        y_policy=y_policy,
        max_points=max_points,
        max_iters=max_iters,
    )

    fp_arc_length, fp_travel_time = compute_cumulative_travel_time(fp, vx_cell, vy_cell, x, y)
    lrp_arc_length, lrp_pseudo_travel_time = compute_cumulative_travel_time(lrp, vx_cell, vy_cell, x, y)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        field_type=np.array(field_type),
        field_size=np.array(field_size),
        seed=np.array(seed),
        dh=np.array(dh),
        grf_std=np.array(std),
        grf_length_scale=np.array(grf_length_scale),
        fbm_alpha=np.array(fbm_alpha),
        x=x,
        y=y,
        logk=logk,
        h=h,
        vx=vx_cell,
        vy=vy_cell,
        fp=fp,
        fp_arc_length=fp_arc_length,
        fp_travel_time=fp_travel_time,
        lrp=lrp,
        lrp_arc_length=lrp_arc_length,
        lrp_pseudo_travel_time=lrp_pseudo_travel_time,
        source_x=np.array(geometry.source_x),
        target_x=np.array(geometry.target_x),
        source_y_start=np.array(geometry.source_y_start),
        source_y_end=np.array(geometry.source_y_end),
    )
    if visualize:
        _run_visualization(
            npz_path=output_path,
            output_dir=visualize_output_dir,
            dpi=visualize_dpi,
            figure_format=visualize_format,
            stream_density=visualize_stream_density,
            path_linewidth=visualize_path_linewidth,
            save_logk_paths=visualize_logk_paths,
            save_head_velocity=visualize_head_velocity,
            save_travel_time=visualize_travel_time,
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a logK field, FP/LRP paths, and pseudo travel times.")
    parser.add_argument("--field-type", choices=("expcov", "fbm"), required=True)
    parser.add_argument("--field-size", type=int, default=100)
    parser.add_argument("--std", type=float, default=1.0, help="Target logK standard deviation.")
    parser.add_argument("--length-scale", type=float, default=0.2, help="Exponential covariance length scale in normalized units.")
    parser.add_argument("--fbm-alpha", type=float, default=1.0, help="fBm alpha = 2 * Hurst.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dh", type=float, default=1.0)
    parser.add_argument("--solver", choices=("rk45", "pollock"), default="rk45")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--velocity-location", choices=("cell", "face"), default="cell")
    parser.add_argument("--align-with-modflow", action="store_true", default=False)
    parser.add_argument("--density-particle", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--eps-speed", type=float, default=1e-12)
    parser.add_argument("--ds-min", type=float, default=0.05)
    parser.add_argument("--ds-max", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--y-policy", choices=("clip", "terminate"), default="clip")
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--max-iters", type=int, default=200000)
    parser.add_argument("--fmm-backtracking", choices=("discrete", "continuous"), default="continuous")
    parser.add_argument("--fmm-continuous-step-size", type=float, default=0.5)
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--visualize", action="store_true", default=False, help="After saving the .npz, call visualize_lrp_fp.py to save figures.")
    parser.add_argument("--visualize-output-dir", default=None, help="Optional output directory for visualization figures.")
    parser.add_argument("--visualize-dpi", type=int, default=300)
    parser.add_argument("--visualize-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--visualize-stream-density", type=float, default=1.0)
    parser.add_argument("--visualize-path-linewidth", type=float, default=2.0)
    parser.add_argument("--visualize-logk-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-head-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-travel-time", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.field_type == "expcov" and args.length_scale <= 0.0:
        raise ValueError("--length-scale must be positive for expcov fields.")
    if args.field_type == "fbm" and not (0.0 < args.fbm_alpha < 2.0):
        raise ValueError("--fbm-alpha must lie in (0, 2).")
    run(
        field_type=args.field_type,
        field_size=args.field_size,
        std=args.std,
        length_scale=args.length_scale,
        alpha=args.fbm_alpha,
        seed=args.seed,
        dh=args.dh,
        solver=args.solver,
        backend=args.backend,
        velocity_location=args.velocity_location,
        align_with_modflow=args.align_with_modflow,
        density_particle=args.density_particle,
        num_particles=args.num_particles,
        eps_speed=args.eps_speed,
        ds_min=args.ds_min,
        ds_max=args.ds_max,
        tol=args.tol,
        y_policy=args.y_policy,
        max_points=args.max_points,
        max_iters=args.max_iters,
        fmm_backtracking=args.fmm_backtracking,
        fmm_continuous_step_size=args.fmm_continuous_step_size,
        output_path=args.output,
        visualize=args.visualize,
        visualize_output_dir=args.visualize_output_dir,
        visualize_dpi=args.visualize_dpi,
        visualize_format=args.visualize_format,
        visualize_stream_density=args.visualize_stream_density,
        visualize_path_linewidth=args.visualize_path_linewidth,
        visualize_logk_paths=args.visualize_logk_paths,
        visualize_head_velocity=args.visualize_head_velocity,
        visualize_travel_time=args.visualize_travel_time,
    )


if __name__ == "__main__":
    main()
