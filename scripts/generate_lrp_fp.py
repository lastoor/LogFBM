#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


GPU_QUEUE_MAXSIZE = 128
GPU_SUBMIT_TIMEOUT_SEC = 30.0
GPU_RESULT_POLL_SEC = 0.001
MAX_BATCH_SIZE = 32
MAX_BATCH_WAIT_MS = 5


global_flow_modules: dict[str, Any] | None = None
global_fmm_modules: dict[str, Any] | None = None
global_queue_gpu: Any = None
global_result_dict: Any = None
global_pollock_tracker: Any = None
global_pollock_tracker_signature: tuple[Any, ...] | None = None
global_pollock_gpu_modules: dict[str, Any] | None = None


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


@dataclass(slots=True)
class RunConfig:
    field_type: str
    field_size: int
    std: float
    length_scale: float
    alpha: float
    fbm_c: float | None
    dh: float
    lrp_method: str
    solver: str
    backend: str
    velocity_location: str
    align_with_modflow: bool
    density_particle: int
    num_particles: int | None
    eps_speed: float
    ds_min: float
    ds_max: float
    tol: float
    y_policy: str
    max_points: int
    max_iters: int
    fmm_backtracking: str
    fmm_continuous_step_size: float


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool
    output_dir: str | None
    dpi: int
    figure_format: str
    stream_density: float
    path_linewidth: float
    save_logk_paths: bool
    save_head_velocity: bool
    save_travel_time: bool
    save_computation_time: bool
    save_first_arrival_plume: bool


def _add_search_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _flow_repo_path() -> Path:
    workspace_root = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("FLOWSIM_PATH", str(workspace_root.parent / "FlowSimulation"))).expanduser().resolve()


def _fmm_src_path() -> Path:
    workspace_root = Path(__file__).resolve().parent.parent
    return Path(
        os.environ.get(
            "FMM_CORE_PATH",
            str(workspace_root.parent / "FMM_FastestPath" / "Algorithm" / "src"),
        )
    ).expanduser().resolve()


def _import_flow_modules() -> dict[str, Any]:
    flowsim_path = _flow_repo_path()
    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )

    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    _add_search_path(flowsim_path)

    try:
        from simulation import grf_generation as sim_grf
        from simulation import hydraulic as sim_hyd
        from simulation import path_finding as sim_path_finding
        from simulation import particle_tracking as sim_pt
    except Exception as exc:
        raise RuntimeError(
            "Failed to import FlowSimulation dependencies. "
            "Ensure FlowSimulation is available and its Python dependencies are installed."
        ) from exc

    return {
        "sim_grf": sim_grf,
        "sim_hyd": sim_hyd,
        "sim_path_finding": sim_path_finding,
        "sim_pt": sim_pt,
    }


def _import_pollock_gpu_modules() -> dict[str, Any]:
    global global_pollock_gpu_modules
    if global_pollock_gpu_modules is not None:
        return global_pollock_gpu_modules

    flowsim_path = _flow_repo_path()
    if not flowsim_path.is_dir():
        raise RuntimeError(
            f"FlowSimulation path does not exist: {flowsim_path}. "
            "Set FLOWSIM_PATH to your FlowSimulation repository root."
        )
    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    _add_search_path(flowsim_path)

    try:
        from simulation.particle_tracking import pollock_gpu as sim_pt_pollock_gpu
        from simulation.particle_tracking import utils as sim_pt_utils
    except Exception as exc:
        raise RuntimeError(
            "Failed to import FlowSimulation Pollock GPU dependencies. "
            "Ensure the GPU particle-tracking dependencies are installed."
        ) from exc

    global_pollock_gpu_modules = {
        "sim_pt_pollock_gpu": sim_pt_pollock_gpu,
        "sim_pt_utils": sim_pt_utils,
    }
    return global_pollock_gpu_modules


def _import_fmm_modules() -> dict[str, Any]:
    fmm_src = _fmm_src_path()
    if not fmm_src.is_dir():
        raise RuntimeError(
            f"FMM source path does not exist: {fmm_src}. "
            "Set FMM_CORE_PATH to the directory that contains fmm_core/."
        )

    os.environ["LD_LIBRARY_PATH"] = "/apps/generic/gcc/13.3.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    _add_search_path(fmm_src)

    try:
        from fmm_core.fmm import extract_path_continuous, extract_path_source_to_target, fast_marching_2d_regions
    except Exception as exc:
        raise RuntimeError(
            "Failed to import fmm_core for LRP solving. "
            "Ensure the FMM source tree is available and dependencies are installed."
        ) from exc

    return {
        "extract_path_continuous": extract_path_continuous,
        "extract_path_source_to_target": extract_path_source_to_target,
        "fast_marching_2d_regions": fast_marching_2d_regions,
    }


def _get_flow_modules() -> dict[str, Any]:
    global global_flow_modules
    if global_flow_modules is None:
        global_flow_modules = _import_flow_modules()
    return global_flow_modules


def _get_fmm_modules() -> dict[str, Any]:
    global global_fmm_modules
    if global_fmm_modules is None:
        global_fmm_modules = _import_fmm_modules()
    return global_fmm_modules


def generate_logk_expcov(sim_grf: Any, field_size: int, std: float, length_scale: float, seed: int, minpadding: int = 0) -> np.ndarray:
    np.random.seed(seed)
    cov = sim_grf.ExpCov(sigma=std, length_scale=length_scale)
    pts = (np.linspace(0, 1, field_size),) * 2
    mean = np.zeros((field_size, field_size))
    grf = sim_grf.GaussianRandomFieldCircEmbed(mean, cov, pts, minpadding=minpadding)
    return np.asarray(grf.sample(), dtype=float)


def generate_logk_fbm(
    sim_grf: Any,
    field_size: int,
    alpha: float,
    std_target: float,
    seed: int,
    c: float | None = None,
    minpadding: int = 0,
) -> np.ndarray:
    np.random.seed(seed)
    if c is None:
        c = sim_grf.compute_c(field_size, H=alpha / 2, std_target=std_target)
    gen = sim_grf.fBmGenerator(c=c, alpha=alpha, num_pt=field_size, minpadding=minpadding)
    return np.asarray(gen.sample(), dtype=float)


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


def _compute_dijkstra_path(
    field: np.ndarray,
    *,
    geometry: Geometry,
    sim_path_finding: Any,
) -> tuple[np.ndarray, np.ndarray]:
    source_region = [(int(row), int(geometry.source_x)) for row in geometry.source_rows]
    target_region = [(int(row), int(geometry.target_x)) for row in geometry.target_rows]
    path_xy, _total_resistance, _start_node, _end_node, resistance_map = sim_path_finding.dijkstra_2d_least_resistance(
        field,
        source_region,
        target_region,
        average="harmonic",
    )
    if len(path_xy) < 2:
        raise RuntimeError("Dijkstra path extraction failed.")
    return np.asarray(path_xy, dtype=float), np.asarray(resistance_map, dtype=float)


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


def _array_signature(payload: dict[str, Any], key: str) -> tuple[tuple[int, ...], str]:
    shm_key = f"{key}_shm"
    if shm_key in payload:
        meta = dict(payload[shm_key])
        return tuple(meta["shape"]), str(meta["dtype"])
    arr = np.asarray(payload[key])
    return tuple(arr.shape), str(arr.dtype)


def _array_last_value(payload: dict[str, Any], key: str) -> float:
    shm_key = f"{key}_shm"
    if shm_key in payload:
        meta = dict(payload[shm_key])
        return float(meta["last_value"])
    arr = np.asarray(payload[key])
    return float(arr.ravel()[-1])


def _seed_count(payload: dict[str, Any]) -> int:
    if "seed_spec" in payload:
        return int(payload["seed_spec"]["n_particles"])
    return int(np.asarray(payload["seeds_x"]).size)


def _payload_batch_key(func_name: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    vx_shape, vx_dtype = _array_signature(payload, "vx")
    vy_shape, vy_dtype = _array_signature(payload, "vy")
    x_faces_shape, x_faces_dtype = _array_signature(payload, "x_faces")
    y_faces_shape, y_faces_dtype = _array_signature(payload, "y_faces")
    return (
        func_name,
        str(payload.get("method", "")),
        str(payload.get("backend", "")),
        vx_shape,
        vy_shape,
        x_faces_shape,
        y_faces_shape,
        vx_dtype,
        vy_dtype,
        x_faces_dtype,
        y_faces_dtype,
        _seed_count(payload),
        float(payload.get("right_x", _array_last_value(payload, "x_faces"))),
        int(payload.get("max_points", 0)),
        int(payload.get("max_steps", payload.get("max_iters", 0))),
        str(payload.get("tracking_mode", "")),
        str(payload.get("stop_mode", "")),
    )


def _build_seed_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "seed_spec" in payload:
        spec = dict(payload["seed_spec"])
        n_particles = max(2, int(spec["n_particles"]))
        x0 = float(spec["x0"])
        y_start = float(spec["y_start"])
        y_end = float(spec["y_end"])
        seeds_x = np.full(n_particles, x0, np.float32)
        seeds_y = np.linspace(y_start, y_end, n_particles, endpoint=False, dtype=np.float32)
        return seeds_x, seeds_y
    return (
        np.asarray(payload["seeds_x"], dtype=np.float32).ravel(),
        np.asarray(payload["seeds_y"], dtype=np.float32).ravel(),
    )


def _share_array(array: np.ndarray) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    arr = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_view[...] = arr
    descriptor = {
        "name": shm.name,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
        "last_value": float(arr.ravel()[-1]) if arr.size else np.nan,
    }
    return descriptor, shm


def _prepare_payload_for_ipc(payload: dict[str, Any]) -> tuple[dict[str, Any], list[shared_memory.SharedMemory]]:
    payload_ipc = dict(payload)
    shm_handles: list[shared_memory.SharedMemory] = []
    for key in ("vx", "vy", "x_faces", "y_faces"):
        descriptor, shm = _share_array(np.asarray(payload_ipc.pop(key)))
        payload_ipc[f"{key}_shm"] = descriptor
        shm_handles.append(shm)
    return payload_ipc, shm_handles


def _cleanup_shared_memory(handles: list[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except FileNotFoundError:
            pass
        except BufferError:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _materialize_payload_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    payload_local = dict(payload)
    attached: list[shared_memory.SharedMemory] = []
    try:
        for key in ("vx", "vy", "x_faces", "y_faces"):
            shm_key = f"{key}_shm"
            if shm_key not in payload_local:
                continue
            meta = dict(payload_local.pop(shm_key))
            shm = shared_memory.SharedMemory(name=meta["name"])
            attached.append(shm)
            arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf).copy()
            payload_local[key] = arr
    finally:
        for shm in attached:
            try:
                shm.close()
            except FileNotFoundError:
                pass
    return payload_local


def gpu_call(func_name: str, payload: dict[str, Any], timeout_sec: float = 7200.0) -> dict[str, Any]:
    if global_queue_gpu is None or global_result_dict is None:
        raise RuntimeError("GPU queue/result globals are not initialized in this worker.")

    payload_ipc, shm_handles = _prepare_payload_for_ipc(payload)
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "func": func_name,
        "payload": payload_ipc,
        "batch_key": _payload_batch_key(func_name, payload_ipc),
    }
    try:
        global_queue_gpu.put(task, timeout=GPU_SUBMIT_TIMEOUT_SEC)
    except queue.Full as exc:
        _cleanup_shared_memory(shm_handles)
        raise TimeoutError(f"GPU task queue is full after waiting {GPU_SUBMIT_TIMEOUT_SEC} seconds.") from exc

    t0 = time.monotonic()
    try:
        while True:
            result = global_result_dict.pop(task_id, None)
            if result is not None:
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "unknown GPU worker error"))
                return result["result"]
            if (time.monotonic() - t0) > timeout_sec:
                raise TimeoutError(f"GPU task timeout for {func_name} after {timeout_sec} seconds")
            time.sleep(GPU_RESULT_POLL_SEC)
    finally:
        _cleanup_shared_memory(shm_handles)


def _pollock_tracker_signature(
    x_faces: np.ndarray,
    y_faces: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    n_particles: int,
) -> tuple[Any, ...]:
    return (
        tuple(x_faces.shape),
        tuple(y_faces.shape),
        tuple(vx.shape),
        tuple(vy.shape),
        str(x_faces.dtype),
        str(y_faces.dtype),
        str(vx.dtype),
        str(vy.dtype),
        int(n_particles),
    )


def _run_pollock_gpu_summary_single(payload: dict[str, Any]) -> dict[str, Any]:
    global global_pollock_tracker
    global global_pollock_tracker_signature

    modules = _import_pollock_gpu_modules()
    sim_pt_pollock_gpu = modules["sim_pt_pollock_gpu"]
    sim_pt_utils = modules["sim_pt_utils"]

    payload_local = _materialize_payload_arrays(payload)
    if str(payload_local.get("tracking_mode", "arrival")) != "arrival":
        raise ValueError("Persistent GPU Pollock currently supports tracking_mode='arrival' only.")
    if bool(payload_local.get("save_trajectories", False)):
        raise ValueError("Persistent GPU Pollock currently requires save_trajectories=False.")
    if str(payload_local.get("method", "")) != "pollock" or str(payload_local.get("backend", "")) != "gpu":
        raise ValueError("Persistent GPU execution only supports method='pollock', backend='gpu'.")

    x_faces = np.asarray(payload_local["x_faces"], dtype=np.float32)
    y_faces = np.asarray(payload_local["y_faces"], dtype=np.float32)
    vx = np.asarray(payload_local["vx"], dtype=np.float32)
    vy = np.asarray(payload_local["vy"], dtype=np.float32)
    seeds_x, seeds_y = _build_seed_arrays(payload_local)
    right_x = np.float32(payload_local.get("right_x", x_faces[-1]))
    max_points = int(payload_local.get("max_points", 2000))
    max_steps = int(payload_local.get("max_steps", payload_local.get("max_iters", max_points)))

    tracker_sig = _pollock_tracker_signature(x_faces, y_faces, vx, vy, int(seeds_x.size))
    sim_pt_pollock_gpu._ensure_ti_init()
    if global_pollock_tracker is None:
        PollockTracking2D = sim_pt_pollock_gpu._make_pollock_tracker()
        global_pollock_tracker = PollockTracking2D(
            x_faces,
            y_faces,
            vx,
            vy,
            seeds_x,
            seeds_y,
            right_x,
            max_steps,
        )
        global_pollock_tracker_signature = tracker_sig
    elif global_pollock_tracker_signature != tracker_sig:
        raise RuntimeError(
            "Persistent GPU Pollock worker already materialized a tracker for a different "
            "grid/particle shape. Use one field size and exact particle count per worker process."
        )
    else:
        tracker = global_pollock_tracker
        tracker.x_faces.from_numpy(x_faces)
        tracker.y_faces.from_numpy(y_faces)
        tracker.vx_faces.from_numpy(vx)
        tracker.vy_faces.from_numpy(vy)
        tracker.seed_x.from_numpy(seeds_x)
        tracker.seed_y.from_numpy(seeds_y)
        tracker._seeds_x = seeds_x.copy()
        tracker._seeds_y = seeds_y.copy()
        tracker._x_faces_np = x_faces
        tracker._y_faces_np = y_faces
        tracker._vx_faces_np = vx
        tracker._vy_faces_np = vy
        tracker.max_steps = int(max_steps)
        tracker.right_x = np.float32(right_x)

    tracker = global_pollock_tracker
    run_result = tracker.run()
    if len(run_result) < 2:
        raise RuntimeError(
            f"PollockTracking2D.run() returned {len(run_result)} values; expected at least travel_times and status."
        )
    raw_travel_times, status = run_result[:2]
    travel_times = np.where(status == sim_pt_pollock_gpu.STATUS_SUCCESS, raw_travel_times, np.nan).astype(np.float32)
    stop_mode = str(payload_local.get("stop_mode", "first"))
    arrival_fraction = float(payload_local.get("arrival_fraction", 1.0))
    return_fastest = bool(payload_local.get("return_fastest", stop_mode == "first"))
    summary = sim_pt_utils.summarize_arrivals(
        travel_times,
        status,
        stop_mode=stop_mode,
        arrival_fraction=arrival_fraction,
    )
    summary.update(
        {
            "backend": "gpu",
            "method": "pollock",
            "tracking_mode": "arrival",
            "stop_mode": stop_mode,
            "save_trajectories": False,
            "return_fastest": return_fastest,
            "status": np.asarray(status, dtype=np.int32),
            "lengths": None,
        }
    )
    if return_fastest and summary["n_arrived"] > 0:
        fastest_id = sim_pt_utils.fastest_particle_id(travel_times)
        traj = tracker.replay_selected(np.array([fastest_id], dtype=np.int32), right_x, max_points, max_steps)[0]
        traj["particle_id"] = int(fastest_id)
        summary["fastest"] = sim_pt_utils.build_fastest_result_from_trajectory(
            fastest_id,
            float(travel_times[fastest_id]),
            traj,
        )
    return summary


def run_gpu_batch(func_name: str, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if func_name != "particle_tracking_summary":
        raise ValueError(f"Unsupported GPU function: {func_name}")
    return [_run_pollock_gpu_summary_single(payload) for payload in payloads]


def _store_error_results(result_dict: Any, tasks: list[dict[str, Any]], message: str) -> None:
    for task in tasks:
        result_dict[task["id"]] = {"ok": False, "result": None, "error": message}


def _execute_batch(batch: list[dict[str, Any]], result_dict: Any) -> None:
    func_name = str(batch[0]["func"])
    payloads = [task["payload"] for task in batch]
    try:
        outputs = run_gpu_batch(func_name, payloads)
        if len(outputs) != len(batch):
            raise RuntimeError(f"Batched GPU function returned {len(outputs)} outputs for {len(batch)} inputs.")
    except Exception as exc:
        _store_error_results(result_dict, batch, str(exc))
        return

    for task, output in zip(batch, outputs):
        result_dict[task["id"]] = {"ok": True, "result": output, "error": ""}


def parse_visible_gpu_tokens() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env.strip():
        tokens = [token.strip() for token in env.split(",") if token.strip() and token.strip() != "-1"]
        if tokens:
            return tokens

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")]
    return [str(idx) for idx in range(len(lines))]


def gpu_worker(gpu_token: str, queue_gpu: Any, result_dict: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token

    pending: deque[dict[str, Any]] = deque()
    shutdown_requested = False

    while True:
        if pending:
            first_task = pending.popleft()
        else:
            try:
                first_task = queue_gpu.get(timeout=0.5)
            except queue.Empty:
                if shutdown_requested:
                    break
                continue

        if first_task is None:
            shutdown_requested = True
            if not pending:
                break
            continue

        batch = [first_task]
        batch_key = first_task["batch_key"]
        deadline = time.monotonic() + (MAX_BATCH_WAIT_MS / 1000.0)

        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            if pending:
                candidate = pending.popleft()
            else:
                try:
                    candidate = queue_gpu.get(timeout=timeout)
                except queue.Empty:
                    break

            if candidate is None:
                shutdown_requested = True
                continue
            if candidate["batch_key"] == batch_key:
                batch.append(candidate)
            else:
                pending.append(candidate)

        _execute_batch(batch, result_dict)
        if shutdown_requested and not pending:
            break

    while pending:
        task = pending.popleft()
        if task is not None:
            _store_error_results(result_dict, [task], "GPU worker shut down before task execution.")


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
    use_gpu_worker: bool = False,
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

    if use_gpu_worker:
        if backend != "gpu" or solver != "pollock":
            raise ValueError("Persistent GPU worker mode supports only solver='pollock', backend='gpu'.")
        summary_params.pop("seeds_x", None)
        summary_params.pop("seeds_y", None)
        summary_params["seed_spec"] = {
            "n_particles": int(n_particles),
            "x0": float(geometry.source_x),
            "y_start": float(geometry.source_y_start),
            "y_end": float(geometry.source_y_end),
        }
        summary = gpu_call("particle_tracking_summary", summary_params)
    else:
        summary = sim_pt.particle_tracking_summary(summary_params)

    result = summary.get("fastest")
    if result is None:
        raise RuntimeError("Particle tracking summary did not return the fastest trajectory.")
    fp = np.vstack((result["x"], result["y"])).T
    return np.asarray(fp, dtype=float), np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)


def _compute_first_arrival_plume(
    *,
    sim_hyd: Any,
    sim_pt: Any,
    k_field: np.ndarray,
    h_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: Geometry,
    align_with_modflow: bool,
    n_particles: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    vx_face, vy_face = sim_hyd.compute_velocity(
        k_field,
        x,
        y,
        h_field,
        output_location="face",
        align_with_modflow=align_with_modflow,
        face_output="full",
    )
    seeds_x = np.full(n_particles, float(geometry.source_x), np.float32)
    seeds_y = np.linspace(
        float(geometry.source_y_start),
        float(geometry.source_y_end),
        n_particles,
        endpoint=False,
        dtype=np.float32,
    )
    params: dict[str, Any] = {
        "method": "pollock",
        "backend": "gpu",
        "tracking_mode": "arrival",
        "stop_mode": "first",
        "save_trajectories": False,
        "return_fastest": False,
        "snapshot_mode": "fraction_of_first_arrival",
        "snapshot_times": [1.0],
        "x_faces": build_face_coordinates_from_centers(x).astype(np.float32),
        "y_faces": build_face_coordinates_from_centers(y).astype(np.float32),
        "vx": vx_face,
        "vy": vy_face,
        "seeds_x": seeds_x,
        "seeds_y": seeds_y,
        "right_x": float(geometry.target_x),
        "max_points": 2000,
        "max_steps": max_steps,
    }
    summary = sim_pt.particle_tracking_summary(params)
    snaps = summary.get("snapshots")
    if snaps is None:
        return np.full(n_particles, np.nan), np.full(n_particles, np.nan)
    snap_x = np.asarray(snaps["x"], dtype=float)
    snap_y = np.asarray(snaps["y"], dtype=float)
    return snap_x[:, 0].copy(), snap_y[:, 0].copy()


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

    vx_interp = RegularGridInterpolator((y_coords, x_coords), vx_field, bounds_error=False, fill_value=np.nan)
    vy_interp = RegularGridInterpolator((y_coords, x_coords), vy_field, bounds_error=False, fill_value=np.nan)

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
    save_computation_time: bool,
    save_first_arrival_plume: bool,
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
    cmd.append("--save-computation-time" if save_computation_time else "--no-save-computation-time")
    cmd.append("--save-first-arrival-plume" if save_first_arrival_plume else "--no-save-first-arrival-plume")
    subprocess.run(cmd, check=True)


def _build_sample_stem(config: RunConfig, seed: int) -> str:
    stem = f"{config.field_type}_size{config.field_size}_seed{seed}"
    if config.field_type == "expcov":
        stem = f"{stem}_ls{config.length_scale}_std{config.std}"
    elif config.field_type == "fbm":
        stem = f"{stem}_alpha{config.alpha}_std{config.std}"
    return stem


def _resolve_visualization_dir(npz_path: Path, visualize_cfg: VisualizationConfig, batch_mode: bool) -> str | None:
    if not visualize_cfg.output_dir:
        return None
    output_root = Path(visualize_cfg.output_dir).expanduser()
    if batch_mode:
        return str((output_root / f"{npz_path.stem}_figures").resolve())
    return str(output_root.resolve())


def _write_output(
    *,
    output_path: Path,
    field_type: str,
    field_size: int,
    seed: int,
    dh: float,
    lrp_method: str,
    std: float,
    grf_length_scale: float,
    fbm_alpha: float,
    fbm_c: float,
    x: np.ndarray,
    y: np.ndarray,
    logk: np.ndarray,
    h: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    fp: np.ndarray,
    fp_arc_length: np.ndarray,
    fp_travel_time: np.ndarray,
    lrp: np.ndarray,
    lrp_arc_length: np.ndarray,
    lrp_pseudo_travel_time: np.ndarray,
    head_solve_time_sec: float,
    velocity_solve_time_sec: float,
    fp_compute_time_sec: float,
    lrp_compute_time_sec: float,
    fp_total_time_sec: float,
    lrp_total_time_sec: float,
    geometry: Geometry,
    plume_x: np.ndarray,
    plume_y: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        field_type=np.array(field_type),
        field_size=np.array(field_size),
        seed=np.array(seed),
        dh=np.array(dh),
        lrp_method=np.array(lrp_method),
        grf_std=np.array(std),
        grf_length_scale=np.array(grf_length_scale),
        fbm_alpha=np.array(fbm_alpha),
        fbm_c=np.array(fbm_c),
        x=x,
        y=y,
        logk=logk,
        h=h,
        vx=vx,
        vy=vy,
        fp=fp,
        fp_arc_length=fp_arc_length,
        fp_travel_time=fp_travel_time,
        fp_compute_time_sec=np.array(fp_compute_time_sec),
        fp_total_time_sec=np.array(fp_total_time_sec),
        lrp=lrp,
        lrp_arc_length=lrp_arc_length,
        lrp_pseudo_travel_time=lrp_pseudo_travel_time,
        lrp_compute_time_sec=np.array(lrp_compute_time_sec),
        lrp_total_time_sec=np.array(lrp_total_time_sec),
        head_solve_time_sec=np.array(head_solve_time_sec),
        velocity_solve_time_sec=np.array(velocity_solve_time_sec),
        source_x=np.array(geometry.source_x),
        target_x=np.array(geometry.target_x),
        source_y_start=np.array(geometry.source_y_start),
        source_y_end=np.array(geometry.source_y_end),
        plume_x=plume_x,
        plume_y=plume_y,
    )


def _run_single_seed(
    *,
    config: RunConfig,
    seed: int,
    output_path: Path,
    visualize_cfg: VisualizationConfig,
    use_gpu_worker: bool,
    batch_mode: bool,
) -> Path:
    flow_modules = _get_flow_modules()
    sim_grf = flow_modules["sim_grf"]
    sim_hyd = flow_modules["sim_hyd"]
    sim_path_finding = flow_modules["sim_path_finding"]
    sim_pt = flow_modules["sim_pt"]

    geometry = default_geometry(config.field_size)
    x = geometry.x
    y = geometry.y

    if config.field_type == "expcov":
        logk = generate_logk_expcov(
            sim_grf,
            field_size=config.field_size,
            std=config.std,
            length_scale=config.length_scale,
            seed=seed,
        )
        grf_length_scale = float(config.length_scale)
        fbm_alpha = np.nan
    elif config.field_type == "fbm":
        logk = generate_logk_fbm(
            sim_grf,
            field_size=config.field_size,
            alpha=config.alpha,
            std_target=config.std,
            seed=seed,
            c=config.fbm_c,
        )
        grf_length_scale = np.nan
        fbm_alpha = float(config.alpha)
        fbm_c = float(config.fbm_c) if config.fbm_c is not None else float(
            sim_grf.compute_c(config.field_size, H=config.alpha / 2, std_target=config.std)
        )
    else:
        raise ValueError(f"Unsupported field_type={config.field_type!r}")

    k_field = np.exp(logk)
    t_head_start = time.perf_counter()
    h = _solve_head_field(sim_hyd, k_field, x, y, config.dh)
    head_solve_time_sec = float(time.perf_counter() - t_head_start)

    t_velocity_start = time.perf_counter()
    grad_h_y, grad_h_x = np.gradient(h, y, x, edge_order=2)
    vx_cell = -k_field * grad_h_x
    vy_cell = -k_field * grad_h_y
    velocity_solve_time_sec = float(time.perf_counter() - t_velocity_start)

    dh_mean = abs(float(config.dh))
    if dh_mean <= 0.0 or not np.isfinite(dh_mean):
        raise ValueError(f"Invalid dh={config.dh}; expected a positive finite magnitude.")

    lrp_speed = k_field * dh_mean
    t_lrp_start = time.perf_counter()
    if config.lrp_method == "dijkstra":
        lrp, _t_lrp = _compute_dijkstra_path(
            lrp_speed,
            geometry=geometry,
            sim_path_finding=sim_path_finding,
        )
    elif config.lrp_method == "fmm":
        fmm_modules = _get_fmm_modules()
        lrp, _t_lrp = _compute_fmm_path(
            lrp_speed,
            geometry=geometry,
            grid_spacing=1.0,
            fast_marching_2d_regions=fmm_modules["fast_marching_2d_regions"],
            extract_path_source_to_target=fmm_modules["extract_path_source_to_target"],
            extract_path_continuous=fmm_modules["extract_path_continuous"],
            backtracking=config.fmm_backtracking,
            continuous_step_size=config.fmm_continuous_step_size,
        )
    else:
        raise ValueError(f"Unsupported LRP method: {config.lrp_method!r}")
    lrp_compute_time_sec = float(time.perf_counter() - t_lrp_start)

    t_fp_start = time.perf_counter()
    fp, _vx_track, _vy_track = _run_particle_tracking(
        sim_hyd=sim_hyd,
        sim_pt=sim_pt,
        k_field=k_field,
        h_field=h,
        x=x,
        y=y,
        geometry=geometry,
        solver=config.solver,
        backend=config.backend,
        velocity_location=config.velocity_location,
        align_with_modflow_velocity=config.align_with_modflow,
        density_particle=config.density_particle,
        num_particles=config.num_particles,
        eps_speed=config.eps_speed,
        ds_min=config.ds_min,
        ds_max=config.ds_max,
        tol=config.tol,
        y_policy=config.y_policy,
        max_points=config.max_points,
        max_iters=config.max_iters,
        use_gpu_worker=use_gpu_worker,
    )
    fp_compute_time_sec = float(time.perf_counter() - t_fp_start)

    fp_total_time_sec = head_solve_time_sec + velocity_solve_time_sec + fp_compute_time_sec
    lrp_total_time_sec = head_solve_time_sec + velocity_solve_time_sec + lrp_compute_time_sec

    fp_arc_length, fp_travel_time = compute_cumulative_travel_time(fp, vx_cell, vy_cell, x, y)
    lrp_arc_length, lrp_pseudo_travel_time = compute_cumulative_travel_time(lrp, vx_cell, vy_cell, x, y)

    plume_x, plume_y = _compute_first_arrival_plume(
        sim_hyd=sim_hyd,
        sim_pt=sim_pt,
        k_field=k_field,
        h_field=h,
        x=x,
        y=y,
        geometry=geometry,
        align_with_modflow=config.align_with_modflow,
        n_particles=_resolve_particle_count(
            num_particles=config.num_particles,
            density_particle=config.density_particle,
            y0=geometry.source_y_start,
            y1=geometry.source_y_end,
        ),
        max_steps=config.max_iters,
    )

    output_path = output_path.expanduser().resolve()
    _write_output(
        output_path=output_path,
        field_type=config.field_type,
        field_size=config.field_size,
        seed=seed,
        dh=config.dh,
        lrp_method=config.lrp_method,
        std=config.std,
        grf_length_scale=grf_length_scale,
        fbm_alpha=fbm_alpha,
        fbm_c=fbm_c if config.field_type == "fbm" else np.nan,
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
        head_solve_time_sec=head_solve_time_sec,
        velocity_solve_time_sec=velocity_solve_time_sec,
        fp_compute_time_sec=fp_compute_time_sec,
        lrp_compute_time_sec=lrp_compute_time_sec,
        fp_total_time_sec=fp_total_time_sec,
        lrp_total_time_sec=lrp_total_time_sec,
        geometry=geometry,
        plume_x=plume_x,
        plume_y=plume_y,
    )
    if visualize_cfg.enabled:
        _run_visualization(
            npz_path=output_path,
            output_dir=_resolve_visualization_dir(output_path, visualize_cfg, batch_mode=batch_mode),
            dpi=visualize_cfg.dpi,
            figure_format=visualize_cfg.figure_format,
            stream_density=visualize_cfg.stream_density,
            path_linewidth=visualize_cfg.path_linewidth,
            save_logk_paths=visualize_cfg.save_logk_paths,
            save_head_velocity=visualize_cfg.save_head_velocity,
            save_travel_time=visualize_cfg.save_travel_time,
            save_computation_time=visualize_cfg.save_computation_time,
            save_first_arrival_plume=visualize_cfg.save_first_arrival_plume,
        )
    return output_path


def _batch_worker_initializer(queue_gpu: Any, result_dict: Any) -> None:
    global global_queue_gpu
    global global_result_dict
    global_queue_gpu = queue_gpu
    global_result_dict = result_dict
    _get_flow_modules()


def _run_batch_job(job: tuple[RunConfig, int, str, VisualizationConfig, bool]) -> dict[str, Any]:
    config, seed, output_path_str, visualize_cfg, use_gpu_worker = job
    try:
        path = _run_single_seed(
            config=config,
            seed=seed,
            output_path=Path(output_path_str),
            visualize_cfg=visualize_cfg,
            use_gpu_worker=use_gpu_worker,
            batch_mode=True,
        )
    except Exception as exc:
        return {
            "seed": int(seed),
            "ok": False,
            "output_path": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    return {
        "seed": int(seed),
        "ok": True,
        "output_path": str(path),
        "error_type": "",
        "error_message": "",
        "traceback": "",
    }


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        cpu_count = os.cpu_count()
        return int(cpu_count) if cpu_count is not None else 1


def _write_batch_summary(
    *,
    output_dir: Path,
    config: RunConfig,
    requested_mc: int,
    initial_seeds: list[int],
    failure_policy: str,
    results: list[dict[str, Any]],
) -> Path:
    summary_path = output_dir / "batch_summary.json"
    successful_outputs = [
        {
            "seed": int(record["seed"]),
            "output_path": str(record["output_path"]),
        }
        for record in results
        if bool(record["ok"])
    ]
    failed_attempts = [
        {
            "seed": int(record["seed"]),
            "error_type": str(record["error_type"]),
            "error_message": str(record["error_message"]),
        }
        for record in results
        if not bool(record["ok"])
    ]
    summary = {
        "field_type": config.field_type,
        "field_size": int(config.field_size),
        "requested_mc": int(requested_mc),
        "initial_seeds": [int(seed) for seed in initial_seeds],
        "failure_policy": failure_policy,
        "attempted_seeds": [int(record["seed"]) for record in results],
        "num_attempted": len(results),
        "num_succeeded": len(successful_outputs),
        "num_failed": len(failed_attempts),
        "successful_outputs": successful_outputs,
        "failed_attempts": failed_attempts,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def _run_batch_mode(
    *,
    config: RunConfig,
    seeds: list[int],
    output_dir: Path,
    visualize_cfg: VisualizationConfig,
    cpu_workers: int | None,
    gpu_device: str | None,
    failure_policy: str,
) -> list[Path]:
    mp.set_start_method("spawn", force=True)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_mc = len(seeds)
    initial_seeds = list(seeds)
    worker_count = int(cpu_workers) if cpu_workers is not None else min(len(initial_seeds), _available_cpu_count())
    worker_count = max(1, min(max(1, len(initial_seeds)), worker_count))

    manager = None
    result_dict = None
    queue_gpu = None
    process_gpu_worker = None
    all_results: list[dict[str, Any]] = []
    successful_paths: list[Path] = []
    attempted_seeds: set[int] = set()
    next_seed = (max(initial_seeds) + 1) if initial_seeds else 0
    try:
        if config.backend == "gpu":
            gpu_tokens = [gpu_device] if gpu_device else parse_visible_gpu_tokens()
            if not gpu_tokens:
                raise RuntimeError("No visible GPU found for batch GPU mode.")
            manager = mp.Manager()
            result_dict = manager.dict()
            queue_gpu = mp.Queue(maxsize=GPU_QUEUE_MAXSIZE)
            process_gpu_worker = mp.Process(
                target=gpu_worker,
                args=(gpu_tokens[0], queue_gpu, result_dict),
                name=f"GPU-Worker-{gpu_tokens[0]}",
            )
            process_gpu_worker.start()

        with mp.Pool(
            processes=worker_count,
            initializer=_batch_worker_initializer,
            initargs=(queue_gpu, result_dict),
        ) as pool:
            pending_seeds = list(initial_seeds)
            while pending_seeds:
                jobs = [
                    (
                        config,
                        seed,
                        str((output_dir / f"{_build_sample_stem(config, seed)}.npz").resolve()),
                        visualize_cfg,
                        config.backend == "gpu",
                    )
                    for seed in pending_seeds
                ]
                batch_results = pool.map(_run_batch_job, jobs)
                all_results.extend(batch_results)
                for record in batch_results:
                    attempted_seeds.add(int(record["seed"]))
                    if bool(record["ok"]):
                        successful_paths.append(Path(str(record["output_path"])))

                if failure_policy == "skip":
                    break

                remaining = requested_mc - len(successful_paths)
                if remaining <= 0:
                    break

                pending_seeds = []
                while len(pending_seeds) < remaining:
                    if next_seed in attempted_seeds:
                        next_seed += 1
                        continue
                    pending_seeds.append(next_seed)
                    attempted_seeds.add(next_seed)
                    next_seed += 1
    finally:
        if queue_gpu is not None:
            queue_gpu.put(None)
        if process_gpu_worker is not None:
            process_gpu_worker.join(timeout=30)
            if process_gpu_worker.is_alive():
                process_gpu_worker.terminate()
                process_gpu_worker.join(timeout=10)
        if manager is not None:
            manager.shutdown()

    summary_path = _write_batch_summary(
        output_dir=output_dir,
        config=config,
        requested_mc=requested_mc,
        initial_seeds=initial_seeds,
        failure_policy=failure_policy,
        results=all_results,
    )
    print(
        f"[batch] failure_policy={failure_policy} requested_mc={requested_mc} "
        f"attempted={len(all_results)} succeeded={len(successful_paths)} "
        f"failed={len(all_results) - len(successful_paths)}"
    )
    print(f"[batch] summary={summary_path}")
    return successful_paths


def run(
    *,
    config: RunConfig,
    seed: int,
    output_path: Path,
    visualize_cfg: VisualizationConfig,
) -> Path:
    return _run_single_seed(
        config=config,
        seed=seed,
        output_path=output_path,
        visualize_cfg=visualize_cfg,
        use_gpu_worker=False,
        batch_mode=False,
    )


def _resolve_batch_seeds(args: argparse.Namespace) -> list[int] | None:
    if args.batch_seeds is not None:
        return list(args.batch_seeds)
    if args.num_mc is None:
        return None
    return [int(args.seed_start) + idx for idx in range(int(args.num_mc))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate logK fields, FP/LRP paths, and pseudo travel times.")
    parser.add_argument("--field-type", choices=("expcov", "fbm"), required=True)
    parser.add_argument("--field-size", type=int, default=100)
    parser.add_argument("--std", type=float, default=1.0, help="Target logK standard deviation.")
    parser.add_argument("--length-scale", type=float, default=0.2, help="Exponential covariance length scale in normalized units.")
    parser.add_argument("--fbm-alpha", type=float, default=1.0, help="fBm alpha = 2 * Hurst.")
    parser.add_argument("--fbm-c", type=float, default=None, help="Optional fBm scale parameter override. If omitted, it is derived from --fbm-alpha and --std.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-seeds", nargs="+", type=int, default=None, help="Batch mode: run one sample per seed.")
    parser.add_argument("--num-mc", type=int, default=None, help="Batch mode convenience option: number of Monte Carlo seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Batch mode convenience option: first seed used with --num-mc.")
    parser.add_argument("--dh", type=float, default=1.0)
    parser.add_argument("--lrp-method", choices=("dijkstra", "fmm"), default="dijkstra")
    parser.add_argument("--solver", choices=("rk45", "pollock"), default="rk45")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--velocity-location", choices=("cell", "face"), default="cell")
    parser.add_argument("--align-with-modflow", action="store_true", default=False)
    parser.add_argument("--density-particle", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=1000000)
    parser.add_argument("--eps-speed", type=float, default=1e-12)
    parser.add_argument("--ds-min", type=float, default=0.05)
    parser.add_argument("--ds-max", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--y-policy", choices=("clip", "terminate"), default="clip")
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--max-iters", type=int, default=200000)
    parser.add_argument("--fmm-backtracking", choices=("discrete", "continuous"), default="continuous")
    parser.add_argument("--fmm-continuous-step-size", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Single-run mode: output .npz path.")
    parser.add_argument("--output-dir", default=None, help="Batch mode: output directory for per-seed .npz files.")
    parser.add_argument("--cpu-workers", type=int, default=None, help="Batch mode: CPU worker count.")
    parser.add_argument("--gpu-device", default=None, help="Batch mode: pin the persistent GPU worker to a specific visible GPU token.")
    parser.add_argument("--failure-policy", choices=("retry", "skip"), default="retry", help="Batch mode: retry with fresh seeds until MC successes, or skip failed seeds.")
    parser.add_argument("--visualize", action="store_true", default=False, help="After saving outputs, call visualize_lrp_fp.py to save figures.")
    parser.add_argument("--visualize-output-dir", default=None, help="Optional figure output directory. In batch mode, one subdirectory per sample is created here.")
    parser.add_argument("--visualize-dpi", type=int, default=300)
    parser.add_argument("--visualize-format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--visualize-stream-density", type=float, default=1.0)
    parser.add_argument("--visualize-path-linewidth", type=float, default=2.0)
    parser.add_argument("--visualize-logk-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-head-velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-travel-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-computation-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-first-arrival-plume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_mc is not None and args.num_mc <= 0:
        raise ValueError("--num-mc must be positive.")
    if args.field_type == "expcov" and args.length_scale <= 0.0:
        raise ValueError("--length-scale must be positive for expcov fields.")
    if args.field_type == "fbm" and not (0.0 < args.fbm_alpha < 2.0):
        raise ValueError("--fbm-alpha must lie in (0, 2).")
    if args.fbm_c is not None and args.fbm_c <= 0.0:
        raise ValueError("--fbm-c must be positive.")
    if args.cpu_workers is not None and args.cpu_workers <= 0:
        raise ValueError("--cpu-workers must be positive.")
    if args.batch_seeds is not None and args.num_mc is not None:
        raise ValueError("Use either --batch-seeds or --num-mc/--seed-start, not both.")

    batch_seeds = _resolve_batch_seeds(args)
    if batch_seeds:
        if args.output is not None:
            raise ValueError("--output cannot be used with batch mode; use --output-dir instead.")
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --batch-seeds or --num-mc.")
        if args.backend == "gpu" and args.solver != "pollock":
            raise ValueError("Batch GPU mode currently supports --solver pollock only.")
    else:
        if args.output_dir is not None:
            raise ValueError("--output-dir is only valid with --batch-seeds or --num-mc.")
        if args.output is None:
            raise ValueError("--output is required for single-run mode.")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    batch_seeds = _resolve_batch_seeds(args)

    config = RunConfig(
        field_type=args.field_type,
        field_size=args.field_size,
        std=args.std,
        length_scale=args.length_scale,
        alpha=args.fbm_alpha,
        fbm_c=args.fbm_c,
        dh=args.dh,
        lrp_method=args.lrp_method,
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
    )
    visualize_cfg = VisualizationConfig(
        enabled=bool(args.visualize),
        output_dir=args.visualize_output_dir,
        dpi=args.visualize_dpi,
        figure_format=args.visualize_format,
        stream_density=args.visualize_stream_density,
        path_linewidth=args.visualize_path_linewidth,
        save_logk_paths=bool(args.visualize_logk_paths),
        save_head_velocity=bool(args.visualize_head_velocity),
        save_travel_time=bool(args.visualize_travel_time),
        save_computation_time=bool(args.visualize_computation_time),
        save_first_arrival_plume=bool(args.visualize_first_arrival_plume),
    )

    if batch_seeds:
        _run_batch_mode(
            config=config,
            seeds=batch_seeds,
            output_dir=Path(args.output_dir),
            visualize_cfg=visualize_cfg,
            cpu_workers=args.cpu_workers,
            gpu_device=args.gpu_device,
            failure_policy=args.failure_policy,
        )
        return

    run(
        config=config,
        seed=args.seed,
        output_path=Path(args.output),
        visualize_cfg=visualize_cfg,
    )


if __name__ == "__main__":
    main()
