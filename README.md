# LRP_FP

`LRP_FP` is a small standalone workspace for generating log-permeability fields, solving flow, extracting `FP` and `LRP`, saving results as `.npz`, and optionally producing figure outputs.

Main scripts:

- `scripts/generate_lrp_fp.py`: generate one field or a seed batch, solve flow, compute `FP` and `LRP`, save `.npz`, and optionally trigger visualization
- `scripts/visualize_lrp_fp.py`: load one saved `.npz` and save figure files
- `scripts/submit_generate_lrp_fp.sh`: submit a SLURM job for generation and optional visualization

All implementation logic lives in the standalone scripts.

## Repository Layout

- `scripts/generate_lrp_fp.py`: single-run and seed-batch generation, Darcy solve, particle tracking, FMM path extraction, `.npz` output, optional post-save visualization
- `scripts/visualize_lrp_fp.py`: figure generation from a saved `.npz`
- `scripts/submit_generate_lrp_fp.sh`: SLURM submission wrapper for batch runs
- `outputs/`: generated `.npz`, figures, and SLURM logs

## Agent Workflow

- Read `AGENT.md` first, then follow `agent/guide.md`.
- Append task entries to `agent/log.md` for each meaningful change.

## Dependencies

`generate_lrp_fp.py` now uses a split dependency model:

- `FlowSimulation` is the primary runtime dependency for:
  - GRF / fBm generation
  - hydraulic head solve
  - velocity computation
  - particle tracking
- `FMM_FastestPath/Algorithm/src/fmm_core` is used only for the LRP FMM solve

Environment overrides:

- `FLOWSIM_PATH`: override the default sibling `FlowSimulation` path
- `FMM_CORE_PATH`: override the default sibling `FMM_FastestPath/Algorithm/src` path

Default local layout assumed by the scripts:

- repo checkout: `/home1/binhaoli/LogFBM`
- FlowSimulation: `/home1/binhaoli/FlowSimulation`
- FMM core source: `/home1/binhaoli/FMM_FastestPath/Algorithm/src`

The hydraulic solve uses `petsc4py` through `FlowSimulation`.

## Quick Start

Generate one field and save the `.npz`:

```bash
python scripts/generate_lrp_fp.py \
  --field-type fbm \
  --field-size 100 \
  --fbm-alpha 1.0 \
  --seed 42 \
  --output outputs/test_fbm.npz
```

Run a batch over several seeds and write one `.npz` per seed:

```bash
python scripts/generate_lrp_fp.py \
  --field-type fbm \
  --field-size 100 \
  --fbm-alpha 1.0 \
  --fbm-c 0.25 \
  --backend gpu \
  --solver pollock \
  --velocity-location face \
  --failure-policy retry \
  --num-mc 4 \
  --seed-start 42 \
  --output-dir outputs/test_fbm_batch \
  --cpu-workers 4
```

Generate and immediately save figures:

```bash
python scripts/generate_lrp_fp.py \
  --field-type fbm \
  --field-size 100 \
  --fbm-alpha 1.0 \
  --seed 42 \
  --output outputs/test_fbm.npz \
  --visualize
```

Visualize an existing `.npz` directly:

```bash
python scripts/visualize_lrp_fp.py \
  --input outputs/test_fbm.npz
```

Submit the batch workflow:

```bash
bash scripts/submit_generate_lrp_fp.sh
```

The submission wrapper keeps the code checkout path (`repo_root`) separate from the result destination (`output_root_base`), so SLURM jobs can run from the repository while writing `.npz`, figures, and logs to an external directory.

## Output Files

`generate_lrp_fp.py` saves one `.npz` per sample with:

- metadata: field type, seed, field size, `dh`, GRF/fBm parameters
- fields: `logk`, `h`, `vx`, `vy`
- paths: `fp`, `lrp`
- path summaries: `fp_arc_length`, `fp_travel_time`, `lrp_arc_length`, `lrp_pseudo_travel_time`
- computation times:
  - `head_solve_time_sec`
  - `velocity_solve_time_sec`
  - `fp_compute_time_sec`
  - `lrp_compute_time_sec`
  - `fp_total_time_sec`
  - `lrp_total_time_sec`
- geometry metadata for the source and target boundaries

In batch mode, files are written under `--output-dir` and named from the fixed parameter set plus each seed, for example:

- `fbm_size100_seed42_alpha1.0_std1.0.npz`
- `fbm_size100_seed43_alpha1.0_std1.0.npz`

For `fbm` fields, the saved metadata includes both `fbm_alpha` and `fbm_c`. If `--fbm-c` is omitted, `fbm_c` is derived from `--fbm-alpha` and `--std`.

Batch mode also writes `batch_summary.json`, which records all attempted seeds, successful outputs, and failed attempts.

With `python scripts/visualize_lrp_fp.py --input <file.npz>`, the script saves up to four per-file figures:

- `logk_paths`
- `head_velocity`
- `travel_time_profiles`
- `computation_time_profiles`

By default, figures are saved in `<input_stem>_figures/` beside the input `.npz`.
If `--visualize-output-dir` is set in batch mode, one subdirectory per sample is created inside that directory.

Computation-time definitions:

- FP computation time is particle-tracking time only
- LRP computation time is FMM time only
- head and velocity solve times are recorded separately
- inclusive totals add head and velocity solve times to FP or LRP for comparison

Aggregate comparison figures:

- `python scripts/visualize_lrp_fp.py --input-dir <dir>` scans all `.npz` files directly inside the directory
- `travel_time_comparison` is a scatter plot with one point per `.npz`, using final dimensionless travel times `t_FP / t_ref` versus `t_LRP / t_ref`, where `t_ref = |dh|` is derived from each `.npz`
- `computation_time_comparison` is a scatter plot with one point per `.npz` for exclusive timing and one point per `.npz` for inclusive timing, with a `y=x` reference line

## Visualization Options

`generate_lrp_fp.py` can forward figure settings to `visualize_lrp_fp.py`:

- `--visualize`
- `--visualize-output-dir`
- `--visualize-dpi`
- `--visualize-format`
- `--visualize-stream-density`
- `--visualize-path-linewidth`
- `--visualize-logk-paths` / `--no-visualize-logk-paths`
- `--visualize-head-velocity` / `--no-visualize-head-velocity`
- `--visualize-travel-time` / `--no-visualize-travel-time`
- `--visualize-computation-time` / `--no-visualize-computation-time`

Aggregate comparison-only options in `scripts/visualize_lrp_fp.py`:

- `--input-dir`
- `--save-travel-time-comparison` / `--no-save-travel-time-comparison`
- `--save-computation-time-comparison` / `--no-save-computation-time-comparison`

Mode-aware defaults:

- with `--input <file>`, aggregate comparison plots default to off unless explicitly enabled
- with `--input-dir <dir>`, per-file plots default to off unless explicitly enabled

## Batch Mode

`generate_lrp_fp.py` supports two execution modes:

- single-run mode: use `--seed` plus `--output`
- batch mode: use `--batch-seeds ...` or `--num-mc N --seed-start S`, plus `--output-dir`

Batch mode parallelizes over seeds only. Physics and solver parameters remain scalar for the whole invocation.

Seed controls:

- `--batch-seeds 42 43 44 45` lets you specify the exact seed list
- `--num-mc 4 --seed-start 42` expands to seeds `42 43 44 45`
- `--seed` is still only for single-run mode

fBm controls:

- `--fbm-alpha` sets the Hurst-related exponent
- `--fbm-c` optionally overrides the scale parameter used by the fBm generator
- if `--fbm-c` is not provided, the script computes it from `--fbm-alpha` and `--std`

When batch mode runs with `--backend gpu`, the script uses FMM-style CPU/GPU orchestration:

- a CPU `multiprocessing.Pool` handles field generation, head solve, FMM/LRP, travel-time postprocessing, and output writing
- a persistent GPU worker process handles Pollock particle tracking requests
- large particle-tracking arrays are transferred through shared memory
- compatible GPU tasks are batched and reuse a persistent Pollock tracker

Batch GPU restrictions:

- `--backend gpu` in batch mode currently requires `--solver pollock`
- `--solver pollock` requires `--velocity-location face`

Batch resource controls:

- `--cpu-workers`: optional CPU worker count override
- `--gpu-device`: optional visible GPU token for the persistent GPU worker
- `--failure-policy {retry,skip}`:
  - `retry`: keep trying fresh seeds until the number of successful `.npz` files equals `MC`
  - `skip`: attempt only the requested seeds once; failed seeds are logged in `batch_summary.json` and produce no `.npz`

## Notes

- `generate_lrp_fp.py` requires `FlowSimulation` plus the FMM core source tree to be available at runtime.
- If `petsc4py`, `FlowSimulation`, or `fmm_core` are missing, generation will fail before `.npz` output is produced.
- Batch GPU mode additionally requires a visible CUDA device and the FlowSimulation Pollock GPU dependencies.
- `visualize_lrp_fp.py` works only from an already generated `.npz` and does not require rerunning the solver.
