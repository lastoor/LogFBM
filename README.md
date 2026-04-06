# LRP_FP

`LRP_FP` is a small standalone workspace for generating one log-permeability field, solving flow, extracting `FP` and `LRP`, saving the result as `.npz`, and optionally producing figure outputs.

Main scripts:

- `generate_lrp_fp.py`: generate one `expcov` or `fbm` field, solve flow, compute `FP` and `LRP`, save `.npz`, and optionally trigger visualization
- `visualize_lrp_fp.py`: load one saved `.npz` and save figure files
- `submit_generate_lrp_fp.sh`: submit a SLURM job for generation and optional visualization

All implementation logic lives in the standalone scripts.

## Repository Layout

- `generate_lrp_fp.py`: field generation, Darcy solve, particle tracking, FMM path extraction, `.npz` output, optional post-save visualization
- `visualize_lrp_fp.py`: figure generation from a saved `.npz`
- `submit_generate_lrp_fp.sh`: SLURM submission wrapper for batch runs
- `outputs/`: generated `.npz`, figures, and SLURM logs

## Agent Workflow

- Read `AGENT.md` first, then follow `agent/guide.md`.
- Append task entries to `agent/log.md` for each meaningful change.

## Dependencies

Runtime depends on the sibling solver repositories used by `generate_lrp_fp.py`:

- `FMM_FastestPath/Algorithm/src`
- `FlowSimulation`

Set `FLOWSIM_PATH` if the default sibling `FlowSimulation` path is not correct.

The hydraulic solve uses `petsc4py` through `FlowSimulation`.

## Quick Start

Generate one field and save the `.npz`:

```bash
python generate_lrp_fp.py \
  --field-type fbm \
  --field-size 100 \
  --fbm-alpha 1.0 \
  --seed 42 \
  --output outputs/test_fbm.npz
```

Generate and immediately save figures:

```bash
python generate_lrp_fp.py \
  --field-type fbm \
  --field-size 100 \
  --fbm-alpha 1.0 \
  --seed 42 \
  --output outputs/test_fbm.npz \
  --visualize
```

Visualize an existing `.npz` directly:

```bash
python visualize_lrp_fp.py \
  --input outputs/test_fbm.npz
```

Submit the batch workflow:

```bash
bash submit_generate_lrp_fp.sh
```

## Output Files

`generate_lrp_fp.py` saves one `.npz` with:

- metadata: field type, seed, field size, `dh`, GRF/fBm parameters
- fields: `logk`, `h`, `vx`, `vy`
- paths: `fp`, `lrp`
- path summaries: `fp_arc_length`, `fp_travel_time`, `lrp_arc_length`, `lrp_pseudo_travel_time`
- geometry metadata for the source and target boundaries

`visualize_lrp_fp.py` saves up to three figures per `.npz`:

- `logk_paths`
- `head_velocity`
- `travel_time_profiles`

By default, figures are saved in `<input_stem>_figures/` beside the input `.npz`.

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

## Notes

- `generate_lrp_fp.py` requires the external solver stack to be available at runtime.
- If `petsc4py` or the sibling solver repositories are missing, generation will fail before `.npz` output is produced.
- `visualize_lrp_fp.py` works only from an already generated `.npz` and does not require rerunning the solver.
