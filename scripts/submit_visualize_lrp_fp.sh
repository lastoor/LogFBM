#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Input mode: set exactly one of input_file or input_dir.
#   input_file  -> per-file figures (logk_paths, head_velocity, travel_time,
#                  computation_time)
#   input_dir   -> aggregate comparison figures (travel_time_comparison scatter
#                  and/or histogram, computation_time_comparison)
# ---------------------------------------------------------------------------
input_file=""
input_dir="/project2/fbarros_324/binhaoli/task7_lrp_fp/Run3_computation_time/outputs/generate_lrp_fp/fbm_size100_mc1000_seedstart2026_alpha0.1_stdauto"
output_dir="/home1/binhaoli/LogFBM/outputs/generate_lrp_fp/fbm_size100_mc1000_seedstart2026_alpha0.1_stdauto"
log_dir="/home1/binhaoli/LogFBM/outputs/generate_lrp_fp/slurm_logs"

dpi="300"
format="png"                # Options: png, pdf, svg
stream_density="1.0"
path_linewidth="2.0"

# Per-file figure toggles (only used with input_file)
save_logk_paths="true"
save_head_velocity="true"
save_travel_time="true"
save_computation_time="true"

# Aggregate figure toggles (only used with input_dir)
save_travel_time_comparison="true"
travel_time_comparison_style="both"        # Options: scatter, histogram, both
save_computation_time_comparison="true"
computation_time_comparison_style="histogram"   # Options: scatter, histogram, both
computation_time_comparison_components="self" # Options: self, total, both

run_tag="visualize_lrp_fp"
time="00:10:00"
cpus_per_task="2"
mem_per_cpu="8GB"

account="fbarros_324"
partition="main"

repo_root="/home1/binhaoli/LogFBM"
mkdir -p "${log_dir}"

# ---------------------------------------------------------------------------
# Validate input mode
# ---------------------------------------------------------------------------
if [[ -n "${input_file}" && -n "${input_dir}" ]]; then
    echo "Error: set exactly one of input_file or input_dir, not both." >&2
    exit 1
fi
if [[ -z "${input_file}" && -z "${input_dir}" ]]; then
    echo "Error: one of input_file or input_dir must be set." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build job suffix and input argument
# ---------------------------------------------------------------------------
if [[ -n "${input_file}" ]]; then
    mode="file"
    input_arg="--input ${input_file}"
    job_suffix="file_$(basename "${input_file}" .npz)"
else
    mode="dir"
    input_arg="--input-dir ${input_dir}"
    job_suffix="dir_$(basename "${input_dir}")"
fi

# ---------------------------------------------------------------------------
# Build optional arguments
# ---------------------------------------------------------------------------
output_dir_args=""
if [[ -n "${output_dir}" ]]; then
    output_dir_args="--output-dir ${output_dir}"
fi

per_file_args=""
if [[ "${mode}" == "file" ]]; then
    if [[ "${save_logk_paths}" == "true" ]]; then
        per_file_args+=" --save-logk-paths"
    else
        per_file_args+=" --no-save-logk-paths"
    fi
    if [[ "${save_head_velocity}" == "true" ]]; then
        per_file_args+=" --save-head-velocity"
    else
        per_file_args+=" --no-save-head-velocity"
    fi
    if [[ "${save_travel_time}" == "true" ]]; then
        per_file_args+=" --save-travel-time"
    else
        per_file_args+=" --no-save-travel-time"
    fi
    if [[ "${save_computation_time}" == "true" ]]; then
        per_file_args+=" --save-computation-time"
    else
        per_file_args+=" --no-save-computation-time"
    fi
fi

aggregate_args=""
if [[ "${mode}" == "dir" ]]; then
    if [[ "${save_travel_time_comparison}" == "true" ]]; then
        aggregate_args+=" --save-travel-time-comparison"
        aggregate_args+=" --travel-time-comparison-style ${travel_time_comparison_style}"
    else
        aggregate_args+=" --no-save-travel-time-comparison"
    fi
    if [[ "${save_computation_time_comparison}" == "true" ]]; then
        aggregate_args+=" --save-computation-time-comparison"
        aggregate_args+=" --computation-time-comparison-style ${computation_time_comparison_style}"
        aggregate_args+=" --computation-time-comparison-components ${computation_time_comparison_components}"
    else
        aggregate_args+=" --no-save-computation-time-comparison"
    fi
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=viz_lrp_fp_${job_suffix}
#SBATCH --output=${log_dir}/output_${job_suffix}_%j.txt
#SBATCH --error=${log_dir}/error_${job_suffix}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --chdir=${repo_root}

echo "Running visualize_lrp_fp on node \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "mode=${mode}"
echo "input_file=${input_file:-N/A}, input_dir=${input_dir:-N/A}"
echo "output_dir=${output_dir:-default}, format=${format}, dpi=${dpi}"
if [[ "${mode}" == "dir" ]]; then
    echo "computation_time_comparison_style=${computation_time_comparison_style}"
    echo "computation_time_comparison_components=${computation_time_comparison_components}"
fi
echo "repo_root=${repo_root}"

export PYTHONUNBUFFERED=1

source ~/.bashrc
conda deactivate || true
conda activate taichi_env
which python

python ./scripts/visualize_lrp_fp.py \\
  ${input_arg} \\
  ${output_dir_args} \\
  --dpi "${dpi}" \\
  --format "${format}" \\
  --stream-density "${stream_density}" \\
  --path-linewidth "${path_linewidth}" \\
  ${per_file_args} \\
  ${aggregate_args}

echo "End Time: \$(date)"
EOF
