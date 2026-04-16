#!/bin/bash
# Submit a TIP-FBM scaling study (or single run) to SLURM.
# Mirrors the style of submit_generate_lrp_fp.sh.
#
# Usage:
#   bash scripts/submit_tip_fbm.sh
set -euo pipefail

# ---- Simulation parameters -----------------------------------------------
H="0.25"                  # Hurst exponent (0 < H < 1)
mode="site"              # site (imbibition) | bond (drainage)
L_values="128 256 512 1024"  # Grid sizes for scaling study (space-separated)
n_real="50"              # Realizations per L
seed="42"                # Base random seed
minpadding="1"           # Circulant-embedding padding

# Scale mode: choose ONE of the two options below, leave the other empty.
#   std_target : fix marginal std across L values (c computed per-L via compute_c)
#   fix_c      : fix c directly (same c for all L; effective std varies with L)
std_target=""         # used when fix_c is empty
fix_c="1"                 # e.g. "0.5"; if set, overrides std_target

single_run="false"       # "true" for a quick smoke test (first L, one realization)
visualize="false"        # "true" to save figures (only meaningful with single_run)

# ---- Parallelism ----------------------------------------------------------
# Realizations are independent and run in parallel across n_workers processes.
# cpus_per_task is set to match n_workers automatically.
n_workers="8"

# ---- SLURM settings -------------------------------------------------------
account="fbarros_324"
partition="main"
time="04:00:00"          # Increase for large L or many realizations
mem_per_cpu="4GB"

# ---- Paths ----------------------------------------------------------------
repo_root="/home1/binhaoli/LogFBM"
run_tag="tip_fbm"
output_root="${repo_root}/outputs/${run_tag}"
log_dir="${output_root}/slurm_logs"
mkdir -p "${log_dir}"

# ---- Validate scale mode --------------------------------------------------
if [[ -n "${fix_c}" && -n "${std_target}" ]]; then
    echo "ERROR: set either fix_c or std_target, not both." >&2
    exit 1
fi
if [[ -z "${fix_c}" && -z "${std_target}" ]]; then
    echo "ERROR: set at least one of fix_c or std_target." >&2
    exit 1
fi

# ---- Build job name -------------------------------------------------------
L_tag=$(echo "${L_values}" | tr ' ' '-')
if [[ -n "${fix_c}" ]]; then
    scale_tag="fixc${fix_c}"
else
    scale_tag="std${std_target}"
fi

job_suffix="H${H}_mode${mode}_L${L_tag}_nreal${n_real}_${scale_tag}_seed${seed}"
if [[ "${single_run}" == "true" ]]; then
    first_L=$(echo "${L_values}" | awk '{print $1}')
    job_suffix="single_H${H}_mode${mode}_L${first_L}_${scale_tag}_seed${seed}"
fi

output_root_full="${output_root}/H${H}_mode${mode}_${scale_tag}"

# ---- Build optional CLI arguments -----------------------------------------
scale_args=""
if [[ -n "${fix_c}" ]]; then
    scale_args="--fix-c ${fix_c}"
else
    scale_args="--std-target ${std_target}"
fi

single_run_args=""
if [[ "${single_run}" == "true" ]]; then
    single_run_args="--single-run"
fi

visualize_args=""
if [[ "${visualize}" == "true" ]]; then
    visualize_args="--visualize"
fi

# For --single-run pass only the first L; otherwise pass the full list.
if [[ "${single_run}" == "true" ]]; then
    L_cli=$(echo "${L_values}" | awk '{print $1}')
else
    L_cli="${L_values}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=tip_fbm_${job_suffix}
#SBATCH --output=${log_dir}/output_${job_suffix}_%j.txt
#SBATCH --error=${log_dir}/error_${job_suffix}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${n_workers}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH --chdir=${repo_root}

echo "=== TIP-FBM Simulation ==="
echo "Node:        \$SLURM_NODELIST"
echo "Start time:  \$(date)"
echo "H=${H}  mode=${mode}  L=[${L_values}]"
echo "n_real=${n_real}  seed=${seed}  scale=${scale_tag}  n_workers=${n_workers}"
echo "single_run=${single_run}  visualize=${visualize}"
echo "Output dir:  ${output_root_full}"
echo ""

mkdir -p "${output_root_full}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda deactivate || true
conda activate taichi_env
which python

python scripts/tip_fbm.py \\
  --L ${L_cli} \\
  --H "${H}" \\
  --mode "${mode}" \\
  --n-real "${n_real}" \\
  --seed "${seed}" \\
  --minpadding "${minpadding}" \\
  --n-workers "${n_workers}" \\
  --output-dir "${output_root_full}" \\
  ${scale_args} \\
  ${single_run_args} \\
  ${visualize_args}

echo ""
echo "End time: \$(date)"
EOF

echo "Submitted: tip_fbm_${job_suffix}"
echo "Logs:      ${log_dir}/"
echo "Output:    ${output_root_full}/"
