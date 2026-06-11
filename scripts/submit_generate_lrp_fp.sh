#!/bin/bash
set -euo pipefail

field_type="fbm"           # Options: expcov, fbm
field_size="100"
std=""
length_scale="0.2"         # Used for expcov
fbm_alpha="0.2"            # Used for fbm
fbm_c="0.027"                   # Optional fBm c override; leave empty to auto-derive from alpha/std
num_mc="1000"
seed_start="2026"
dh="1.0"

solver="pollock"              # Options: rk45, pollock
backend="cpu"              # Options: cpu, gpu
velocity_location="face"   # Options: cell, face
align_with_modflow="false"
density_particle="2"
num_particles="1000000"      # Leave empty to use density_particle
eps_speed="1e-12"
ds_min="0.05"
ds_max="0.5"
tol="1e-6"
y_policy="terminate"          # Options: clip, terminate
max_points="200000"
max_iters="200000"

fmm_backtracking="continuous"  # Options: discrete, continuous
fmm_continuous_step_size="0.5"

visualize="false"
visualize_output_dir=""        # Leave empty to use generate_lrp_fp.py default
visualize_dpi="300"
visualize_format="png"         # Options: png, pdf, svg
visualize_stream_density="1.0"
visualize_path_linewidth="2.0"
visualize_logk_paths="true"
visualize_head_velocity="true"
visualize_travel_time="true"
visualize_computation_time="true"

run_tag="generate_lrp_fp"
time="12:00:00"
cpus_per_task="8"
mem_per_cpu="2GB"
gpus_per_task="1"
cpu_workers=""              # Leave empty to auto-size the batch CPU pool.
gpu_device=""                # Leave empty to use the first visible GPU token.
failure_policy="retry"       # Options: retry, skip

account="fbarros_324"
partition="main" # Options: main, gpu

repo_root="/home1/binhaoli/LogFBM"
output_root_base="/project2/fbarros_324/binhaoli/task7_lrp_fp/Run3_computation_time"
output_root="${output_root_base}/outputs/${run_tag}"
flowsim_root="/home1/binhaoli/FlowSimulation"
fmm_core_root="/home1/binhaoli/FMM_FastestPath/Algorithm/src"
log_dir="${output_root}/slurm_logs"
mkdir -p "${log_dir}"

job_suffix="${field_type}_size${field_size}_mc${num_mc}_seedstart${seed_start}"
std_suffix="${std:-auto}"
if [[ "${field_type}" == "expcov" ]]; then
    param_suffix="ls${length_scale}_std${std_suffix}"
    job_suffix="${job_suffix}_${param_suffix}"
elif [[ "${field_type}" == "fbm" ]]; then
    param_suffix="alpha${fbm_alpha}_std${std_suffix}"
    job_suffix="${job_suffix}_${param_suffix}"
else
    echo "Unsupported field_type=${field_type}; expected expcov or fbm." >&2
    exit 1
fi

output_dir="${output_root}/${job_suffix}"
default_visualize_output_dir="${output_root}/${job_suffix}_figures"

length_scale_args=""
if [[ "${field_type}" == "expcov" ]]; then
    length_scale_args="--length-scale ${length_scale}"
fi

std_args=""
if [[ -n "${std}" ]]; then
    std_args="--std ${std}"
fi

fbm_args=""
if [[ "${field_type}" == "fbm" ]]; then
    fbm_args="--fbm-alpha ${fbm_alpha}"
    if [[ -n "${fbm_c}" ]]; then
        fbm_args+=" --fbm-c ${fbm_c}"
    fi
fi

num_particles_args=""
if [[ -n "${num_particles}" ]]; then
    num_particles_args="--num-particles ${num_particles}"
fi

align_args=""
if [[ "${align_with_modflow}" == "true" ]]; then
    align_args="--align-with-modflow"
fi

gpu_args=""
if [[ "${backend}" == "gpu" ]]; then
    gpu_args="#SBATCH --gpus-per-task=${gpus_per_task}"
fi

cpu_worker_args=""
if [[ -n "${cpu_workers}" ]]; then
    cpu_worker_args="--cpu-workers ${cpu_workers}"
fi

gpu_device_args=""
if [[ -n "${gpu_device}" ]]; then
    gpu_device_args="--gpu-device ${gpu_device}"
fi

visualize_args=""
visualize_dir_display="disabled"
if [[ "${visualize}" == "true" ]]; then
    visualize_args+=" --visualize"
    visualize_args+=" --visualize-dpi ${visualize_dpi}"
    visualize_args+=" --visualize-format ${visualize_format}"
    visualize_args+=" --visualize-stream-density ${visualize_stream_density}"
    visualize_args+=" --visualize-path-linewidth ${visualize_path_linewidth}"
    if [[ -n "${visualize_output_dir}" ]]; then
        visualize_args+=" --visualize-output-dir ${visualize_output_dir}"
        visualize_dir_display="${visualize_output_dir}"
    else
        visualize_dir_display="${default_visualize_output_dir}"
    fi
    if [[ "${visualize_logk_paths}" == "true" ]]; then
        visualize_args+=" --visualize-logk-paths"
    else
        visualize_args+=" --no-visualize-logk-paths"
    fi
    if [[ "${visualize_head_velocity}" == "true" ]]; then
        visualize_args+=" --visualize-head-velocity"
    else
        visualize_args+=" --no-visualize-head-velocity"
    fi
    if [[ "${visualize_travel_time}" == "true" ]]; then
        visualize_args+=" --visualize-travel-time"
    else
        visualize_args+=" --no-visualize-travel-time"
    fi
    if [[ "${visualize_computation_time}" == "true" ]]; then
        visualize_args+=" --visualize-computation-time"
    else
        visualize_args+=" --no-visualize-computation-time"
    fi
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=lrp_fp_${job_suffix}
#SBATCH --output=${log_dir}/output_${job_suffix}_%j.txt
#SBATCH --error=${log_dir}/error_${job_suffix}_%j.txt
#SBATCH --partition=${partition}
#SBATCH --time=${time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}
${gpu_args}
#SBATCH --chdir=${repo_root}

echo "Running generate_lrp_fp on node \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "field_type=${field_type}, field_size=${field_size}, std=${std_suffix}, num_mc=${num_mc}, seed_start=${seed_start}, dh=${dh}"
echo "solver=${solver}, backend=${backend}, velocity_location=${velocity_location}, align_with_modflow=${align_with_modflow}"
echo "density_particle=${density_particle}, num_particles=${num_particles:-auto}, y_policy=${y_policy}"
echo "fmm_backtracking=${fmm_backtracking}, fmm_continuous_step_size=${fmm_continuous_step_size}"
echo "repo_root=${repo_root}"
echo "output_root=${output_root}"
echo "FLOWSIM_PATH=${flowsim_root}"
echo "FMM_CORE_PATH=${fmm_core_root}"
echo "output_dir=${output_dir}"
echo "cpu_workers=${cpu_workers:-auto}, gpu_device=${gpu_device:-first_visible}"
echo "failure_policy=${failure_policy}"
echo "visualize=${visualize}, visualize_output_dir=${visualize_dir_display}, format=${visualize_format}, dpi=${visualize_dpi}"

mkdir -p "${output_dir}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLOWSIM_PATH="${flowsim_root}"
export FMM_CORE_PATH="${fmm_core_root}"

source ~/.bashrc
conda deactivate || true
conda activate taichi_env
which python

python ./scripts/generate_lrp_fp.py \\
  --field-type "${field_type}" \\
  --field-size "${field_size}" \\
  ${std_args} \\
  ${length_scale_args} \\
  ${fbm_args} \\
  --num-mc "${num_mc}" \\
  --seed-start "${seed_start}" \\
  --dh "${dh}" \\
  --solver "${solver}" \\
  --backend "${backend}" \\
  --velocity-location "${velocity_location}" \\
  ${align_args} \\
  --density-particle "${density_particle}" \\
  ${num_particles_args} \\
  --eps-speed "${eps_speed}" \\
  --ds-min "${ds_min}" \\
  --ds-max "${ds_max}" \\
  --tol "${tol}" \\
  --y-policy "${y_policy}" \\
  --max-points "${max_points}" \\
  --max-iters "${max_iters}" \\
  --fmm-backtracking "${fmm_backtracking}" \\
  --fmm-continuous-step-size "${fmm_continuous_step_size}" \\
  --output-dir "${output_dir}" \\
  ${cpu_worker_args} \\
  ${gpu_device_args} \\
  --failure-policy "${failure_policy}" \\
  ${visualize_args}

echo "End Time: \$(date)"
EOF
