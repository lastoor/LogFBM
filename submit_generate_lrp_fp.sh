#!/bin/bash
set -euo pipefail

field_type="fbm"        # Options: expcov, fbm
field_size="100"
std="1.0"
length_scale="0.2"         # Used for expcov
fbm_alpha="3.0"            # Used for fbm
seed="42"
dh="1.0"

solver="pollock"              # Options: rk45, pollock
backend="gpu"              # Options: cpu, gpu
velocity_location="face"   # Options: cell, face
align_with_modflow="false"
density_particle="2"
num_particles="1000000"           # Leave empty to use density_particle
eps_speed="1e-12"
ds_min="0.05"
ds_max="0.5"
tol="1e-6"
y_policy="terminate"            # Options: clip, terminate
max_points="200000"
max_iters="200000"

fmm_backtracking="continuous"  # Options: discrete, continuous
fmm_continuous_step_size="0.5"

visualize="true"
visualize_output_dir=""        # Leave empty to use generate_lrp_fp.py default
visualize_dpi="300"
visualize_format="png"         # Options: png, pdf, svg
visualize_stream_density="1.0"
visualize_path_linewidth="2.0"
visualize_logk_paths="true"
visualize_head_velocity="true"
visualize_travel_time="true"

run_tag="generate_lrp_fp"
time="00:10:00"
cpus_per_task="2"
mem_per_cpu="2GB"
gpus_per_task="1"

account="fbarros_324"
partition="gpu"

repo_root="/home1/binhaoli/LRP_FP"
output_root="${repo_root}/outputs/${run_tag}"
log_dir="${output_root}/slurm_logs"
mkdir -p "${log_dir}"

job_suffix="${field_type}_size${field_size}_seed${seed}"
if [[ "${field_type}" == "expcov" ]]; then
    job_suffix="${job_suffix}_ls${length_scale}_std${std}"
elif [[ "${field_type}" == "fbm" ]]; then
    job_suffix="${job_suffix}_alpha${fbm_alpha}_std${std}"
else
    echo "Unsupported field_type=${field_type}; expected expcov or fbm." >&2
    exit 1
fi

output_file="${output_root}/${job_suffix}.npz"
default_visualize_output_dir="${output_root}/${job_suffix}_figures"

length_scale_args=""
if [[ "${field_type}" == "expcov" ]]; then
    length_scale_args="--length-scale ${length_scale}"
fi

fbm_args=""
if [[ "${field_type}" == "fbm" ]]; then
    fbm_args="--fbm-alpha ${fbm_alpha}"
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
echo "field_type=${field_type}, field_size=${field_size}, std=${std}, seed=${seed}, dh=${dh}"
echo "solver=${solver}, backend=${backend}, velocity_location=${velocity_location}, align_with_modflow=${align_with_modflow}"
echo "density_particle=${density_particle}, num_particles=${num_particles:-auto}, y_policy=${y_policy}"
echo "fmm_backtracking=${fmm_backtracking}, fmm_continuous_step_size=${fmm_continuous_step_size}"
echo "output_file=${output_file}"
echo "visualize=${visualize}, visualize_output_dir=${visualize_dir_display}, format=${visualize_format}, dpi=${visualize_dpi}"

mkdir -p "$(dirname "${output_file}")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda deactivate || true
conda activate taichi_env
which python

python ./generate_lrp_fp.py \\
  --field-type "${field_type}" \\
  --field-size "${field_size}" \\
  --std "${std}" \\
  ${length_scale_args} \\
  ${fbm_args} \\
  --seed "${seed}" \\
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
  --output "${output_file}" \\
  ${visualize_args}

echo "End Time: \$(date)"
EOF
