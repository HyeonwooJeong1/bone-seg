#!/usr/bin/env bash
# Full-run job scheduler: keep N GPUs busy running an nnU-Net training queue.
# Each GPU runs its assigned jobs SEQUENTIALLY (one model per GPU at a time =
# that GPU fully dedicated), and all GPUs run in parallel, so wall-clock is
# about (total jobs / #GPUs) x per-job time. Every job resumes via `--c`.
#
# Usage:  bash run_queue.sh "0,1" jobs.txt
# jobs.txt: one job per line ->  DATASET_ID CONFIG FOLD TRAINER [PRETRAINED_CKPT]
#   e.g.    510 3d_fullres 0 nnUNetTrainerNoMirroring_ES_PL /data1/hyeonwoo/bone/nnunet/results/Dataset500_AxialPretrain/nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/fold_all/checkpoint_final.pth
# Lines starting with # are ignored.
set -uo pipefail
GPUS_CSV="${1:?comma-separated gpu ids, e.g. 0,1}"
JOBS="${2:?jobs file}"

source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/hyeonwoo/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/hyeonwoo/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
# Feed the H100 so it isn't starved by CPU augmentation; raise if util stays low.
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-24}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NG=${#GPUS[@]}
mkdir -p /data1/hyeonwoo/bone/train_logs

# Round-robin the job list into one queue file per GPU.
declare -a QF
for i in "${!GPUS[@]}"; do QF[$i]=$(mktemp); done
idx=0
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  echo "$line" >> "${QF[$((idx % NG))]}"
  idx=$((idx + 1))
done < "$JOBS"

run_queue_on_gpu() {   # $1=gpu id   $2=queue file
  local gpu="$1" qf="$2"
  while read -r did cfg fold tr pre; do
    [ -z "${did:-}" ] && continue
    local log="/data1/hyeonwoo/bone/train_logs/d${did}_${cfg}_f${fold}_gpu${gpu}.log"
    local args=("$did" "$cfg" "$fold" -p nnUNetPlans_iso06 -tr "$tr" --c)
    [ -n "${pre:-}" ] && args+=(-pretrained_weights "$pre")
    echo "[gpu$gpu] START d$did $cfg fold$fold ($tr) -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" nnUNetv2_train "${args[@]}" > "$log" 2>&1
    echo "[gpu$gpu] DONE  d$did $cfg fold$fold rc=$?"
  done < "$qf"
}

for i in "${!GPUS[@]}"; do
  run_queue_on_gpu "${GPUS[$i]}" "${QF[$i]}" &
done
wait
echo "all queued jobs finished"
