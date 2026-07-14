#!/usr/bin/env bash
# Run ONE nnU-Net training job INSIDE the bone-pipeline Docker image on a chosen GPU.
# GPU-containerized path (server requires GPU jobs in Docker).
#
# Usage: bash docker_train.sh <GPU_ID> <DATASET_ID> <CONFIG> <FOLD> <TRAINER> [PRETRAINED_CKPT]
#   e.g. bash docker_train.sh 2 510 3d_fullres 0 nnUNetTrainerNoMirroring_ES_PL \
#          /data1/bone/nnunet/results/Dataset500_AxialPretrain/nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/fold_all/checkpoint_final.pth
#
# Notes:
# - bone-pipeline:latest bakes in the ai_bone code AND the custom trainers, so no
#   bind-mounts are needed beyond /data1 (data + checkpoints).
# - CUDA_VISIBLE_DEVICES picks the GPU; --gpus all just exposes the devices.
# - Preprocessed cache is read from /data1 here (mounted). For the faster local
#   SSD you used before, add `-v /home/ubuntu/nnunet_pre:/nnpre` and set
#   nnUNet_preprocessed=/nnpre instead.
set -euo pipefail
GPU="${1:?gpu id}"; DID="${2:?dataset id}"; CFG="${3:?config}"; FOLD="${4:?fold}"
TR="${5:?trainer}"; PRE="${6:-}"

IMAGE=bone-pipeline:latest

ARGS=("$DID" "$CFG" "$FOLD" -p nnUNetPlans_iso06 -tr "$TR" --c)
[ -n "$PRE" ] && ARGS+=(-pretrained_weights "$PRE")

docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" \
  -e nnUNet_raw=/data1/bone/nnunet/raw \
  -e nnUNet_preprocessed=/data1/bone/nnunet/preprocessed \
  -e nnUNet_results=/data1/bone/nnunet/results \
  -e nnUNet_compile=f -e nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-24}" \
  -v /data1:/data1 \
  "$IMAGE" nnUNetv2_train "${ARGS[@]}"
