#!/usr/bin/env bash
# Stage1: CADS 사전학습 (GPU 1장). 사용법: bash stage1_pretrain.sh <GPU_ID>
set -euo pipefail
GPU="${1:-0}"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/hyeonwoo/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/hyeonwoo/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 500 3d_fullres all \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL --c
