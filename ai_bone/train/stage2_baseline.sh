#!/usr/bin/env bash
# Stage2: joint-pooling baseline. 사용법: bash stage2_baseline.sh <FOLD> <GPU_ID>
set -euo pipefail
FOLD="${1:?fold}"; GPU="${2:-0}"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 510 3d_fullres "$FOLD" \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL \
  -pretrained_weights /data1/bone/nnunet/results/Dataset500_AxialPretrain/*_all/checkpoint_final.pth --c
