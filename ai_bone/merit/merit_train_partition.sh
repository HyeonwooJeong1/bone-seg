#!/usr/bin/env bash
# MERIT 파티션 fine-tune. 사용법: bash merit_train_partition.sh <PARTITION_DATASET_ID> <FOLD> <GPU_ID>
set -euo pipefail
DID="${1:?dataset id}"; FOLD="${2:?fold}"; GPU="${3:-0}"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/hyeonwoo/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/hyeonwoo/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train "$DID" 3d_fullres "$FOLD" \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerMERITFinetune \
  -pretrained_weights /data1/hyeonwoo/bone/nnunet/results/Dataset500_AxialPretrain/*_all/checkpoint_final.pth --c
