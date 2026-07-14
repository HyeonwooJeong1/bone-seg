#!/usr/bin/env bash
# Run ANY ai_bone command inside the bone-pipeline container (CPU data-prep/eval).
# Everything (combine, make_pairs, build_raw, download, eval, conflict analysis)
# runs in the container with /data1 mounted. Add GPU jobs via docker_train.sh.
#
# Usage:
#   bash run_in_docker.sh python -m ai_bone.datasets.combine --root ... --out ...
#   bash run_in_docker.sh python -m ai_bone.build_raw --pairs p.json --dataset totalseg --out ...
set -euo pipefail
IMAGE="${BONE_IMAGE:-bone-pipeline:latest}"
docker run --rm \
  -e nnUNet_raw=/data1/bone/nnunet/raw \
  -e nnUNet_preprocessed=/data1/bone/nnunet/preprocessed \
  -e nnUNet_results=/data1/bone/nnunet/results \
  -e nnUNet_compile=f \
  -v /data1:/data1 -w /data1/bone \
  "$IMAGE" "$@"
