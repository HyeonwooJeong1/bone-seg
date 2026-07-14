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
# Run as the host user so outputs on /data1 are not root-owned.
docker run --rm --user "$(id -u):$(id -g)" \
  -e nnUNet_raw=/data1/bone/nnunet/raw \
  -e nnUNet_preprocessed=/data1/bone/nnunet/preprocessed \
  -e nnUNet_results=/data1/bone/nnunet/results \
  -e nnUNet_compile=f \
  -v /data1:/data1 -w /data1 \
  "$IMAGE" "$@"
# NOTE: workdir is /data1 (NOT /data1/bone) so the host copy at /data1/bone/ai_bone
# does not shadow the image's baked-in code (/opt/ai_bone on PYTHONPATH).
