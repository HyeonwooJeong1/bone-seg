#!/bin/bash
# run_all.sh — 컨테이너 안에서 전처리(고속) + 5-fold 병렬 학습.
# 자원 최대 활용: CPU 124 / RAM 1.7TB / H100×8.
#   - 전처리 -np 60 (케이스=60, 코어당 1)
#   - 학습 5-fold를 GPU 0-4에 동시 (DA worker는 ENV nnUNet_n_proc_DA=24 → 24×5=120코어)
# checkpoint가 저장되므로 컨테이너 재시작/중단에도 재개 안전.
set -e
mkdir -p /data1/bone/train_logs

echo "[$(date)] 전처리 시작 (-np 60)"
nnUNetv2_preprocess -d 490 -plans_name nnUNetPlans_iso06 -c 3d_fullres -np 60
echo "[$(date)] 전처리 완료 → 5-fold 학습 시작"

for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$f nnUNetv2_train 490 3d_fullres $f \
      -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES \
      > /data1/bone/train_logs/d490_f$f.log 2>&1 &
  echo "  fold$f → GPU$f (PID $!)"
done
wait
echo "[$(date)] 5-fold 학습 완료"
