#!/bin/bash
# train_all.sh — 8×H100에 2부위(476,481) × 5-fold 분산 학습 (등방 0.6mm iso06)
# 10 job / 8 GPU: gpu0,1은 2 job 순차, 나머지 1 job.
# checkpoint 저장되므로 중단/재개 안전.

cd /data1/bone
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate pt210_py312
# conda의 libstdc++(CXXABI_1.3.15)를 시스템 것보다 우선 로드 — scipy import 충돌 방지
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# torch.compile 비활성화 — 8 job 동시 컴파일 CPU 경합으로 GPU가 놀던 문제 해결
export nnUNet_compile=f
# augmentation worker 증량 — job 6개로 줄여 job당 CPU 확보(124/6≈20), GPU 굶주림 방지
export nnUNet_n_proc_DA=18
export nnUNet_raw=/data1/bone/ai_bone/nnunet/raw
# 전처리 데이터를 로컬 SSD에서 로드 — Lustre NAS보다 빠르고 /dev/shm처럼 사라지지 않음
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/ai_bone/nnunet/results

LOG=/data1/bone/train_logs
mkdir -p $LOG
PLANS="-p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES"

run() {  # $1=gpu $2=dataset $3=fold
  echo "[start] GPU$1 Dataset$2 fold$3 $(date)" >> $LOG/schedule.log
  CUDA_VISIBLE_DEVICES=$1 nnUNetv2_train $2 3d_fullres $3 $PLANS \
      > $LOG/d$2_f$3_gpu$1.log 2>&1
  echo "[done ] GPU$1 Dataset$2 fold$3 $(date)" >> $LOG/schedule.log
}

# 균형 구성: 부위당 3 fold = 6 job, GPU 0-5 각 1개(CPU 여유). GPU 6,7은 여유.
( run 0 476 0 ) &
( run 1 476 1 ) &
( run 2 476 2 ) &
( run 3 481 0 ) &
( run 4 481 1 ) &
( run 5 481 2 ) &
wait
echo "=== 전체 학습 완료 $(date) ===" >> $LOG/schedule.log
