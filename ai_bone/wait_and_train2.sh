#!/bin/bash
# wait_and_train2.sh — 전처리(nnUNetv2_preprocess) + SSD복사(cp) 완료를 감지하면 자동 학습 시작
LOG=/data1/bone/train_logs
mkdir -p $LOG
echo "대기 시작 $(date)" > $LOG/wait2.log
# 1) 전처리 프로세스 종료 대기
while pgrep -f "nnUNetv2_preprocess" >/dev/null; do sleep 20; done
echo "전처리 완료 $(date)" >> $LOG/wait2.log
# 2) SSD 복사(cp) 종료 대기
while pgrep -f "cp -r /data1.*nnunet_pre" >/dev/null; do sleep 5; done
echo "SSD복사 완료 $(date)" >> $LOG/wait2.log
sleep 10
# 3) SSD 데이터 존재 확인 후 학습
if [ -f /home/ubuntu/nnunet_pre/Dataset476_PelvisThighs/nnUNetPlans_iso06.json ] \
   && [ -f /home/ubuntu/nnunet_pre/Dataset481_ShanksFeet/nnUNetPlans_iso06.json ]; then
  echo "학습 시작 $(date)" >> $LOG/wait2.log
  bash /data1/bone/ai_bone/train_all.sh
else
  echo "SSD 데이터 미비 — 학습 미시작 $(date)" >> $LOG/wait2.log
fi
