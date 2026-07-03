#!/bin/bash
# wait_and_train.sh — 전처리(nnUNetv2_preprocess) 완료를 감지하면 자동으로 학습 시작
LOG=/data1/bone/train_logs
mkdir -p $LOG
echo "전처리 완료 대기 시작 $(date)" > $LOG/wait.log
while pgrep -f "nnUNetv2_preprocess" >/dev/null; do
  sleep 30
done
echo "전처리 완료 감지 $(date) → 학습 시작" >> $LOG/wait.log
sleep 10
bash /data1/bone/ai_bone/train_all.sh
