#!/bin/bash
# live.sh — 학습 진행(dice/loss/epoch) + GPU 사용률 실시간 대시보드
# 사용: ssh -t ... "bash /data1/bone/ai_bone/live.sh [갱신초=10]"  (Ctrl+C 종료)
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate pt210_py312
INT="${1:-10}"
while true; do
  clear
  echo "###### 학습 실시간 모니터  $(date '+%Y-%m-%d %H:%M:%S')  (${INT}s 갱신, Ctrl+C 종료) ######"
  python /data1/bone/ai_bone/monitor.py
  sleep "$INT"
done
