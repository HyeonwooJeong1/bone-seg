#!/bin/bash
# 학습 프로세스 완전 정리 — 스크립트 파일로 실행해 pgrep 자기참조 회피
pkill -9 -f "bash.*train_all" 2>/dev/null
sleep 1
pkill -9 -f "bin/nnUNetv2_train" 2>/dev/null
sleep 3
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
sleep 4
echo "정리 완료"
