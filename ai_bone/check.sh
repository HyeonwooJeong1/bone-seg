#!/bin/bash
# check.sh — 학습 상태 한 번만 확인 (dice/loss/epoch + GPU)
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate pt210_py312
python /data1/bone/ai_bone/monitor.py
