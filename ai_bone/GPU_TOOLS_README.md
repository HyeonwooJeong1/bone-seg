# 서버 GPU 공용 도구  (/data1/shared/gpu)

여유 GPU를 확인하고, 필요 시 사용률을 유지(유휴 회수 방지)하는 공용 스크립트.

## 1. 실시간 GPU 현황 보기  (sudo 불필요, 누구나)
```bash
python /data1/shared/gpu/gpu_monitor.py       # 3초마다 갱신
python /data1/shared/gpu/gpu_monitor.py 5     # 5초 간격
python /data1/shared/gpu/gpu_monitor.py 0     # 1회만
```
각 GPU의 util·메모리·온도·전력과 **실행 중인 프로세스**, 맨 아래 **여유 GPU 번호**를 보여줍니다.

## 2. 유휴 GPU에 부하 유지 (keepalive)
```bash
# 여유 GPU(예 5 6 7)에 부하 — 공유 사용이므로 이름에 본인 표시
sudo docker run -d --name gpu-keep-<이름> --gpus all -v /data1:/data1 \
  bone-nnunet:2.8.1 python /data1/shared/gpu/gpu_keepalive.py 5 6 7

# 상태만 확인
sudo docker run --rm --gpus all -v /data1:/data1 bone-nnunet:2.8.1 \
  python /data1/shared/gpu/gpu_keepalive.py

# 종료 (GPU 다시 쓰거나 필요 없을 때)
sudo docker stop gpu-keep-<이름> && sudo docker rm gpu-keep-<이름>
```
옵션: `-e KEEP_SLEEP_MS=50`(부하↓) · `-e KEEP_N=20000`(메모리↑).

## 주의
- keepalive는 하드웨어엔 무해하지만 **전력·연산 낭비**입니다. 실제 작업이 있으면 그것을 우선하고, **GPU가 필요해지거나 학습이 끝나면 즉시 종료**하세요.
- 여러 명이 쓰므로 `gpu_monitor.py`로 **먼저 여유 GPU를 확인**하고, 남의 학습 GPU는 건드리지 마세요.
- 컨테이너 이름은 `gpu-keep-<본인>` 처럼 구분해 충돌을 피하세요.
