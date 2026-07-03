"""
gpu_keepalive.py — 선택한 GPU에 지속 부하를 주어 사용률 유지(유휴 회수 방지).

먼저 현재 GPU 상태(어떤 GPU가 노는지)를 보고, 원하는 GPU 번호를 골라 실행한다.
- 인자 없이 실행 → 현재 GPU 상태만 출력 (선택 참고용)
- 인자로 GPU 번호 지정 → 그 GPU들에만 부하

컨테이너로 실행 (모든 GPU를 노출 → 스크립트에서 번호로 선택. -v /data1 필수):
  # 1) 현재 상태만 확인 (여유 GPU 파악)
  sudo docker run --rm --gpus all -v /data1:/data1 bone-nnunet:2.8.1 \
    python /data1/shared/gpu/gpu_keepalive.py
  # 2) 예: GPU 5,6,7 에 부하 (백그라운드). 공유 사용이므로 이름에 본인 표시.
  sudo docker run -d --name gpu-keep-<이름> --gpus all -v /data1:/data1 \
    bone-nnunet:2.8.1 python /data1/shared/gpu/gpu_keepalive.py 5 6 7

인자   : 대상 GPU 번호 (예:  5 6 7   또는   5,6,7)
환경변수: KEEP_N(행렬크기, 기본 16384) · KEEP_SLEEP_MS(부하 낮추기, 기본 0)
종료   : sudo docker stop gpu-keep-<이름>   (또는 Ctrl+C)
"""
import os
# torch 인덱스를 nvidia-smi 번호와 일치시킴 (반드시 torch import 전에)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys, time, threading, subprocess


def gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], text=True)
    except Exception as e:
        print("nvidia-smi 실패:", e)
        return
    print("현재 GPU 상태:")
    print(f"  {'GPU':>3} {'util':>7} {'mem_used':>11} {'mem_total':>11}  상태")
    for l in out.strip().splitlines():
        idx, util, used, total = [x.strip() for x in l.split(",")]
        u = int("".join(c for c in used if c.isdigit()) or 0)
        tag = "사용중" if u > 1000 else "여유 ←"
        print(f"  {idx:>3} {util:>7} {used:>11} {total:>11}  {tag}")


def parse_targets(args):
    toks = []
    for a in args:
        toks += [t for t in a.replace(",", " ").split() if t]
    return [int(t) for t in toks]


def burn(dev, N, sleep):
    import torch
    torch.cuda.set_device(dev)
    a = torch.randn(N, N, device=f"cuda:{dev}")
    b = torch.randn(N, N, device=f"cuda:{dev}")
    while True:
        _ = a @ b                       # 대형 행렬곱 반복 (결과 버림 → 값 발산 없음)
        torch.cuda.synchronize(dev)
        if sleep:
            time.sleep(sleep)


def main():
    gpu_status()
    targets = parse_targets(sys.argv[1:])
    if not targets:
        print("\n사용법:  python gpu_keepalive.py <GPU번호...>")
        print("예)      python gpu_keepalive.py 5 6 7")
        print("(대상 GPU를 지정하지 않아 상태만 출력했습니다. 여유 GPU 번호를 골라 다시 실행하세요.)")
        return
    import torch
    N = int(os.environ.get("KEEP_N", "16384"))
    sleep = float(os.environ.get("KEEP_SLEEP_MS", "0")) / 1000.0
    print(f"\n[keepalive] 대상 GPU {targets} · 행렬 {N}x{N} · sleep {sleep*1000:.0f}ms → 시작")
    for d in targets:
        threading.Thread(target=burn, args=(d, N, sleep), daemon=True).start()
    print("[keepalive] 실행 중... (docker stop gpu-keepalive 로 종료)")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[keepalive] 종료")


if __name__ == "__main__":
    main()
