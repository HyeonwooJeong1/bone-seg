"""
watch_train.py — Dataset490 5-fold 학습 실시간 모니터 (dice/loss/epoch/ETA + GPU).

표준 라이브러리만 사용 → 서버 호스트에서 바로 실행 (로그는 /data1, nvidia-smi는 호스트).
  python /data1/bone/ai_bone/watch_train.py         # 15초마다 자동 갱신
  python /data1/bone/ai_bone/watch_train.py 30      # 30초 간격
  python /data1/bone/ai_bone/watch_train.py 0       # 1회만 출력(루프 없음)
Ctrl+C 로 종료.
"""
import os, glob, re, subprocess, sys, time

R = "/data1/bone/ai_bone/nnunet/results"
DS = "Dataset490_LowerLimb"
FOLDS = [0, 1, 2, 3, 4]
MAX_EPOCH = 1000            # 상한 (early stopping으로 더 일찍 끝날 수 있음)
TRAIN_GPUS = [0, 1, 2, 3, 4]


def find_log(fold):
    logs = glob.glob(f"{R}/{DS}/*NoMirroring_ES__*/fold_{fold}/training_log_*.txt")
    return sorted(logs, key=os.path.getmtime)[-1] if logs else None


def parse(path):
    """로그에서 최신 epoch/loss/dice/epoch_time 및 best EMA dice 추출."""
    ep = tl = vl = dice = etime = None
    best_ema = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)", line)
            if m:
                ep = int(m.group(1))
            if "train_loss" in line:
                mm = re.search(r"train_loss\s+(-?[\d.]+)", line)
                if mm:
                    tl = float(mm.group(1))
            if "val_loss" in line:
                mm = re.search(r"val_loss\s+(-?[\d.]+)", line)
                if mm:
                    vl = float(mm.group(1))
            if "Pseudo dice" in line:
                nums = [float(x) for x in re.findall(r"[\d.]+(?=\))", line) if x not in ("", ".")]
                if nums:
                    dice = sum(nums) / len(nums)
            if "Epoch time" in line:
                mm = re.search(r"Epoch time:?\s+([\d.]+)", line)
                if mm:
                    etime = float(mm.group(1))
            if "EMA pseudo Dice" in line:
                mm = re.search(r"EMA pseudo Dice:?\s+([\d.]+)", line)
                if mm:
                    v = float(mm.group(1))
                    best_ema = v if best_ema is None else max(best_ema, v)
    return ep, tl, vl, dice, etime, best_ema


def fmt_eta(ep, etime):
    if not ep or not etime:
        return "-"
    sec = (MAX_EPOCH - ep) * etime
    h = int(sec // 3600); m = int((sec % 3600) // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def gpu_table():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True)
    except Exception:
        return ["(nvidia-smi 사용 불가)"]
    rows = []
    for l in out.strip().splitlines():
        idx, util, mem = [x.strip() for x in l.split(",")]
        used = int(re.sub(r"\D", "", mem.split()[0]) or 0)
        tag = "학습중" if used > 1000 else "여유"
        star = " *" if int(idx) in TRAIN_GPUS else "  "
        rows.append(f"  GPU{idx}{star} util {util:>5}   mem {mem:>10}   [{tag}]")
    return rows


def render():
    lines = []
    lines.append("=" * 78)
    lines.append(f" Dataset490 5-fold 학습 모니터   ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("=" * 78)
    lines.append(f" {'fold':<5}{'epoch':>10}{'dice(now)':>11}{'best(EMA)':>11}"
                 f"{'train':>9}{'val':>9}{'ep_t':>8}{'ETA':>8}")
    lines.append("-" * 78)
    done = 0
    for fold in FOLDS:
        log = find_log(fold)
        if not log:
            lines.append(f" f{fold:<4}{'대기중':>10}")
            continue
        ep, tl, vl, dice, etime, best = parse(log)
        if ep is not None and ep >= MAX_EPOCH - 1:
            done += 1
        eps = f"{ep}/{MAX_EPOCH}" if ep is not None else "-"
        dn = f"{dice:.3f}" if dice is not None else "-"
        db = f"{best:.3f}" if best is not None else "-"
        tls = f"{tl:.3f}" if tl is not None else "-"
        vls = f"{vl:.3f}" if vl is not None else "-"
        ets = f"{etime:.1f}s" if etime is not None else "-"
        lines.append(f" f{fold:<4}{eps:>10}{dn:>11}{db:>11}{tls:>9}{vls:>9}{ets:>8}"
                     f"{fmt_eta(ep, etime):>8}")
    lines.append("-" * 78)
    lines.extend(gpu_table())
    lines.append("=" * 78)
    lines.append(" * = 학습 배정 GPU(0-4) · dice(now)=최근 epoch · best=EMA 최고 · ETA=1000ep 기준 상한")
    return "\n".join(lines)


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    if interval <= 0:
        print(render()); return
    try:
        while True:
            os.system("clear")
            print(render(), flush=True)
            print(f"\n [{interval}초마다 갱신 · Ctrl+C 종료]")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n모니터 종료.")


if __name__ == "__main__":
    main()
