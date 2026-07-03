"""
monitor.py — 학습 중인 모든 fold의 진행 상황 요약 (dice/loss/epoch)

서버에서 실행: python /data1/bone/ai_bone/monitor.py
각 fold의 현재 epoch, 최근 mean pseudo dice, best dice, train/val loss, GPU 상태 표시.
"""
import os, glob, re, subprocess

R = "/data1/bone/ai_bone/nnunet/results"
DATASETS = ["Dataset476_PelvisThighs", "Dataset481_ShanksFeet"]
FOLDS = [0, 1, 2]


def parse_log(path):
    ep, tl, vl, dice_mean, best = None, None, None, None, None
    dices = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r": Epoch (\d+)", line)
            if m:
                ep = int(m.group(1))
            if "train_loss" in line:
                tl = line.split("train_loss")[-1].strip()
            if "val_loss" in line:
                vl = line.split("val_loss")[-1].strip()
            if "Pseudo dice" in line:
                nums = re.findall(r"[\d.]+(?=\))", line)
                vals = [float(x) for x in nums if x not in ("", ".")]
                vals = [v for v in vals if v == v]  # nan 제외 안됨(문자열) → 이미 숫자만
                if vals:
                    dice_mean = sum(vals) / len(vals)
                    dices.append(dice_mean)
            if "EMA" in line and "best" in line.lower():
                mm = re.search(r"[\d.]+", line.split("EMA")[-1])
    best = max(dices) if dices else None
    return ep, tl, vl, dice_mean, best


def main():
    print("=" * 72)
    print(f"{'fold':<22}{'epoch':>6}{'dice(now)':>11}{'dice(best)':>12}{'val_loss':>10}")
    print("-" * 72)
    for ds in DATASETS:
        for fold in FOLDS:
            # ES trainer 폴더만 (이전 non-ES 로그 배제), 그중 가장 최근 수정 파일
            logs = glob.glob(f"{R}/{ds}/*NoMirroring_ES__*/fold_{fold}/training_log_*.txt")
            if not logs:
                print(f"{ds[7:]+' f'+str(fold):<22}{'대기중':>6}")
                continue
            log = sorted(logs, key=os.path.getmtime)[-1]
            ep, tl, vl, dnow, dbest = parse_log(log)
            name = ds.replace("Dataset4", "4")[:18] + f" f{fold}"
            dn = f"{dnow:.3f}" if dnow is not None else "-"
            db = f"{dbest:.3f}" if dbest is not None else "-"
            vll = vl if vl else "-"
            print(f"{name:<22}{ep if ep is not None else '-':>6}{dn:>11}{db:>12}{vll:>10}")
    print("=" * 72)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True)
        for l in out.strip().splitlines():
            idx, util, mem = [x.strip() for x in l.split(",")]
            used = int(mem.split()[0])
            tag = "학습중" if used > 1000 else "여유"
            print(f"  GPU{idx}: util {util:>5}  mem {mem:>10}  [{tag}]")
    except Exception:
        pass


if __name__ == "__main__":
    main()
