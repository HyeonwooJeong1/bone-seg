"""
gpu_monitor.py — 서버 GPU 실시간 대시보드 (공용).

각 GPU의 util/메모리/온도/전력 + 실행 중인 프로세스(pid, 이름, mem)를 표로 보여주고
자동 갱신한다. nvidia-smi만 사용 → sudo 불필요, 누구나 실행 가능.

  python /data1/shared/gpu/gpu_monitor.py        # 3초마다 갱신 (기본)
  python /data1/shared/gpu/gpu_monitor.py 5      # 5초 간격
  python /data1/shared/gpu/gpu_monitor.py 0      # 1회만 출력
Ctrl+C 로 종료.
"""
import sys, os, time, subprocess

FREE_MEM_MB = 500   # 이 이하면 '여유'로 표시


def _q(args):
    return subprocess.check_output(["nvidia-smi"] + args, text=True).strip()


def gpus():
    out = _q(["--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,"
              "temperature.gpu,power.draw", "--format=csv,noheader,nounits"])
    res = []
    for l in out.splitlines():
        p = [x.strip() for x in l.split(",")]
        res.append(dict(idx=int(p[0]), uuid=p[1], util=p[2], mu=p[3], mt=p[4],
                        temp=p[5], pw=p[6].split(".")[0]))
    return res


def apps():
    try:
        out = _q(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                  "--format=csv,noheader,nounits"])
    except Exception:
        return {}
    m = {}
    for l in out.splitlines():
        if not l.strip():
            continue
        p = [x.strip() for x in l.split(",")]
        if len(p) < 4:
            continue
        m.setdefault(p[0], []).append((p[1], p[2].split("/")[-1], p[3]))
    return m


def cmdline(pid):
    """호스트 /proc에서 프로세스 전체 명령어 (컨테이너 프로세스도 호스트 PID로 보임)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode(errors="replace")
    except Exception:
        return ""


def classify(procs):
    """GPU에서 도는 프로세스들을 보고 상태 판정: 학습 / 스트레스 / 기타 / 여유."""
    kinds = set()
    for p, _n, _m in procs:
        cl = cmdline(p).lower()
        if "gpu_keepalive" in cl:
            kinds.add("스트레스")
        elif "nnunetv2_train" in cl or "nnunet" in cl or "run_all.sh" in cl:
            kinds.add("학습")
        elif "predict" in cl or "infer" in cl:
            kinds.add("추론")
        elif cl.strip():
            kinds.add("기타")
    if not kinds:
        return "여유"
    # 우선순위: 학습 > 추론 > 스트레스 > 기타
    for k in ("학습", "추론", "스트레스", "기타"):
        if k in kinds:
            return k
    return "기타"


def render():
    g = gpus(); a = apps()
    L = []
    L.append("=" * 96)
    L.append(f" 서버 GPU 실시간 현황   ({time.strftime('%Y-%m-%d %H:%M:%S')})   H100 80GB x {len(g)}")
    L.append("=" * 96)
    L.append(f" {'GPU':>3} {'util':>6} {'mem(used/total)':>17} {'온도':>5} {'전력':>6} "
             f"{'상태':>8}  프로세스(pid)")
    L.append("-" * 96)
    free = []
    for x in g:
        procs = a.get(x["uuid"], [])
        used = int(x["mu"] or 0)
        state = classify(procs)
        if used <= FREE_MEM_MB and state == "여유":
            free.append(str(x["idx"]))
        pstr = ", ".join(f"{n}({p})" for p, n, mm in procs) if procs else "-"
        L.append(f" {x['idx']:>3} {x['util']+'%':>6} {x['mu']+'/'+x['mt']+'MB':>17} "
                 f"{x['temp']+'C':>5} {x['pw']+'W':>6} {state:>8}  {pstr[:34]}")
    L.append("=" * 96)
    L.append(" 상태:  학습=실제 모델 학습 · 추론=예측 · 스트레스=keepalive(회수방지용) · 기타 · 여유")
    L.append(f" 여유 GPU: {', '.join(free) if free else '없음'}"
             f"   ·  keepalive 실행/종료법은 /data1/shared/gpu/README.md")
    return "\n".join(L)


def main():
    itv = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if itv <= 0:
        print(render()); return
    try:
        while True:
            os.system("clear")
            print(render(), flush=True)
            print(f"\n [{itv}초 갱신 · Ctrl+C 종료]")
            time.sleep(itv)
    except KeyboardInterrupt:
        print("\n종료.")


if __name__ == "__main__":
    main()
