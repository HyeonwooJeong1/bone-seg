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


def render():
    g = gpus(); a = apps()
    L = []
    L.append("=" * 90)
    L.append(f" 서버 GPU 실시간 현황   ({time.strftime('%Y-%m-%d %H:%M:%S')})   H100 80GB x {len(g)}")
    L.append("=" * 90)
    L.append(f" {'GPU':>3} {'util':>6} {'mem(used/total)':>17} {'온도':>5} {'전력':>6}  프로세스(pid,mem)")
    L.append("-" * 90)
    free = []
    for x in g:
        procs = a.get(x["uuid"], [])
        used = int(x["mu"] or 0)
        if used <= FREE_MEM_MB:
            free.append(str(x["idx"]))
        pstr = ", ".join(f"{n}({p},{mm}MB)" for p, n, mm in procs) if procs else "-"
        tag = "  ← 여유" if used <= FREE_MEM_MB else ""
        L.append(f" {x['idx']:>3} {x['util']+'%':>6} {x['mu']+'/'+x['mt']+'MB':>17} "
                 f"{x['temp']+'C':>5} {x['pw']+'W':>6}  {pstr[:40]}{tag}")
    L.append("=" * 90)
    L.append(f" 여유 GPU: {', '.join(free) if free else '없음'}"
             f"     keepalive:  sudo docker run -d --name gpu-keep-<이름> --gpus all \\")
    L.append("             -v /data1:/data1 bone-nnunet:2.8.1 "
             "python /data1/shared/gpu/gpu_keepalive.py <GPU번호>")
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
