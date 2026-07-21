"""
web_monitor.py — Dataset490 5-fold 학습 진행을 웹페이지로 제공 (휴대폰 브라우저용).

서버 호스트에서 실행 (백그라운드):
  nohup python3 /data1/shared/gpu/web_monitor.py 8080 mysecret \
    > /data1/shared/gpu/web.log 2>&1 &
휴대폰 브라우저:  http://<서버공인IP>:8080/mysecret   (토큰 생략 시 http://IP:8080/)
  * 클라우드 보안그룹/방화벽에서 해당 포트(예 8080)를 먼저 열어야 접속됩니다.
종료:  pkill -f web_monitor.py

인자: [port=8080] [token=없음]   (token을 주면 그 경로에서만 페이지 제공)
"""
import sys, os, glob, re, time, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

R = "/data1/bone/ai_bone/nnunet/results"
DS = "Dataset490_LowerLimb"
FOLDS = [0, 1, 2, 3, 4]
MAX_EPOCH = 1000
REFRESH = 15
TOKEN = ""


def find_log(fold):
    logs = glob.glob(f"{R}/{DS}/*NoMirroring_ES__*/fold_{fold}/training_log_*.txt")
    return sorted(logs, key=os.path.getmtime)[-1] if logs else None


def parse(path):
    ep = tl = vl = dice = etime = None
    best = None
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
                    best = v if best is None else max(best, v)
    return ep, tl, vl, dice, etime, best


def gpu_badges():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return ""
    spans = []
    for l in out.strip().splitlines():
        i, u, m = [x.strip() for x in l.split(",")]
        cls = "on" if (int(u) > 0 or int(m) > 500) else "off"
        spans.append(f"<span class='g {cls}'>GPU{i} {u}%</span>")
    return "".join(spans)


def eta_str(ep, etime):
    if not ep or not etime:
        return "-"
    sec = (MAX_EPOCH - ep) * etime
    return f"{int(sec // 3600)}h{int((sec % 3600) // 60):02d}m"


def build_rows():
    rows = ""
    for f in FOLDS:
        log = find_log(f)
        if not log:
            rows += f"<tr><td>f{f}</td><td colspan='4'>대기중</td></tr>"
            continue
        ep, tl, vl, dice, etime, best = parse(log)
        pct = int((ep or 0) / MAX_EPOCH * 100)
        dstr = f"{dice:.3f}" if dice is not None else "-"
        bstr = f"{best:.3f}" if best is not None else "-"
        bar = f"<div class='barbg'><div class='bar' style='width:{pct}%'></div></div>"
        rows += (f"<tr><td>f{f}</td>"
                 f"<td>{ep if ep is not None else '-'}/{MAX_EPOCH}{bar}</td>"
                 f"<td class='big'>{dstr}</td><td>{bstr}</td>"
                 f"<td>{eta_str(ep, etime)}</td></tr>")
    return rows


def page():
    css = ("body{font-family:sans-serif;margin:0;background:#0f2a4a;color:#fff}"
           "h1{font-size:17px;padding:12px;margin:0;background:#1f6fb2}"
           "table{width:100%;border-collapse:collapse;font-size:15px}"
           "td,th{padding:10px;border-bottom:1px solid #24405f;text-align:center}"
           ".big{font-size:20px;font-weight:bold;color:#9cdcfe}"
           ".barbg{background:#24405f;height:6px;border-radius:3px;margin-top:5px}"
           ".bar{background:#2e8b57;height:6px;border-radius:3px}"
           ".g{display:inline-block;padding:4px 8px;margin:4px;border-radius:6px;"
           "font-size:12px;background:#24405f}.g.on{background:#2e8b57}"
           ".foot{padding:12px;font-size:12px;color:#88a5c0}")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<meta http-equiv='refresh' content='{REFRESH}'>"
            f"<title>학습 모니터</title><style>{css}</style></head><body>"
            f"<h1>Dataset490 5-fold 학습 · {time.strftime('%Y-%m-%d %H:%M:%S')}</h1>"
            f"<table><tr><th>fold</th><th>epoch</th><th>dice</th><th>best</th><th>ETA</th></tr>"
            f"{build_rows()}</table>"
            f"<div class='foot'>{gpu_badges()}<br>{REFRESH}초마다 자동 갱신 · "
            f"ETA는 1000ep 기준 상한(early stopping으로 더 일찍 끝남)</div></body></html>")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        want = ("/" + TOKEN) if TOKEN else "/"
        path = self.path.split("?")[0].rstrip("/")
        if path != want.rstrip("/"):
            self.send_response(404); self.end_headers()
            self.wfile.write(b"not found"); return
        try:
            body = page().encode("utf-8")
        except Exception as e:
            body = f"error: {e}".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    global TOKEN
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    TOKEN = sys.argv[2] if len(sys.argv) > 2 else ""
    url = f"http://<서버IP>:{port}/{TOKEN}"
    print(f"web monitor on 0.0.0.0:{port}  →  {url}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
