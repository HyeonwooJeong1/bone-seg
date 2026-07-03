"""
download_vsd.py — VSDFullBodyBoneReconstruction 전체 subject 다운로드 (Zenodo 8302449)

각 subject zip을 ai_bone/data/vsd/ 에 받고, 이미 있으면 건너뜀.
"""
import sys, json, time, urllib.request, ssl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = 8  # Zenodo는 연결당 속도제한 → 병렬로 총 처리량 확보

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REC = "8302449"
DEST = Path("ai_bone/data/vsd")
DEST.mkdir(parents=True, exist_ok=True)
ctx = ssl.create_default_context()


def get_files():
    url = f"https://zenodo.org/api/records/{REC}"
    with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
        d = json.load(r)
    return [(f["key"], f["links"]["self"], f.get("size", 0)) for f in d.get("files", [])]


def download(key, link, size):
    out = DEST / key
    if out.exists() and out.stat().st_size >= size * 0.99:
        print(f"[skip] {key} (이미 있음)"); return
    t0 = time.time()
    print(f"[get ] {key} ({size/1e6:.0f} MB) ...", flush=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(link, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(out)
    print(f"[done] {key} ({(time.time()-t0):.0f}s)", flush=True)


def main():
    files = get_files()
    tot = sum(s for _, _, s in files)
    print(f"총 {len(files)}개 파일, {tot/1e9:.1f} GB — {WORKERS}개 병렬 다운로드", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download, k, l, s): k for k, l, s in files}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERR ] {futs[fut]}: {e}", flush=True)
    print(f"=== 다운로드 완료 ({time.time()-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
