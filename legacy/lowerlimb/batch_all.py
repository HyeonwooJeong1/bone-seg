"""
batch_all.py — 5명 환자 전체를 TS로 사전 분할(캐시 생성) + QC 오버레이

각 환자 폴더 → 축상 시리즈 → gap 블록분할 → 블록별 TS(total+appendicular)
→ per-bone 라벨맵 저장 → 블록별 QC PNG.

실행 (GPU, background 권장):
  ct_env python ai_bone\\batch_all.py
"""
import sys, time, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_segment import run
from qc_overlay import overlay

BASE = Path("11423945")
OUT = Path("results/ai_bone")
PATIENTS = ["07049679", "10603640", "10624783", "10635809", "11423945"]


def main():
    t_all = time.time()
    for pid in PATIENTS:
        dicom_dir = BASE / pid
        out_dir = OUT / pid
        if not dicom_dir.is_dir():
            print(f"[skip] {dicom_dir} 없음"); continue
        print(f"\n############ 환자 {pid} ############", flush=True)
        t0 = time.time()
        try:
            run(str(dicom_dir), str(out_dir))
        except Exception:
            print(f"[ERROR] {pid} 분할 실패:\n{traceback.format_exc()}", flush=True)
            continue
        # QC
        for blk in sorted(out_dir.glob("block*")):
            try:
                overlay(str(blk))
            except Exception as e:
                print(f"[QC 실패] {blk}: {e}", flush=True)
        print(f"############ {pid} 완료 ({time.time()-t0:.0f}s) ############", flush=True)
    print(f"\n===== 전체 완료 ({time.time()-t_all:.0f}s) =====")


if __name__ == "__main__":
    main()
