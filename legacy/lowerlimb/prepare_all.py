"""
prepare_all.py — VSD zip 일괄 압축해제 + 476/481 nnU-Net 변환 (서버용)

다운로드 완료 후 실행:
  cd /data1/bone && conda activate pt210_py312 && python ai_bone/prepare_all.py

동작:
  1) ai_bone/data/vsd/*.zip 를 같은 폴더에 압축 해제 (subject 폴더 생성)
  2) convert_to_nnunet.py 의 convert()를 476, 481 각각 호출 → nnUNet_raw 생성
"""
import sys, zipfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VSD = Path("ai_bone/data/vsd")


def unzip_all():
    zips = sorted(VSD.glob("*.zip"))
    print(f"=== 압축 해제: {len(zips)}개 zip ===", flush=True)
    for z in zips:
        # 이미 풀렸는지: zip 이름(확장자 제외) 폴더 존재로 판별
        subj = z.stem
        if (VSD / subj).is_dir():
            print(f"[skip] {subj} (이미 해제됨)")
            continue
        t0 = time.time()
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(VSD)
            print(f"[unzip] {z.name} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[ERR] {z.name}: {e}", flush=True)


def main():
    unzip_all()
    print("\n=== nnU-Net 변환 ===", flush=True)
    import convert_to_nnunet as C
    for part in ("476", "481"):
        print(f"\n----- 부위 {part} -----", flush=True)
        C.convert(part)
    print("\n=== prepare_all 완료 ===")


if __name__ == "__main__":
    main()
