"""
infer_mako.py — 학습된 nnU-Net으로 우리 Mako 환자 CT 추론

흐름:
  1) Mako DICOM → 축상 시리즈 선별 → z-gap 블록 분할 → 블록별 NIfTI
     (phase1_segment의 로직 재사용, TotalSegmentator 의존 없음)
  2) 각 블록 NIfTI를 통합 단일 모델(490)로 nnUNetv2_predict (5-fold 앙상블)
     (교수님 방법용 --save_probabilities 포함)
  3) 결과는 블록별 21라벨 라벨맵 pred_490/mako_block{n}.nii.gz (후처리는 다음 단계)

단일 모델이라 부위별 병합이 불필요 — 무릎/발목 경계도 한 모델이 직접 분할.

실행 (도커 컨테이너 안, nnUNet_* ENV 설정됨):
  python ai_bone/infer_mako.py <dicom_dir> <out_dir>
"""
import os, sys, subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pydicom
import nibabel as nib

GAP_MM = 10.0
MIN_SLICES = 16
RESULTS = "/data1/bone/ai_bone/nnunet/results"


def _zpos(ds):
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    n = np.cross(iop[:3], iop[3:])
    return float(np.dot(n, np.array(ds.ImagePositionPatient, dtype=float)))


def select_axial(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm")]
    groups = {}
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        it = getattr(ds, "ImageType", [])
        if "LOCALIZER" in it or "SCOUT" in it:
            continue
        if not hasattr(ds, "ImageOrientationPatient") or not hasattr(ds, "ImagePositionPatient"):
            continue
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        if iop.size != 6:
            continue
        if abs(np.cross(iop[:3], iop[3:])[2]) < 0.9:
            continue
        groups.setdefault(ds.SeriesInstanceUID, []).append(f)
    if not groups:
        raise RuntimeError("축상 CT 없음")
    uid = max(groups, key=lambda u: len(groups[u]))
    slices = [pydicom.dcmread(f) for f in groups[uid]]
    slices.sort(key=_zpos)
    dedup, seen = [], set()
    for s in slices:
        z = round(_zpos(s), 3)
        if z in seen:
            continue
        seen.add(z); dedup.append(s)
    print(f"[select] slices={len(dedup)}")
    return dedup


def split_blocks(slices):
    zs = np.array([_zpos(s) for s in slices])
    cut = np.where(np.diff(zs) > GAP_MM)[0]
    starts = [0] + [c + 1 for c in cut]
    ends = [c + 1 for c in cut] + [len(slices)]
    blocks = [slices[s:e] for s, e in zip(starts, ends) if e - s >= MIN_SLICES]
    for i, b in enumerate(blocks):
        print(f"  block{i}: {len(b)}장, z {_zpos(b[0]):.0f}..{_zpos(b[-1]):.0f}")
    return blocks


def build_nifti(slices):
    img = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    if slope != 1:
        img = (slope * img.astype(np.float32)).astype(np.int16)
    img = (img.astype(np.int32) + int(intercept)).astype(np.int16)
    s0 = slices[0]
    iop = np.array(s0.ImageOrientationPatient, dtype=float)
    row, col = iop[:3], iop[3:]
    ps = [float(v) for v in s0.PixelSpacing]
    dy, dx = ps[0], ps[1]
    ipp0 = np.array(s0.ImagePositionPatient, dtype=float)
    ipp1 = np.array(slices[-1].ImagePositionPatient, dtype=float)
    nz = len(slices)
    svec = (ipp1 - ipp0) / (nz - 1) if nz > 1 else np.cross(row, col)
    data = np.transpose(img, (2, 1, 0))
    aff = np.eye(4)
    aff[:3, 0] = row * dx; aff[:3, 1] = col * dy; aff[:3, 2] = svec; aff[:3, 3] = ipp0
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff
    nii = nib.Nifti1Image(data, aff); nii.header.set_xyzt_units("mm")
    return nii


def predict(in_dir, out_dir, dataset_id):
    # 컨테이너(Dockerfile ENV)가 이미 경로를 설정했으면 그걸 존중, 없으면 host 기본값.
    env = dict(os.environ)
    env.setdefault("nnUNet_raw", "/data1/bone/ai_bone/nnunet/raw")
    env.setdefault("nnUNet_preprocessed", "/home/ubuntu/nnunet_pre")
    env.setdefault("nnUNet_results", RESULTS)
    env["nnUNet_compile"] = "f"
    cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir,
           "-d", str(dataset_id), "-c", "3d_fullres",
           "-p", "nnUNetPlans_iso06", "-tr", "nnUNetTrainerNoMirroring_ES",
           "-f", "0", "1", "2", "3", "4", "--save_probabilities"]
    print(f"[predict] d{dataset_id}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def run(dicom_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    slices = select_axial(dicom_dir)
    blocks = split_blocks(slices)

    # 블록별 NIfTI를 단일 모델(490) 입력 폴더에 저장 후 5-fold 앙상블 추론
    in_dir = out / "in_490"; in_dir.mkdir(exist_ok=True)
    for i, blk in enumerate(blocks):
        nii = build_nifti(blk)
        nib.save(nii, str(in_dir / f"mako_block{i}_0000.nii.gz"))
    pred_dir = out / "pred_490"; pred_dir.mkdir(exist_ok=True)
    predict(str(in_dir), str(pred_dir), 490)
    print(f"=== d490 추론 완료 → {pred_dir}")
    print("infer_mako 완료")


if __name__ == "__main__":
    dicom = sys.argv[1] if len(sys.argv) > 1 else "/data1/bone/mako/07049679"
    outd = sys.argv[2] if len(sys.argv) > 2 else "/data1/bone/mako/pred_07049679"
    run(dicom, outd)
