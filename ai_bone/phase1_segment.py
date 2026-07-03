"""
phase1_segment.py — Phase 1 개념검증: 우리 Mako CT를 TotalSegmentator로 뼈 분할

핵심: Mako 하지 프로토콜은 하나의 시리즈 안에 z축으로 떨어진 여러 블록
(예: 골반/무릎/발목)을 담고, 블록마다 슬라이스 간격이 다르다(0.625 vs 2.5mm).
전체를 균일 간격으로 뭉치면 기하가 왜곡돼 분할이 틀어지므로, 큰 z-gap 기준으로
블록을 나눠 각 블록을 올바른 간격의 NIfTI로 만들어 개별 분할한다.

흐름:
  1) DICOM 폴더에서 축상(axial) CT 시리즈 선별 (SCOUT/SAG/COR 제외)
  2) 큰 z-gap(>10mm) 기준으로 contiguous 블록 분할
  3) 블록별 HU + 올바른 LPS→RAS affine → NIfTI
  4) 각 블록에 TotalSegmentator total + appendicular_bones (ml=True)
  5) 뼈 클래스만 per-bone 라벨맵으로 병합 + 블록별 통계

실행 (ct_env, GPU 쓰려면 foreground):
  C:\\ProgramData\\anaconda3\\envs\\ct_env\\python.exe ai_bone\\phase1_segment.py <dicom_dir> <out_dir> [block_index]
"""

import os
import sys
import json
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pydicom
import nibabel as nib

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map


GAP_MM = 10.0        # 이 이상 z 점프는 블록 경계
MIN_SLICES = 16      # 블록 최소 슬라이스 수 (너무 얇으면 분할 불안정)

TOTAL_BONE_NAMES = set()
for _id, _name in class_map["total"].items():
    if (_name.startswith("vertebrae_") or _name.startswith("rib_")
            or _name in {"sacrum", "hip_left", "hip_right",
                         "femur_left", "femur_right",
                         "humerus_left", "humerus_right",
                         "scapula_left", "scapula_right",
                         "clavicula_left", "clavicula_right",
                         "sternum", "skull", "costal_cartilages"}):
        TOTAL_BONE_NAMES.add(_name)
APPENDICULAR_NAMES = set(class_map["appendicular_bones"].values())


def _zpos(ds):
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    return float(np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float)))


def select_axial_series(dicom_dir):
    """축상 CT 중 슬라이스가 가장 많은 시리즈의 (position 정렬된) Dataset 리스트."""
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
             if f.lower().endswith(".dcm")]
    groups = {}
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        itype = getattr(ds, "ImageType", [])
        if "LOCALIZER" in itype or "SCOUT" in itype:
            continue
        if not hasattr(ds, "ImageOrientationPatient") or not hasattr(ds, "ImagePositionPatient"):
            continue
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        if iop.size != 6:
            continue
        normal = np.cross(iop[:3], iop[3:])
        if abs(normal[2]) < 0.9:      # 축상만
            continue
        groups.setdefault(ds.SeriesInstanceUID, []).append(f)

    if not groups:
        raise RuntimeError("축상 CT 시리즈를 찾지 못했습니다.")
    uid = max(groups, key=lambda u: len(groups[u]))
    slices = [pydicom.dcmread(f) for f in groups[uid]]
    slices.sort(key=_zpos)

    # 중복 위치 제거
    dedup, seen = [], set()
    for s in slices:
        z = round(_zpos(s), 3)
        if z in seen:
            continue
        seen.add(z)
        dedup.append(s)
    desc = getattr(dedup[0], "SeriesDescription", "?")
    print(f"[select] UID …{uid[-8:]} desc='{desc}' slices={len(dedup)} "
          f"(중복 {len(slices)-len(dedup)} 제거)")
    return dedup


def split_blocks(slices):
    """큰 z-gap 기준으로 contiguous 블록 분할."""
    zs = np.array([_zpos(s) for s in slices])
    d = np.diff(zs)
    cut = np.where(d > GAP_MM)[0]
    starts = [0] + [c + 1 for c in cut]
    ends = [c + 1 for c in cut] + [len(slices)]
    blocks = []
    for s, e in zip(starts, ends):
        blk = slices[s:e]
        if len(blk) >= MIN_SLICES:
            blocks.append(blk)
    print(f"[split] {len(blocks)}개 블록:")
    for i, blk in enumerate(blocks):
        z0, z1 = _zpos(blk[0]), _zpos(blk[-1])
        step = (z1 - z0) / (len(blk) - 1) if len(blk) > 1 else 0
        print(f"  block {i}: {len(blk)}장, z {z0:.1f}..{z1:.1f} (span {z1-z0:.1f}mm, step {step:.3f}mm)")
    return blocks


def build_nifti(slices):
    img = np.stack([s.pixel_array for s in slices]).astype(np.int16)  # (nz,ny,nx)
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    if slope != 1:
        img = (slope * img.astype(np.float32)).astype(np.int16)
    img = (img.astype(np.int32) + int(intercept)).astype(np.int16)

    s0 = slices[0]
    iop = np.array(s0.ImageOrientationPatient, dtype=float)
    row_cos, col_cos = iop[:3], iop[3:]
    ps = [float(v) for v in s0.PixelSpacing]      # [dy(row), dx(col)]
    dy, dx = ps[0], ps[1]
    ipp0 = np.array(s0.ImagePositionPatient, dtype=float)
    ipp_last = np.array(slices[-1].ImagePositionPatient, dtype=float)
    nz = len(slices)
    slice_vec = (ipp_last - ipp0) / (nz - 1) if nz > 1 else np.cross(row_cos, col_cos)

    data = np.transpose(img, (2, 1, 0))            # (nx,ny,nz)
    aff = np.eye(4)
    aff[:3, 0] = row_cos * dx
    aff[:3, 1] = col_cos * dy
    aff[:3, 2] = slice_vec
    aff[:3, 3] = ipp0
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff    # LPS→RAS
    nii = nib.Nifti1Image(data, aff)
    nii.header.set_xyzt_units("mm")
    return nii


def segment_block(nii, out_dir, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(out_dir / "ct_hu.nii.gz"))

    segs = {}
    for task in ("total", "appendicular_bones"):
        t0 = time.time()
        print(f"  [{tag}] task='{task}' ...", flush=True)
        seg = totalsegmentator(nii, out_dir / f"seg_{task}", ml=True,
                               task=task, device="gpu", quiet=True)
        nib.save(seg, str(out_dir / f"seg_{task}_ml.nii.gz"))
        segs[task] = seg
        print(f"  [{tag}] task='{task}' done ({time.time()-t0:.0f}s)", flush=True)

    ref = segs["total"]
    combined = np.zeros(ref.shape, dtype=np.uint16)
    names, nid, found = {}, 1, []
    for task, allowed in (("total", TOTAL_BONE_NAMES),
                          ("appendicular_bones", APPENDICULAR_NAMES)):
        arr = np.asanyarray(segs[task].dataobj).astype(np.int32)
        for cls_id, name in class_map[task].items():
            if name not in allowed:
                continue
            m = arr == cls_id
            cnt = int(m.sum())
            if cnt == 0:
                continue
            combined[m] = nid
            names[nid] = name
            found.append((name, cnt))
            nid += 1
    nib.save(nib.Nifti1Image(combined, ref.affine), str(out_dir / "bone_labels.nii.gz"))
    nib.save(nib.Nifti1Image((combined > 0).astype(np.uint8), ref.affine),
             str(out_dir / "bone_binary.nii.gz"))
    with open(out_dir / "bone_labels.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in names.items()}, f, ensure_ascii=False, indent=2)
    return found


def run(dicom_dir, out_dir, only_block=None):
    slices = select_axial_series(dicom_dir)
    blocks = split_blocks(slices)
    out = Path(out_dir)
    for i, blk in enumerate(blocks):
        if only_block is not None and i != only_block:
            continue
        z0, z1 = _zpos(blk[0]), _zpos(blk[-1])
        tag = f"block{i}"
        print(f"\n===== {tag} (z {z0:.0f}..{z1:.0f}, {len(blk)}장) =====", flush=True)
        nii = build_nifti(blk)
        print(f"  nifti shape={nii.shape}", flush=True)
        found = segment_block(nii, out / tag, tag)
        print(f"  --- {tag} 발견 뼈 ---")
        for name, cnt in sorted(found, key=lambda x: -x[1]):
            print(f"     {name:22s} {cnt:>10,}")
    print(f"\n결과: {out.resolve()}")


if __name__ == "__main__":
    dicom_dir = sys.argv[1] if len(sys.argv) > 1 else "11423945/07049679"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/ai_bone/07049679"
    only_block = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run(dicom_dir, out_dir, only_block)
