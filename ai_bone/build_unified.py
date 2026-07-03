"""
build_unified.py — VSD 26명 × 2스캔(Pelvis-Thighs, Shanks-Feet)을
하나의 통일 21라벨 nnU-Net 데이터셋(Dataset490_LowerLimb)으로 변환.

각 스캔을 개별 case로 만든다: LL_{subj}_PT, LL_{subj}_SF.
라벨은 seg.nrrd segment 이름 → 통일 21라벨 id로 remap(부위 무관, 이름 기반).

실행:
  ct_env python ai_bone/build_unified.py
"""
import sys, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VSD = Path("ai_bone/data/vsd")
RAW = Path("ai_bone/nnunet/raw")

DATASET_ID = 490
DATASET_NAME = "LowerLimb"

# 통일 21라벨 (배경=0) — Global Constraints와 반드시 일치
UNI = {
    "Femur_L": 1, "Femur_R": 2, "Hip_L": 3, "Hip_R": 4, "Sacrum": 5,
    "Patella_L": 6, "Patella_R": 7, "Tibia_L": 8, "Tibia_R": 9,
    "Fibula_L": 10, "Fibula_R": 11, "Talus_L": 12, "Talus_R": 13,
    "Calcaneus_L": 14, "Calcaneus_R": 15, "Tarsals_L": 16, "Tarsals_R": 17,
    "Metatarsals_L": 18, "Metatarsals_R": 19, "Phalanges_L": 20, "Phalanges_R": 21,
}
# 부위 태그 → case suffix
PARTS = [("Pelvis-Thighs", "PT"), ("Shanks-Feet", "SF")]


def read_seg_namemap(seg_img):
    """seg.nrrd 메타 → {원본 label value(int): segment name}"""
    segs = {}
    for k in seg_img.GetMetaDataKeys():
        if k.endswith("_Name"):
            segs.setdefault(k.split("_")[0], {})["name"] = seg_img.GetMetaData(k)
        if k.endswith("_LabelValue"):
            segs.setdefault(k.split("_")[0], {})["lv"] = int(seg_img.GetMetaData(k))
    return {v["lv"]: v["name"] for v in segs.values() if "lv" in v and "name" in v}


def convert_scan(rec, tag, case, img_dir, lab_dir):
    """단일 스캔(rec: Reconstruction.seg.nrrd)을 통일 라벨 case로 변환. 성공 시 present 라벨 반환."""
    scan_folder = rec.parent
    # CT 선택: 부위명이 파일명에 있으면 그걸로, 없으면(폴더가 부위별이라 CT 하나) 그 CT로.
    cands = [p for p in scan_folder.glob("*.nrrd") if not p.name.endswith("seg.nrrd")]
    tagged = [p for p in cands if tag in p.name]
    ct_list = tagged if tagged else cands
    if not ct_list:
        print(f"[skip] {case}: CT 없음"); return None
    ct = sitk.ReadImage(str(ct_list[0]))
    sitk.WriteImage(ct, str(img_dir / f"{case}_0000.nii.gz"))

    seg = sitk.ReadImage(str(rec))
    arr = sitk.GetArrayFromImage(seg)
    namemap = read_seg_namemap(seg)
    out = np.zeros_like(arr, dtype=np.uint8)
    for origval, nm in namemap.items():
        if nm in UNI:
            out[arr == origval] = UNI[nm]
    out_img = sitk.GetImageFromArray(out)
    if seg.GetSize() == ct.GetSize():
        out_img.CopyInformation(ct)
    else:
        out_img.CopyInformation(seg)
        out_img = sitk.Resample(out_img, ct, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sitk.WriteImage(out_img, str(lab_dir / f"{case}.nii.gz"))
    present = sorted({namemap[v] for v in np.unique(arr) if v in namemap})
    print(f"[ok] {case}  labels={len(present)}  {present}")
    return present


def main():
    ds_dir = RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    img_dir = ds_dir / "imagesTr"; lab_dir = ds_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p for p in VSD.iterdir() if p.is_dir()])
    n = 0
    for subj in subjects:
        for tag, suffix in PARTS:
            recs = list(subj.glob(f"*/*-{tag}_Reconstruction.seg.nrrd"))
            if not recs:
                continue
            case = f"LL_{subj.name}_{suffix}"
            if convert_scan(recs[0], tag, case, img_dir, lab_dir) is not None:
                n += 1

    labels = {"background": 0}
    labels.update(UNI)
    dj = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": n,
        "file_ending": ".nii.gz",
    }
    with open(ds_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dj, f, ensure_ascii=False, indent=2)
    print(f"\n=== {ds_dir.name}: {n} cases, {len(UNI)} classes ===")


if __name__ == "__main__":
    main()
