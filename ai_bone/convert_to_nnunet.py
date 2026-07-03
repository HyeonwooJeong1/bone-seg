"""
convert_to_nnunet.py — VSD seg.nrrd + CT → nnU-Net v2 raw dataset

부위별(476 Pelvis-Thighs / 481 Shanks-Feet) 통일 클래스로 재매핑하여
nnUNet_raw/Dataset{ID}_{name}/{imagesTr,labelsTr,dataset.json} 생성.

- GT는 Reconstruction.seg.nrrd 사용 (배경 클래스 없이 깔끔).
- 라벨은 seg.nrrd 메타데이터의 Segment 이름으로 식별 → 통일 클래스 ID로 remap
  (부위마다 원본 LabelValue가 다르므로 이름 기반이 안전).

실행:
  ct_env python ai_bone\\convert_to_nnunet.py 481
  ct_env python ai_bone\\convert_to_nnunet.py 476
"""
import sys, json, shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VSD = Path("ai_bone/data/vsd")
RAW = Path("ai_bone/nnunet/raw")

# 부위별 통일 클래스 (배경=0)
PART = {
    "476": {
        "name": "PelvisThighs",
        "dataset_id": 476,
        "part_tag": "Pelvis-Thighs",
        "labels": ["Femur_L", "Femur_R", "Hip_L", "Hip_R", "Sacrum",
                   "Patella_L", "Patella_R", "Tibia_L", "Tibia_R",
                   "Fibula_L", "Fibula_R"],
    },
    "481": {
        "name": "ShanksFeet",
        "dataset_id": 481,
        "part_tag": "Shanks-Feet",
        "labels": ["Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R",
                   "Talus_L", "Talus_R", "Calcaneus_L", "Calcaneus_R",
                   "Tarsals_L", "Tarsals_R", "Metatarsals_L", "Metatarsals_R",
                   "Phalanges_L", "Phalanges_R"],
    },
}


def read_seg_namemap(seg_img):
    """seg.nrrd 메타데이터 → {원본 label value(int): segment name}"""
    segs = {}
    for k in seg_img.GetMetaDataKeys():
        if k.endswith("_Name"):
            segs.setdefault(k.split("_")[0], {})["name"] = seg_img.GetMetaData(k)
        if k.endswith("_LabelValue"):
            segs.setdefault(k.split("_")[0], {})["lv"] = int(seg_img.GetMetaData(k))
    return {v["lv"]: v["name"] for v in segs.values() if "lv" in v and "name" in v}


def convert(part_key):
    cfg = PART[part_key]
    name2id = {n: i + 1 for i, n in enumerate(cfg["labels"])}  # 1..K
    tag = cfg["part_tag"]

    ds_dir = RAW / f"Dataset{cfg['dataset_id']:03d}_{cfg['name']}"
    img_dir = ds_dir / "imagesTr"
    lab_dir = ds_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p for p in VSD.iterdir() if p.is_dir()])
    n = 0
    for subj in subjects:
        # 부위 스캔 폴더 찾기 (…CT.<num>/) — Reconstruction seg 존재로 판별
        rec = list(subj.glob(f"*/*-{tag}_Reconstruction.seg.nrrd"))
        cts = list(subj.glob(f"*/*.nrrd"))
        if not rec:
            continue
        rec = rec[0]
        scan_folder = rec.parent
        # CT는 반드시 seg와 같은 부위(tag)여야 함. 그런데 파일명 규칙이 케이스마다 다름:
        #  - z-접두사: 한 폴더에 두 부위 CT, 파일명에 부위명 포함 (…-Pelvis-Thighs.nrrd)
        #  - 숫자(016 등): 부위별 폴더 분리, CT 파일명은 부위번호 (…476.nrrd, 부위명 없음)
        # → 부위명이 파일명에 있으면 그걸로, 없으면(폴더가 이미 부위별이라 CT 하나) 그 CT로.
        cands = [p for p in scan_folder.glob("*.nrrd") if not p.name.endswith("seg.nrrd")]
        tagged = [p for p in cands if tag in p.name]
        ct_list = tagged if tagged else cands
        if not ct_list:
            print(f"[skip] {subj.name}: CT 없음"); continue
        ct_path = ct_list[0]

        case = f"{cfg['name']}_{subj.name}"
        # CT
        ct = sitk.ReadImage(str(ct_path))
        sitk.WriteImage(ct, str(img_dir / f"{case}_0000.nii.gz"))
        # Label remap
        seg = sitk.ReadImage(str(rec))
        arr = sitk.GetArrayFromImage(seg)
        namemap = read_seg_namemap(seg)          # origval -> name
        out = np.zeros_like(arr, dtype=np.uint8)
        for origval, nm in namemap.items():
            if nm in name2id:
                out[arr == origval] = name2id[nm]
        out_img = sitk.GetImageFromArray(out)
        # CT와 seg의 정합 방식을 size로 분기:
        #  - 같은 size: 배열 인덱스가 이미 정렬됨(z-prefix는 seg 메타 direction/origin만 flip이고
        #    array는 CT와 정렬). CT geometry를 그대로 씌운다(resample 금지 — 물리좌표 resample하면
        #    flip 메타 때문에 라벨이 CT 밖으로 날아가 전멸함). z001 검증 겹침 97.5%.
        #  - 다른 size(예 016: CT 722 vs seg 808): 물리좌표 기준 nearest resample로 정합.
        if seg.GetSize() == ct.GetSize():
            out_img.CopyInformation(ct)
        else:
            out_img.CopyInformation(seg)
            out_img = sitk.Resample(out_img, ct, sitk.Transform(),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(out_img, str(lab_dir / f"{case}.nii.gz"))
        n += 1
        present = sorted({namemap[v] for v in np.unique(arr) if v in namemap})
        print(f"[ok] {case}  labels={len(present)}  {present}")

    # dataset.json
    labels = {"background": 0}
    labels.update(name2id)
    dj = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": n,
        "file_ending": ".nii.gz",
    }
    with open(ds_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dj, f, ensure_ascii=False, indent=2)
    print(f"\n=== {ds_dir.name}: {n} cases, {len(name2id)} classes ===")
    print(f"labels: {labels}")


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "481")
