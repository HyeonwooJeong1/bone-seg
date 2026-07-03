"""
verify_dataset.py — 재변환된 nnU-Net 데이터셋 다각도 검증

각 dataset(476/481)에 대해:
  1) 빈 라벨 케이스 (라벨 클래스 0개)
  2) CT-라벨 geometry 일치 (size/spacing/origin/direction)
  3) CT-라벨 정합 (라벨 voxel이 CT 뼈 HU>=THR와 겹치는 비율)
  4) 라벨 값 무결성 (음수/범위밖/비정수)
  5) 클래스별 등장 케이스 수 (분포, 희귀 클래스)

실행: ct_env python ai_bone/verify_dataset.py
"""
import os, json, glob
import numpy as np
import SimpleITK as sitk

RAW = "/data1/bone/ai_bone/nnunet/raw"
DATASETS = ["Dataset490_LowerLimb"]
THR = 200  # 뼈 HU 임계 (정합 확인용)


def verify(ds):
    ddir = os.path.join(RAW, ds)
    dj = json.load(open(os.path.join(ddir, "dataset.json")))
    id2name = {v: k for k, v in dj["labels"].items()}
    ncls = len(dj["labels"]) - 1
    labs = sorted(glob.glob(os.path.join(ddir, "labelsTr", "*.nii.gz")))

    print(f"\n{'='*70}\n{ds}  (케이스 {len(labs)}, 클래스 {ncls})\n{'='*70}")
    empty, geom_bad, bad_val, low_overlap = [], [], [], []
    class_count = {i: 0 for i in range(1, ncls + 1)}
    overlaps = []

    for lp in labs:
        case = os.path.basename(lp).replace(".nii.gz", "")
        ip = os.path.join(ddir, "imagesTr", case + "_0000.nii.gz")
        lab_img = sitk.ReadImage(lp)
        ct_img = sitk.ReadImage(ip)
        la = sitk.GetArrayFromImage(lab_img)
        ca = sitk.GetArrayFromImage(ct_img)

        # 1) 빈 라벨
        present = sorted(int(x) for x in np.unique(la) if x > 0)
        if not present:
            empty.append(case); continue
        for c in present:
            if c in class_count:
                class_count[c] += 1

        # 2) geometry 일치
        def close(a, b): return all(abs(x - y) < 1e-3 for x, y in zip(a, b))
        if not (lab_img.GetSize() == ct_img.GetSize()
                and close(lab_img.GetSpacing(), ct_img.GetSpacing())
                and close(lab_img.GetOrigin(), ct_img.GetOrigin())
                and close(lab_img.GetDirection(), ct_img.GetDirection())):
            geom_bad.append(case)

        # 3) 정합 (라벨 voxel 중 CT 뼈와 겹치는 비율)
        bone = ca >= THR
        lab_mask = la > 0
        ov = 100.0 * (bone & lab_mask).sum() / max(1, lab_mask.sum())
        overlaps.append((case, ov))
        if ov < 80:
            low_overlap.append((case, ov))

        # 4) 라벨 값 무결성
        mx = int(la.max())
        if mx > ncls or la.min() < 0:
            bad_val.append((case, int(la.min()), mx))

    # 리포트
    print(f"[1] 빈 라벨: {len(empty)}개 {empty if empty else '(없음 ✓)'}")
    print(f"[2] geometry 불일치: {len(geom_bad)}개 {geom_bad if geom_bad else '(없음 ✓)'}")
    print(f"[3] CT뼈 정합 낮음(<80%): {len(low_overlap)}개 {low_overlap if low_overlap else '(없음 ✓)'}")
    if overlaps:
        ovs = [o for _, o in overlaps]
        print(f"    겹침율 평균 {np.mean(ovs):.1f}% / 최소 {np.min(ovs):.1f}% / 최대 {np.max(ovs):.1f}%")
    print(f"[4] 라벨값 이상: {len(bad_val)}개 {bad_val if bad_val else '(없음 ✓)'}")
    print(f"[5] 클래스별 등장 케이스 수 (전체 {len(labs)-len(empty)}개 중):")
    for i in range(1, ncls + 1):
        nm = id2name.get(i, "?")
        c = class_count[i]
        flag = " ⚠️희귀" if c < 3 else ""
        print(f"      {i:2d} {nm:<16} {c:2d}개{flag}")


if __name__ == "__main__":
    for ds in DATASETS:
        verify(ds)
    print("\n검증 완료.")
