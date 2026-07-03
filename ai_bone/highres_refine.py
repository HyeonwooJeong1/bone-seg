"""
highres_refine.py — TS 라벨(1.5mm) + 원본 CT(0.625mm) 하이브리드 고해상 정밀화

TS 마스크는 '어느 뼈인가'만 신뢰하고, 실제 뼈 경계는 원본 CT 고해상 정보로
재획정한다. 결과: TS의 per-bone 라벨 + 원본 해상도 정밀 경계.

알고리즘:
  1) highres_bone = 원본 HU >= THRESHOLD  (0.625mm 정밀 뼈 마스크)
  2) TS 라벨을 highres_bone 안으로 nearest-label 전파(EDT 기반)
     → 각 고해상 뼈 voxel이 가장 가까운 TS 뼈 라벨을 받음
  3) 결과 라벨맵 저장 + 3D 표면 비교 PNG

실행: ct_env python ai_bone\\highres_refine.py <block_out_dir> [threshold=200]
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_closing, label as cc_label
import pyvista as pv

pv.OFF_SCREEN = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def refine(block_dir, threshold=200):
    d = Path(block_dir)
    ct_img = nib.load(str(d / "ct_hu.nii.gz"))
    ct = np.asanyarray(ct_img.dataobj).astype(np.int16)
    lab_img = nib.load(str(d / "bone_labels.nii.gz"))
    lab = np.asanyarray(lab_img.dataobj).astype(np.uint16)
    names = json.loads((d / "bone_labels.json").read_text(encoding="utf-8"))
    sp = np.abs(np.diag(lab_img.affine))[:3]

    # 1) 원본 고해상 뼈 마스크
    highres_bone = ct >= threshold
    # 작은 구멍 메움(피질골 내부 등) — 얇게만
    highres_bone = binary_closing(highres_bone, iterations=1)

    # 2) TS 라벨을 고해상 뼈 안으로 nearest-label 전파
    #    distance_transform_edt(inverse mask) 의 indices로 최근접 라벨 좌표 획득
    ts_fg = lab > 0
    if not ts_fg.any():
        print(f"[{d.name}] TS 라벨 없음"); return
    # 각 voxel에서 가장 가까운 TS-전경 voxel의 인덱스
    _, inds = distance_transform_edt(~ts_fg, return_indices=True)
    nearest_label = lab[tuple(inds)]              # 전 격자에 최근접 TS 라벨
    refined = np.where(highres_bone, nearest_label, 0).astype(np.uint16)

    # TS가 커버하던 영역에서 너무 먼 곳(다른 뼈/노이즈)이 붙는 것 방지:
    # TS 전경으로부터의 거리 > MARGIN 이면 버림
    dist_to_ts = distance_transform_edt(~ts_fg, sampling=sp)
    margin_mm = 8.0
    refined[dist_to_ts > margin_mm] = 0

    # 뼈별 작은 조각 제거: 각 라벨에서 가장 큰 연결 덩어리만 유지
    struct = np.ones((3, 3, 3))
    for i in np.unique(refined):
        if i == 0:
            continue
        m = refined == i
        cc, n = cc_label(m, structure=struct)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        keep = int(sizes.argmax())
        refined[m & (cc != keep)] = 0

    nib.save(nib.Nifti1Image(refined, lab_img.affine), str(d / "bone_labels_highres.nii.gz"))
    print(f"[{d.name}] threshold={threshold}  refined bones voxel={int((refined>0).sum()):,} "
          f"(TS={int(ts_fg.sum()):,})")

    # 3) 3D 표면 비교: TS raw vs highres refine
    ids = sorted((int(k) for k in names), key=lambda i: -(lab == i).sum())[:2]
    colors = ["#d9c8a9", "#b0c4de"]

    def surf(volume, i):
        m = (volume == i)
        if not m.any():
            return None
        g = pv.ImageData(dimensions=m.shape, spacing=sp)
        g.point_data["v"] = m.astype(np.float32).ravel(order="F")
        s = g.contour([0.5], scalars="v")
        return s if s.n_points else None

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1100, 700))
    for col, (vol, title) in enumerate([(lab, "TS raw (1.5mm)"),
                                        (refined, f"Hybrid highres (0.625mm, T={threshold})")]):
        pl.subplot(0, col)
        for k, i in enumerate(ids):
            s = surf(vol, i)
            if s is not None:
                pl.add_mesh(s, color=colors[k % 2], smooth_shading=True, specular=0.3)
        pl.add_text(title, font_size=11)
        pl.camera_position = "xz"
    out = d / "qc_highres.png"
    pl.screenshot(str(out))
    pl.close()
    print(f"[{d.name}] 저장: {out}")


if __name__ == "__main__":
    thr = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    refine(sys.argv[1], thr)
