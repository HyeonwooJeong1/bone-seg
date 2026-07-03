"""
qc_surface.py — AI 마스크의 3D 표면 품질 비교 (우둘투둘 원인/해법 실증)

블록의 bone_labels에서 뼈별 마스크를 꺼내 3가지 방식으로 표면 생성 후
off-screen 렌더 PNG 비교:
  (A) raw           : 마스크 그대로 marching cubes (1.5mm 계단)
  (B) gaussian      : 마스크 gaussian smooth 후 contour
  (C) gaussian+taubin: 위 + Taubin mesh smoothing (앱 파이프라인)

실행: ct_env python ai_bone\\qc_surface.py <block_out_dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import pyvista as pv

pv.OFF_SCREEN = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def make_surface(mask, spacing, mode):
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 1)  # cell data
    # point-based: use dimensions = shape, scalars on points
    grid = pv.ImageData(dimensions=mask.shape, spacing=spacing)
    if mode == "raw":
        vals = mask.astype(np.float32)
    else:
        # z축 계단이 가장 크므로 z에 더 큰 sigma (voxel 단위)
        vals = gaussian_filter(mask.astype(np.float32), sigma=(1.2, 1.2, 2.0))
    grid.point_data["v"] = vals.ravel(order="F")
    surf = grid.contour([0.5], scalars="v")
    if surf.n_points == 0:
        return surf
    if mode == "gauss_taubin":
        surf = surf.smooth_taubin(n_iter=60, pass_band=0.05)
    return surf


def run(block_dir):
    d = Path(block_dir)
    lab_img = nib.load(str(d / "bone_labels.nii.gz"))
    lab = np.asanyarray(lab_img.dataobj)
    names = json.loads((d / "bone_labels.json").read_text(encoding="utf-8"))
    sp = np.abs(np.diag(lab_img.affine))[:3]

    # 가장 큰 뼈 2개만 (femur, tibia 등)
    ids = sorted((int(k) for k in names), key=lambda i: -(lab == i).sum())[:2]
    print(f"[{d.name}] 표면 대상: {[names[str(i)] for i in ids]}  spacing={sp.round(3)}")

    modes = [("raw", "A) raw (1.5mm 계단)"),
             ("gauss", "B) gaussian smooth"),
             ("gauss_taubin", "C) gaussian + Taubin")]
    colors = ["#d9c8a9", "#b0c4de"]

    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1500, 600))
    for col, (mode, title) in enumerate(modes):
        pl.subplot(0, col)
        for k, i in enumerate(ids):
            m = (lab == i)
            surf = make_surface(m, sp, mode)
            if surf.n_points:
                pl.add_mesh(surf, color=colors[k % 2], smooth_shading=True, specular=0.3)
        pl.add_text(title, font_size=10)
        pl.camera_position = "xz"
    out = d / "qc_surface.png"
    pl.screenshot(str(out))
    pl.close()
    print(f"[{d.name}] 저장: {out}")


if __name__ == "__main__":
    run(sys.argv[1])
