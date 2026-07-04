"""render_clean.py — clean_block*.nii.gz를 약한 스무딩으로 3D 렌더(육안 QA).

gaussian 없이 라벨 mask → marching cubes contour → 약한 taubin (표면만 다듬기).
블록은 물리 z 위치로 배치. 실행:
  ct_env python ai_bone/render_clean.py <dir> <out.png>
"""
import sys
import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True
d = sys.argv[1] if len(sys.argv) > 1 else "mako_pred_490"
out = sys.argv[2] if len(sys.argv) > 2 else "mako_pred_490/clean_490.png"

tab = plt.get_cmap("tab20")
pl = pv.Plotter(off_screen=True, window_size=(900, 1300))
present = []
for b in [0, 1, 2]:
    p = f"{d}/clean_block{b}.nii.gz"
    try:
        img = nib.load(p)
    except Exception:
        continue
    a = np.asanyarray(img.dataobj)
    sp = np.abs(np.diag(img.affine))[:3]
    oz = img.affine[2, 3]
    for cid in np.unique(a):
        if cid == 0:
            continue
        m = (a == cid).astype(np.float32)
        if m.sum() < 200:
            continue
        g = pv.ImageData(dimensions=m.shape, spacing=sp, origin=(0, 0, oz))
        g.point_data["v"] = m.ravel(order="F")
        s = g.contour([0.5], scalars="v")           # gaussian 없이 바로 등고선
        if s.n_points == 0:
            continue
        s = s.smooth_taubin(n_iter=12, pass_band=0.1)   # 표면만 약하게
        pl.add_mesh(s, color=tab((int(cid) - 1) % 20)[:3], smooth_shading=True, specular=0.3)
        present.append(int(cid))
pl.camera_position = "xz"
pl.add_text(f"Mako 통합모델(490) — {len(present)} bones", font_size=12)
pl.screenshot(out)
pl.close()
print(f"저장: {out} | 라벨: {sorted(set(present))}")
