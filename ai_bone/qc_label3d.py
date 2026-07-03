"""qc_label3d.py — nnU-Net 라벨 nii.gz의 클래스별 3D 표면 렌더 (학습 GT 품질 확인)

실행: ct_env python ai_bone\\qc_label3d.py <label.nii.gz> <dataset.json> <out.png>
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
import pyvista as pv

pv.OFF_SCREEN = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def run(lab_path, dj_path, out_png):
    img = nib.load(lab_path)
    lab = np.asanyarray(img.dataobj).astype(np.uint8)
    sp = np.abs(np.diag(img.affine))[:3]
    dj = json.loads(Path(dj_path).read_text(encoding="utf-8"))
    id2name = {v: k for k, v in dj["labels"].items() if v != 0}

    import matplotlib.pyplot as plt
    tab = plt.get_cmap("tab20")

    pl = pv.Plotter(off_screen=True, window_size=(900, 1000))
    handles = []
    for idx, (cid, name) in enumerate(sorted(id2name.items())):
        m = lab == cid
        if not m.any():
            continue
        g = pv.ImageData(dimensions=m.shape, spacing=sp)
        g.point_data["v"] = m.astype(np.float32).ravel(order="F")
        s = g.contour([0.5], scalars="v")
        if s.n_points == 0:
            continue
        s = s.smooth_taubin(n_iter=20, pass_band=0.1)
        color = tab(idx % 20)[:3]
        pl.add_mesh(s, color=color, smooth_shading=True, specular=0.3)
    pl.camera_position = "yz"
    pl.add_text(Path(lab_path).stem, font_size=10)
    pl.screenshot(out_png)
    pl.close()
    print(f"저장: {out_png}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
