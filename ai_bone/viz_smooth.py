"""viz_smooth.py — 표면 다듬기 전/후 비교 (무릎 대퇴골+경골)

좌: raw (라벨 그대로 marching cubes)
우: gaussian(마스크 스무딩) + Taubin mesh smoothing
"""
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import pyvista as pv
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True

# 무릎(block1): 476 Femur_R(2), 481 Tibia_R(2)
JOBS = [("mako_pred/pred_476/mako_block1.nii.gz", 2, "#d9c8a9"),   # Femur_R
        ("mako_pred/pred_481/mako_block1.nii.gz", 2, "#b0c4de")]   # Tibia_R


def surf(path, cid, sp, mode):
    lab = np.asanyarray(nib.load(path).dataobj)
    m = (lab == cid).astype(np.float32)
    if mode == "smooth":
        m = gaussian_filter(m, sigma=(1.2, 1.2, 1.8))
    g = pv.ImageData(dimensions=m.shape, spacing=sp)
    g.point_data["v"] = m.ravel(order="F")
    s = g.contour([0.5], scalars="v")
    if s.n_points and mode == "smooth":
        s = s.smooth_taubin(n_iter=50, pass_band=0.05)
    return s


def main():
    sp = np.abs(np.diag(nib.load(JOBS[0][0]).affine))[:3]
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1300, 800))
    for col, (mode, title) in enumerate([("raw", "다듬기 전 (raw)"),
                                         ("smooth", "다듬기 후 (gaussian+Taubin)")]):
        pl.subplot(0, col)
        for path, cid, color in JOBS:
            s = surf(path, cid, sp, mode)
            if s.n_points:
                pl.add_mesh(s, color=color, smooth_shading=True, specular=0.3)
        pl.add_text(title, font_size=12)
        pl.camera_position = "xz"
    pl.screenshot("mako_pred/smooth_compare.png")
    pl.close()
    print("저장: mako_pred/smooth_compare.png")


if __name__ == "__main__":
    main()
