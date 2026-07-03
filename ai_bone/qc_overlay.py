"""
qc_overlay.py — 분할 품질 육안 검증용 오버레이 PNG 생성

블록 출력 폴더(ct_hu.nii.gz + bone_labels.nii.gz + bone_labels.json)를 읽어
뼈가 가장 많은 axial/coronal 슬라이스에 라벨 마스크를 색으로 덧씌운 PNG 저장.

실행:
  ct_env python ai_bone\\qc_overlay.py <block_out_dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def overlay(block_dir):
    d = Path(block_dir)
    ct = nib.load(str(d / "ct_hu.nii.gz")).get_fdata()          # (nx,ny,nz)
    lab = np.asanyarray(nib.load(str(d / "bone_labels.nii.gz")).dataobj)
    names = json.loads((d / "bone_labels.json").read_text(encoding="utf-8"))
    ids = sorted(int(k) for k in names)
    if not ids:
        print(f"[{d.name}] 뼈 라벨 없음 — 건너뜀"); return

    # 라벨별 색상
    palette = plt.get_cmap("tab10")
    cmap_colors = {i: palette((idx) % 10) for idx, i in enumerate(ids)}

    # 뼈 voxel이 가장 많은 axial slice(z) 및 coronal slice(y)
    z_counts = (lab > 0).sum(axis=(0, 1))
    y_counts = (lab > 0).sum(axis=(0, 2))
    zc = int(np.argmax(z_counts))
    yc = int(np.argmax(y_counts))

    def draw(ax, ct2d, lab2d, title):
        ax.imshow(ct2d.T, cmap="gray", vmin=-200, vmax=1500, origin="lower")
        rgba = np.zeros(lab2d.T.shape + (4,))
        for i in ids:
            m = lab2d.T == i
            rgba[m] = mcolors.to_rgba(cmap_colors[i], alpha=0.45)
        ax.imshow(rgba, origin="lower")
        ax.set_title(title, fontsize=9); ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    draw(axes[0], ct[:, :, zc], lab[:, :, zc], f"{d.name}  axial z={zc}")
    draw(axes[1], ct[:, yc, :], lab[:, yc, :], f"{d.name}  coronal y={yc}")

    # 범례
    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=9,
               markerfacecolor=cmap_colors[i], label=names[str(i)]) for i in ids]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(ids), 5),
               fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.16)
    out = d / "qc_overlay.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[{d.name}] 저장: {out}  (bones: {', '.join(names[str(i)] for i in ids)})")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        overlay(arg)
