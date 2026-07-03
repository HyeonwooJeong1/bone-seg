"""
qc_grid.py — 한 블록의 axial 슬라이스를 여러 장 격자로 오버레이 (분할 실태 정밀 점검)

실행: ct_env python ai_bone\\qc_grid.py <block_out_dir> [n=12]
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


def grid(block_dir, n=12):
    d = Path(block_dir)
    ct = nib.load(str(d / "ct_hu.nii.gz")).get_fdata()
    lab = np.asanyarray(nib.load(str(d / "bone_labels.nii.gz")).dataobj)
    names = json.loads((d / "bone_labels.json").read_text(encoding="utf-8"))
    ids = sorted(int(k) for k in names)
    palette = plt.get_cmap("tab10")
    cmap_colors = {i: palette(idx % 10) for idx, i in enumerate(ids)}

    nz = ct.shape[2]
    # 뼈가 있는 z 범위에서 균등 샘플
    zc = (lab > 0).sum(axis=(0, 1))
    zvalid = np.where(zc > 0)[0]
    if len(zvalid) == 0:
        print(f"[{d.name}] 뼈 없음"); return
    z0, z1 = zvalid[0], zvalid[-1]
    zs = np.linspace(z0, z1, n).astype(int)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, z in zip(axes, zs):
        ax.imshow(ct[:, :, z].T, cmap="gray", vmin=-200, vmax=1500, origin="lower")
        l2 = lab[:, :, z].T
        rgba = np.zeros(l2.shape + (4,))
        for i in ids:
            rgba[l2 == i] = mcolors.to_rgba(cmap_colors[i], alpha=0.5)
        ax.imshow(rgba, origin="lower")
        ax.set_title(f"z={z}", fontsize=8); ax.axis("off")
    for ax in axes[len(zs):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=8,
               markerfacecolor=cmap_colors[i], label=names[str(i)]) for i in ids]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(ids), 5), fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.08)
    out = d / "qc_grid.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[{d.name}] 저장: {out}")


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    grid(sys.argv[1], n)
