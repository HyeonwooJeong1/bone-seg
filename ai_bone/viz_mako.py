"""
viz_mako.py — Mako 추론 결과 3D 시각화 (부위별 맞는 모델 라벨만 병합)

블록별로 담당 모델의 뼈만 취해 3D 표면 렌더:
  - block2(고관절): 476 (Femur, Hip, Sacrum)
  - block1(무릎):   476(Femur, Patella) + 481(Tibia, Fibula)
  - block0(발목):   481 (Tibia, Fibula, Talus, Calcaneus, Tarsals, Metatarsals, Phalanges)

실행: python ai_bone/viz_mako.py <pred_dir> <block> <out.png>
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 클래스 이름→id (dataset.json 없이 로컬 렌더용 하드코딩)
NAME2ID = {
    476: {"Femur_L": 1, "Femur_R": 2, "Hip_L": 3, "Hip_R": 4, "Sacrum": 5,
          "Patella_L": 6, "Patella_R": 7, "Tibia_L": 8, "Tibia_R": 9,
          "Fibula_L": 10, "Fibula_R": 11},
    481: {"Tibia_L": 1, "Tibia_R": 2, "Fibula_L": 3, "Fibula_R": 4,
          "Talus_L": 5, "Talus_R": 6, "Calcaneus_L": 7, "Calcaneus_R": 8,
          "Tarsals_L": 9, "Tarsals_R": 10, "Metatarsals_L": 11, "Metatarsals_R": 12,
          "Phalanges_L": 13, "Phalanges_R": 14},
}
# 블록별 담당 모델의 취할 뼈 (부위 밖 오분류 제외)
BLOCK_MODELS = {
    0: [(481, ["Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R", "Talus_L", "Talus_R",
              "Calcaneus_L", "Calcaneus_R", "Tarsals_L", "Tarsals_R",
              "Metatarsals_L", "Metatarsals_R", "Phalanges_L", "Phalanges_R"])],
    1: [(476, ["Femur_L", "Femur_R", "Patella_L", "Patella_R"]),
        (481, ["Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R"])],
    2: [(476, ["Femur_L", "Femur_R", "Hip_L", "Hip_R", "Sacrum"])],
}
DS_NAME = {476: "PelvisThighs", 481: "ShanksFeet"}


def load_labelmap(pred_dir, did):
    img = nib.load(f"{pred_dir}/pred_{did}/mako_block{BLK}.nii.gz")
    return np.asanyarray(img.dataobj), NAME2ID[did], np.abs(np.diag(img.affine))[:3]


def run(pred_dir, block, out_png):
    global BLK
    BLK = block
    pl = pv.Plotter(off_screen=True, window_size=(1100, 1000))
    tab = plt.get_cmap("tab20")
    ci = 0
    legend = []
    for did, bones in BLOCK_MODELS[block]:
        lab, name2id, sp = load_labelmap(pred_dir, did)
        for b in bones:
            cid = name2id.get(b)
            if cid is None:
                continue
            m = lab == cid
            if m.sum() < 200:
                continue
            g = pv.ImageData(dimensions=m.shape, spacing=sp)
            g.point_data["v"] = m.astype(np.float32).ravel(order="F")
            s = g.contour([0.5], scalars="v")
            if s.n_points == 0:
                continue
            s = s.smooth_taubin(n_iter=20, pass_band=0.1)
            color = tab(ci % 20)[:3]; ci += 1
            pl.add_mesh(s, color=color, smooth_shading=True, specular=0.3)
            legend.append((b, color))
    pl.camera_position = "xz"
    blk_name = {0: "발목/발", 1: "무릎", 2: "고관절"}[block]
    pl.add_text(f"Mako block{block} ({blk_name}) — {len(legend)} bones", font_size=11)
    pl.screenshot(out_png)
    pl.close()
    print(f"저장: {out_png} | 뼈: {[n for n,_ in legend]}")


if __name__ == "__main__":
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "/data1/bone/mako/pred_07049679"
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out_png = sys.argv[3] if len(sys.argv) > 3 else f"/data1/bone/mako/viz_block{block}.png"
    run(pred_dir, block, out_png)
