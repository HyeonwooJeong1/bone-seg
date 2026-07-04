"""
infer_app.py — 오프라인 로컬 추론 모듈 (번들 모델). 앱이 subprocess로 호출.

흐름: DICOM 폴더 → 축상 시리즈 선별 → z-gap 스테이션 분할 → 각 스테이션을
번들된 통합모델(Dataset490)로 예측(21라벨) → 후처리(금속·대상다리·L/R·closing)
→ 앱이 읽는 npz(블록별 라벨 + z범위 + shape + id→이름)로 저장.

- device 자동: GPU 있으면 cuda, 없으면 cpu.
- folds: GPU면 5-fold 앙상블, CPU면 단일 fold 권장(속도). --folds로 지정.
- 모델 위치: 기본 <repo>/models (nnUNet_results 구조). --model로 override.

실행(앱/CLI):
  python -m ai_bone.infer_app <dicom_dir> <out.npz> [--folds 0,1,2,3,4]
                              [--device auto|cuda|cpu] [--model DIR]
"""
import os, sys, json, argparse, tempfile, shutil
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_closing, binary_dilation,
                           generate_binary_structure, label as cclabel)

# DICOM→스테이션→NIfTI 로직 재사용 (standalone import)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_mako import select_axial, split_blocks, build_nifti  # noqa: E402

# 통합 21라벨 id→이름
ID2NAME = {
    1: "Femur_L", 2: "Femur_R", 3: "Hip_L", 4: "Hip_R", 5: "Sacrum",
    6: "Patella_L", 7: "Patella_R", 8: "Tibia_L", 9: "Tibia_R",
    10: "Fibula_L", 11: "Fibula_R", 12: "Talus_L", 13: "Talus_R",
    14: "Calcaneus_L", 15: "Calcaneus_R", 16: "Tarsals_L", 17: "Tarsals_R",
    18: "Metatarsals_L", 19: "Metatarsals_R", 20: "Phalanges_L", 21: "Phalanges_R",
}
# L/R 통합 쌍 (단일 다리라 좌우 무의미 → R로 합침)
LR_PAIRS = [(1, 2), (3, 4), (6, 7), (8, 9), (10, 11),
            (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]
METAL_THR = 1900


def pick_device(arg):
    if arg and arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def default_model_dir():
    # <repo>/models  (nnUNet_results 구조: models/Dataset490_LowerLimb/...)
    return str(Path(__file__).resolve().parent.parent / "models")


def run_predict(in_dir, out_dir, model_dir, folds, device):
    """번들 모델로 nnUNetv2_predict 실행 (subprocess)."""
    import subprocess
    env = dict(os.environ)
    env["nnUNet_results"] = model_dir
    # predict엔 raw/preprocessed 불필요하지만 미설정 시 에러 → 존재 경로로 채움
    env.setdefault("nnUNet_raw", model_dir)
    env.setdefault("nnUNet_preprocessed", model_dir)
    env["nnUNet_compile"] = "f"
    cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir,
           "-d", "490", "-c", "3d_fullres",
           "-p", "nnUNetPlans_iso06", "-tr", "nnUNetTrainerNoMirroring_ES",
           "-f", *[str(f) for f in folds], "-device", device]
    print(f"[infer_app] device={device} folds={folds}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def postprocess_block(lab, ct, citer=3, leg_dil=6):
    """단일 스테이션 라벨 후처리: 금속제거 → 대상다리 추출 → L/R통합 → per-bone closing."""
    st = generate_binary_structure(3, 1)
    lab = lab.astype(np.uint8).copy()
    metal = ct >= METAL_THR
    lab[metal] = 0
    # 대상 다리만 (뼈 dilate → 최대 연결덩어리 유지)
    bone = lab > 0
    if bone.sum() > 5000:
        bd = binary_dilation(bone, st, iterations=leg_dil)
        cc, _ = cclabel(bd, structure=st)
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        keep = cc == int(sizes.argmax())
        lab[~keep] = 0
    # L/R 통합 (항상 R)
    for lid, rid in LR_PAIRS:
        lab[lab == lid] = rid
    # per-bone closing (배경/금속 아닌 곳만)
    if citer > 0:
        for cid in np.unique(lab):
            if cid == 0:
                continue
            m = lab == cid
            mc = binary_closing(m, structure=st, iterations=citer)
            add = mc & (lab == 0) & (~metal)
            lab[add] = cid
    return lab


def run(dicom_dir, out_npz, folds, device, model_dir):
    device = pick_device(device)
    model_dir = model_dir or default_model_dir()
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"모델 폴더 없음: {model_dir}")

    work = tempfile.mkdtemp(prefix="aiseg_")
    in_dir = os.path.join(work, "in"); os.makedirs(in_dir, exist_ok=True)
    pred_dir = os.path.join(work, "pred"); os.makedirs(pred_dir, exist_ok=True)
    try:
        slices = select_axial(dicom_dir)
        blocks = split_blocks(slices)
        n = len(blocks)
        # 스테이션별 CT NIfTI 저장 (예측 입력) + CT 배열 보관(후처리·npz용)
        cts = []
        for i, blk in enumerate(blocks):
            nii = build_nifti(blk)
            nib.save(nii, os.path.join(in_dir, f"st{i:02d}_0000.nii.gz"))
            cts.append(nii)
        run_predict(in_dir, pred_dir, model_dir, folds, device)

        # 후처리 + 앱용 npz 패키징
        out = {"n_blocks": n, "id2name": json.dumps(ID2NAME, ensure_ascii=False)}
        for i in range(n):
            pnii = nib.load(os.path.join(pred_dir, f"st{i:02d}.nii.gz"))
            lab_xyz = np.asanyarray(pnii.dataobj)                 # (x,y,z)
            ct_xyz = np.asanyarray(cts[i].dataobj)                # (x,y,z)
            lab_clean = postprocess_block(lab_xyz, ct_xyz)
            # 앱 image_hu는 (nz,ny,nx). NIfTI (x,y,z) → transpose (z,y,x).
            lab_zyx = np.transpose(lab_clean, (2, 1, 0)).astype(np.uint8)
            aff = pnii.affine
            # 스테이션 world z-범위 (origin z ~ aff[2,3], z-spacing ~ aff[2,2])
            nzz = lab_xyz.shape[2]
            z0 = float(aff[2, 3]); z1 = float(aff[2, 3] + aff[2, 2] * (nzz - 1))
            out[f"block{i}_label"] = lab_zyx
            out[f"block{i}_zrange"] = np.array([min(z0, z1), max(z0, z1)], dtype=np.float64)
            out[f"block{i}_shape"] = np.array(lab_zyx.shape, dtype=np.int64)
        np.savez_compressed(out_npz, **out)
        print(f"[infer_app] 저장: {out_npz} (블록 {n}개)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dicom_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    folds = [int(x) for x in str(a.folds).replace(" ", "").split(",") if x != ""]
    run(a.dicom_dir, a.out_npz, folds, a.device, a.model or None)


if __name__ == "__main__":
    main()
