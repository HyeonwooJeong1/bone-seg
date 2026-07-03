"""
postprocess_mako.py — 추론 라벨 후처리 (금속봉+아티팩트 제거, 뼈 조각 보존/연결)

문제: Mako CT의 금속 봉(HU>=1900, 다리 밖 원형 기구)이 뼈로 오분류.
      봉이 z축으로 길어 주변 streak artifact가 여러 높이에서 뼈로 잘못 잡힘.
해결:
  1) 금속 제거: HU>=METAL_THR voxel을 라벨에서 0
  2) 금속봉 아티팩트 제거: 금속 덩어리 중 '뼈와 붙지 않은(외부)' 것 = 봉.
     그 봉을 크게 dilate한 영역의 라벨을 제거 (봉 주변 오분류 clean).
     - 뼈 내부 고밀도(피질골이 HU>=1900인 부분)는 뼈와 붙어 있으므로 보존.
  3) 뼈별 closing: 금속/아티팩트로 끊긴 같은 뼈를 연결 (조각 보존, largest-CC 안 함)

서버 실행:
  python ai_bone/postprocess_mako.py <pred_dir> [closing_iter=3] [rod_dilate=18]
"""
import sys, os
import numpy as np
import nibabel as nib
from scipy.ndimage import (binary_closing, binary_dilation,
                           generate_binary_structure, label as cclabel)

METAL_THR = 1900


def run(pred_dir, citer=3, leg_dil=6):
    st = generate_binary_structure(3, 1)
    for b in [0, 1, 2]:
        mp = f"{pred_dir}/pred_490/mako_block{b}.nii.gz"
        cp = f"{pred_dir}/in_490/mako_block{b}_0000.nii.gz"
        if not os.path.exists(mp):
            print(f"[skip] block{b}"); continue
        limg = nib.load(mp)
        lab = np.asanyarray(limg.dataobj).astype(np.uint8).copy()
        ct = np.asanyarray(nib.load(cp).dataobj)

        # 금속(HU>=THR) 제거
        metal = ct >= METAL_THR
        n_metal = int((metal & (lab > 0)).sum())
        lab[metal] = 0

        # 대상 다리만 분리 (뼈 안 자름): 뼈를 dilate하면 대상 다리의 뼈들은
        # 관절로 가까워 하나로 연결되고, 반대쪽 다리(수십mm 떨어짐)·봉 파편은
        # 별개 덩어리가 됨. 가장 큰 덩어리(=대상 다리) 영역의 라벨만 유지.
        rod_removed = 0
        bone = lab > 0
        if bone.sum() > 5000:
            bd = binary_dilation(bone, st, iterations=leg_dil)
            cc, n = cclabel(bd, structure=st)
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            main = int(sizes.argmax())
            keep = cc == main                 # 대상 다리 영역 (dilate 포함)
            rod_removed = int((lab > 0)[~keep].sum())
            lab[~keep] = 0                    # 대상 다리 밖(반대다리·봉) 제거

        # L/R 통합: 대상 다리는 한쪽뿐이라 좌우 구분이 무의미.
        # nnU-Net이 한 뼈를 좌우로 섞어 라벨한 것(예 Femur_L 2k + Femur_R 374k, x거리 7mm)을
        # 다수 라벨로 합쳐 하나의 뼈로 만든다.
        # 항상 R(오른쪽 id)로 통합 → 블록마다 라벨이 일관되어야 병합/스무딩 시
        # 대퇴골 등이 한 라벨(=한 색)로 이어짐. (대상 다리 한쪽뿐이라 L/R 이름은 무의미)
        PAIRS = [(1, 2), (3, 4), (6, 7), (8, 9), (10, 11),
                 (12, 13), (14, 15), (16, 17), (18, 19), (20, 21)]
        for lid, rid in PAIRS:
            lab[lab == lid] = rid

        # 뼈별 closing (조각 연결, 배경/금속 아닌 곳만)
        filled = 0
        if citer > 0:
            for cid in np.unique(lab):
                if cid == 0:
                    continue
                m = lab == cid
                mc = binary_closing(m, structure=st, iterations=citer)
                add = mc & (lab == 0) & (~metal)
                lab[add] = cid
                filled += int(add.sum())

        nib.save(nib.Nifti1Image(lab, limg.affine), f"{pred_dir}/clean_block{b}.nii.gz")
        print(f"block{b}: 금속 {n_metal//1000}k, 반대다리·봉 {rod_removed//1000}k 제거, 연결 +{filled//1000}k")
    print("후처리 완료")


if __name__ == "__main__":
    pred = sys.argv[1] if len(sys.argv) > 1 else "/data1/bone/mako/pred_07049679"
    it = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ld = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    run(pred, it, ld)
