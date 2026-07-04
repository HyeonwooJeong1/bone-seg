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


def _ascii_temp_dir():
    """한글 사용자명(예: C:\\Users\\정현우\\...) 회피용 ASCII 임시 베이스.

    nnU-Net(SimpleITK)이 non-ASCII 경로의 이미지를 못 열어 워커가 죽으므로,
    임시 작업폴더를 완전 ASCII 경로(C:\\Temp 등)에 만든다. 8.3 단축은 사용자
    프로필 폴더엔 적용 안 돼(정현우 유지) 소용없음이 확인됨.
    """
    if os.name != "nt":
        return None
    for base in (r"C:\Temp", r"C:\ai_tmp", r"C:\Windows\Temp"):
        if all(ord(c) < 128 for c in base):
            try:
                os.makedirs(base, exist_ok=True)
                t = tempfile.mkdtemp(prefix="probe_", dir=base)  # 쓰기 가능 확인
                os.rmdir(t)
                return base
            except Exception:
                continue
    return None


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


def ensure_trainer():
    """번들된 커스텀 트레이너(nnUNetTrainerNoMirroring_ES)를 설치된 nnunetv2에 보장.

    예측 시 nnU-Net이 이 트레이너 클래스를 import해 네트워크를 구성하므로,
    없으면 번들 .py를 패키지에 복사한다(오프라인 자체완결).
    """
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "nnUNetTrainerNoMirroring_ES.py")
    try:
        import nnunetv2
        dst_dir = os.path.join(os.path.dirname(nnunetv2.__file__),
                               "training", "nnUNetTrainer")
        dst = os.path.join(dst_dir, "nnUNetTrainerNoMirroring_ES.py")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print("[infer_app] ES 트레이너를 nnunetv2에 설치", flush=True)
    except Exception as e:
        print(f"[infer_app] 트레이너 설치 확인 실패(계속): {e}", flush=True)


def _avail_ram_bytes():
    """여유 RAM(bytes). Windows는 GlobalMemoryStatusEx, 그 외는 보수적 기본값."""
    try:
        import ctypes
        class M(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong),
                        ("e", ctypes.c_ulonglong)]
        s = M(); s.l = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.avail)
    except Exception:
        return 8 * 10**9


def _max_block_voxels(in_dir):
    import glob
    mx = 0
    for f in glob.glob(os.path.join(in_dir, "*_0000.nii.gz")):
        mx = max(mx, int(np.prod(nib.load(f).shape)))
    return mx


def _predict_lowmem(predictor, in_dir, out_dir, dev):
    """저메모리 예측: GPU에서 argmax → 라벨(1채널)만 리샘플(원본해상도).

    nnU-Net 기본 export는 22채널 logit을 원본해상도 float64로 리샘플해 RAM을 크게
    먹는다(무릎블록 16.6GB). 여기선 logit을 GPU에서 argmax한 뒤 라벨만 리샘플하므로
    수십분의 1 메모리로 동일 결과(경계 스무딩은 앱에서 별도 수행하므로 품질 영향 미미).
    """
    import glob
    import torch
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    try:
        from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    except Exception:
        from nnunetv2.utilities.bounding_boxes import bounding_box_to_slice  # fallback

    pm = predictor.plans_manager
    cm = predictor.configuration_manager
    dj = predictor.dataset_json
    pp = DefaultPreprocessor(verbose=False)

    for f in sorted(glob.glob(os.path.join(in_dir, "*_0000.nii.gz"))):
        data, _seg, props = pp.run_case([f], None, pm, cm, dj)
        logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data))
        # GPU에서 argmax (메모리 절약) → 라벨
        if not isinstance(logits, torch.Tensor):
            logits = torch.from_numpy(logits)
        label_iso = torch.argmax(logits, dim=0).to(torch.uint8).cpu().numpy()
        del logits
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # 라벨을 원본(크롭) 해상도로 리샘플 후 bbox로 원위치
        tgt = props["shape_after_cropping_and_before_resampling"]
        new_sp = [props["spacing"][i] for i in pm.transpose_forward]
        lab_res = cm.resampling_fn_seg(label_iso[None], tgt, cm.spacing, new_sp)[0]
        out = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
        out[bounding_box_to_slice(props["bbox_used_for_cropping"])] = lab_res
        base = os.path.basename(f).replace("_0000.nii.gz", ".nii.gz")
        SimpleITKIO().write_seg(out, os.path.join(out_dir, base), props)
        print(f"[infer_app][lowmem] {base} done", flush=True)


def run_predict(in_dir, out_dir, model_dir, folds, device):
    """번들 모델로 추론 — nnUNetPredictor Python API 직접 호출.

    여유 RAM을 자동 감지해, 확률 리샘플이 안 들어가면 저메모리(라벨 리샘플) 모드로 전환.
    CLI(nnUNetv2_predict) console script에 의존하지 않아 PATH/설치 위치 문제가 없다.
    """
    ensure_trainer()
    os.environ["nnUNet_results"] = model_dir
    os.environ.setdefault("nnUNet_raw", model_dir)
    os.environ.setdefault("nnUNet_preprocessed", model_dir)
    os.environ["nnUNet_compile"] = "f"

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    ds = next(d for d in os.listdir(model_dir) if d.startswith("Dataset490"))
    trainer_dir = next(d for d in os.listdir(os.path.join(model_dir, ds)) if d.endswith("3d_fullres"))
    model_folder = os.path.join(model_dir, ds, trainer_dir)
    print(f"[infer_app] device={dev} folds={folds}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=(dev.type == "cuda"),
        device=dev, verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=tuple(int(f) for f in folds),
        checkpoint_name="checkpoint_final.pth")

    # 자원 자동 감지: 확률 리샘플(22채널 float64) 예상 vs 여유 RAM
    n_cls = predictor.label_manager.num_segmentation_heads
    peak = _max_block_voxels(in_dir) * n_cls * 8 * 2   # float64 + 여유
    avail = _avail_ram_bytes()
    print(f"[infer_app] RAM 여유 {avail/1e9:.1f}GB, 확률리샘플 예상 {peak/1e9:.1f}GB", flush=True)
    if avail > peak:
        print("[infer_app] 고품질 모드(확률 리샘플)", flush=True)
        predictor.predict_from_files_sequential(
            in_dir, out_dir, save_probabilities=False, overwrite=True,
            folder_with_segs_from_prev_stage=None)
    else:
        print("[infer_app] 저메모리 모드(GPU argmax + 라벨 리샘플)", flush=True)
        _predict_lowmem(predictor, in_dir, out_dir, dev)


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

    ascii_base = _ascii_temp_dir()     # 한글경로 회피 (SimpleITK)
    if ascii_base:
        os.environ["TMP"] = ascii_base
        os.environ["TEMP"] = ascii_base
    work = tempfile.mkdtemp(prefix="aiseg_", dir=ascii_base)
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
        # 라벨·CT를 둘 다 SimpleITK로 읽어 동일한 (z,y,x) 축 순서로 정합.
        # (앱 image_hu도 (nz,ny,nx) 순서라 그대로 저장하면 됨)
        import SimpleITK as sitk
        out = {"n_blocks": n, "id2name": json.dumps(ID2NAME, ensure_ascii=False)}
        for i in range(n):
            lab_zyx = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(pred_dir, f"st{i:02d}.nii.gz")))    # (z,y,x)
            ct_zyx = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(in_dir, f"st{i:02d}_0000.nii.gz")))  # (z,y,x)
            # nnU-Net writer가 입력과 다른 축 순서로 저장할 수 있어 CT 기준으로 정합.
            # (축상 512x512, z축 크기 고유 → 크기 매칭으로 안전. x/y 순서는 보존)
            if lab_zyx.shape != ct_zyx.shape:
                perm, used = [], set()
                for tsz in ct_zyx.shape:
                    for ax in range(lab_zyx.ndim):
                        if ax not in used and lab_zyx.shape[ax] == tsz:
                            perm.append(ax); used.add(ax); break
                if len(perm) == lab_zyx.ndim:
                    lab_zyx = np.transpose(lab_zyx, perm)
            lab_clean = postprocess_block(lab_zyx.astype(np.uint8), ct_zyx).astype(np.uint8)
            aff = cts[i].affine
            nzz = lab_clean.shape[0]   # z 축이 0
            z0 = float(aff[2, 3]); z1 = float(aff[2, 3] + aff[2, 2] * (nzz - 1))
            sp_xyz = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))   # (x,y,z)
            out[f"block{i}_label"] = lab_clean                 # 이미 (z,y,x) = 앱 순서
            out[f"block{i}_zrange"] = np.array([min(z0, z1), max(z0, z1)], dtype=np.float64)
            out[f"block{i}_shape"] = np.array(lab_clean.shape, dtype=np.int64)
            out[f"block{i}_spacing"] = np.array(
                [sp_xyz[2], sp_xyz[1], sp_xyz[0]], dtype=np.float64)  # (z,y,x)
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
