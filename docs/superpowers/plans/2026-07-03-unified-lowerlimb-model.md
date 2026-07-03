# 통합 단일 하지 뼈 분할 모델 재학습 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 부위별 2모델(476/481) + 병합 구조를 하나의 21라벨 nnU-Net 단일 모델로 대체해 무릎·발목 경계 오분류를 근본 해결한다.

**Architecture:** VSD 26명 × 2스캔(Pelvis-Thighs, Shanks-Feet)을 각각 개별 case로, 통일 21라벨 `Dataset490_LowerLimb`로 변환한다. 등방 0.6mm 커스텀 플랜으로 전처리 후 NoMirroring+EarlyStopping 트레이너로 5-fold 학습, 5-fold 앙상블로 추론한다. Mako 환자는 3블록을 단일 모델로 추론하고 병합 로직 없이 물리 z로 배치한다.

**Tech Stack:** nnU-Net v2, SimpleITK, nibabel, pydicom, PyVista. 로컬 conda `ct_env`(변환·검증·렌더), 서버 conda `pt210_py312`(전처리·학습·추론, H100 8×80GB @ 114.110.134.100).

## Global Constraints

- 신규 데이터셋 ID: **490**, 이름 `Dataset490_LowerLimb`, 라벨 **21개 + background(0)**.
- 통일 라벨 스킴(이름→id) 고정:
  `Femur_L1 Femur_R2 Hip_L3 Hip_R4 Sacrum5 Patella_L6 Patella_R7 Tibia_L8 Tibia_R9 Fibula_L10 Fibula_R11 Talus_L12 Talus_R13 Calcaneus_L14 Calcaneus_R15 Tarsals_L16 Tarsals_R17 Metatarsals_L18 Metatarsals_R19 Phalanges_L20 Phalanges_R21`
- 플랜: `nnUNetPlans_iso06` (등방 0.6mm), configuration `3d_fullres`.
- 트레이너: `nnUNetTrainerNoMirroring_ES` (좌우 라벨 → 미러링 금지 + early stopping).
- CT-라벨 정합 분기(검증된 로직, 반드시 유지): seg와 CT size 같으면 `CopyInformation(ct)`, 다르면 `CopyInformation(seg)` 후 물리좌표 `sitkNearestNeighbor` resample. (size 동일 케이스에 물리좌표 resample 쓰면 flip 메타로 라벨 전멸.)
- 라벨 remap은 **segment 이름 기반**(원본 LabelValue는 부위마다 다름).
- 서버 경로: raw `/data1/bone/ai_bone/nnunet/raw`, preprocessed `/home/ubuntu/nnunet_pre`(로컬 SSD), results `/data1/bone/ai_bone/nnunet/results`, VSD `/data1/bone/ai_bone/data/vsd`.
- 학습 환경 export: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, `nnUNet_compile=f`, `nnUNet_n_proc_DA=18`.
- **기존 476/481 데이터·체크포인트는 삭제 금지**(백업/롤백용). `merge_mako.py`만 제거.

---

### Task 1: `build_unified.py` — 통합 21라벨 데이터셋 변환

**Files:**
- Create: `ai_bone/build_unified.py`

**Interfaces:**
- Produces: `nnunet/raw/Dataset490_LowerLimb/{imagesTr,labelsTr,dataset.json}`. Case 이름 규칙 `LL_{subj}_PT`, `LL_{subj}_SF`. dataset.json labels = background + 21.

- [ ] **Step 1: `build_unified.py` 작성**

`convert_to_nnunet.py`의 검증된 로직(이름맵 파싱, size 분기 정합, robust CT 선택)을 재사용하되, 환자마다 **두 부위 스캔을 모두** 통일 21라벨로 변환한다.

```python
"""
build_unified.py — VSD 26명 × 2스캔(Pelvis-Thighs, Shanks-Feet)을
하나의 통일 21라벨 nnU-Net 데이터셋(Dataset490_LowerLimb)으로 변환.

각 스캔을 개별 case로 만든다: LL_{subj}_PT, LL_{subj}_SF.
라벨은 seg.nrrd segment 이름 → 통일 21라벨 id로 remap(부위 무관, 이름 기반).

실행:
  ct_env python ai_bone/build_unified.py
"""
import sys, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VSD = Path("ai_bone/data/vsd")
RAW = Path("ai_bone/nnunet/raw")

DATASET_ID = 490
DATASET_NAME = "LowerLimb"

# 통일 21라벨 (배경=0) — Global Constraints와 반드시 일치
UNI = {
    "Femur_L": 1, "Femur_R": 2, "Hip_L": 3, "Hip_R": 4, "Sacrum": 5,
    "Patella_L": 6, "Patella_R": 7, "Tibia_L": 8, "Tibia_R": 9,
    "Fibula_L": 10, "Fibula_R": 11, "Talus_L": 12, "Talus_R": 13,
    "Calcaneus_L": 14, "Calcaneus_R": 15, "Tarsals_L": 16, "Tarsals_R": 17,
    "Metatarsals_L": 18, "Metatarsals_R": 19, "Phalanges_L": 20, "Phalanges_R": 21,
}
# 부위 태그 → case suffix
PARTS = [("Pelvis-Thighs", "PT"), ("Shanks-Feet", "SF")]


def read_seg_namemap(seg_img):
    """seg.nrrd 메타 → {원본 label value(int): segment name}"""
    segs = {}
    for k in seg_img.GetMetaDataKeys():
        if k.endswith("_Name"):
            segs.setdefault(k.split("_")[0], {})["name"] = seg_img.GetMetaData(k)
        if k.endswith("_LabelValue"):
            segs.setdefault(k.split("_")[0], {})["lv"] = int(seg_img.GetMetaData(k))
    return {v["lv"]: v["name"] for v in segs.values() if "lv" in v and "name" in v}


def convert_scan(rec, tag, case, img_dir, lab_dir):
    """단일 스캔(rec: Reconstruction.seg.nrrd)을 통일 라벨 case로 변환. 성공 시 present 라벨 반환."""
    scan_folder = rec.parent
    # CT 선택: 부위명이 파일명에 있으면 그걸로, 없으면(폴더가 부위별이라 CT 하나) 그 CT로.
    cands = [p for p in scan_folder.glob("*.nrrd") if not p.name.endswith("seg.nrrd")]
    tagged = [p for p in cands if tag in p.name]
    ct_list = tagged if tagged else cands
    if not ct_list:
        print(f"[skip] {case}: CT 없음"); return None
    ct = sitk.ReadImage(str(ct_list[0]))
    sitk.WriteImage(ct, str(img_dir / f"{case}_0000.nii.gz"))

    seg = sitk.ReadImage(str(rec))
    arr = sitk.GetArrayFromImage(seg)
    namemap = read_seg_namemap(seg)
    out = np.zeros_like(arr, dtype=np.uint8)
    for origval, nm in namemap.items():
        if nm in UNI:
            out[arr == origval] = UNI[nm]
    out_img = sitk.GetImageFromArray(out)
    if seg.GetSize() == ct.GetSize():
        out_img.CopyInformation(ct)
    else:
        out_img.CopyInformation(seg)
        out_img = sitk.Resample(out_img, ct, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sitk.WriteImage(out_img, str(lab_dir / f"{case}.nii.gz"))
    present = sorted({namemap[v] for v in np.unique(arr) if v in namemap})
    print(f"[ok] {case}  labels={len(present)}  {present}")
    return present


def main():
    ds_dir = RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    img_dir = ds_dir / "imagesTr"; lab_dir = ds_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p for p in VSD.iterdir() if p.is_dir()])
    n = 0
    for subj in subjects:
        for tag, suffix in PARTS:
            recs = list(subj.glob(f"*/*-{tag}_Reconstruction.seg.nrrd"))
            if not recs:
                continue
            case = f"LL_{subj.name}_{suffix}"
            if convert_scan(recs[0], tag, case, img_dir, lab_dir) is not None:
                n += 1

    labels = {"background": 0}
    labels.update(UNI)
    dj = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": n,
        "file_ending": ".nii.gz",
    }
    with open(ds_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dj, f, ensure_ascii=False, indent=2)
    print(f"\n=== {ds_dir.name}: {n} cases, {len(UNI)} classes ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 로컬에서 스모크 테스트(016만 압축해제됨)**

Run:
```bash
cd "C:/Users/정현우/Desktop/VPI LAB/stanford medicine" && ct_env python ai_bone/build_unified.py
```
Expected: `LL_016_PT`, `LL_016_SF` 두 case 생성, 각 `[ok] ... labels=N [...]` 출력. 마지막에 `Dataset490_LowerLimb: 2 cases, 21 classes`(로컬엔 016만 풀려 있으므로 2 case가 정상).

- [ ] **Step 3: 산출물 스팟체크 — 라벨 id가 21스킴을 따르는지**

Run:
```bash
cd "C:/Users/정현우/Desktop/VPI LAB/stanford medicine" && PYTHONIOENCODING=utf-8 "/c/ProgramData/anaconda3/envs/ct_env/python.exe" -c "
import SimpleITK as sitk, numpy as np
for c in ['LL_016_PT','LL_016_SF']:
    a=sitk.GetArrayFromImage(sitk.ReadImage(f'ai_bone/nnunet/raw/Dataset490_LowerLimb/labelsTr/{c}.nii.gz'))
    print(c, 'ids=', sorted(int(x) for x in np.unique(a) if x>0))
"
```
Expected: `LL_016_PT ids= [...]` 는 1~11 범위(대퇴/골반/무릎), `LL_016_SF ids= [...]` 는 8~21 범위(하퇴/발). 두 case 모두 비어있지 않음.

- [ ] **Step 4: Commit**

```bash
git add ai_bone/build_unified.py
git commit -m "feat: add unified 21-label dataset builder (Dataset490)"
```

---

### Task 2: `verify_dataset.py` — Dataset490 검증 추가

**Files:**
- Modify: `ai_bone/verify_dataset.py:18` (DATASETS 목록)

**Interfaces:**
- Consumes: Task 1의 `Dataset490_LowerLimb`.
- Produces: 검증 리포트(빈 라벨/기하/overlap/라벨값/21라벨 분포). 게이트: 빈 라벨 0, 기하 불일치 0, overlap 평균 ≥ 90%.

- [ ] **Step 1: DATASETS에 490 추가**

`ai_bone/verify_dataset.py`의 18번째 줄:
```python
DATASETS = ["Dataset476_PelvisThighs", "Dataset481_ShanksFeet"]
```
을 다음으로 교체:
```python
DATASETS = ["Dataset490_LowerLimb"]
```
(476/481은 이미 검증 완료·백업이므로 신규 490만 검증. 필요 시 세 개 다 둬도 무방.)

- [ ] **Step 2: 서버에서 전체 변환 후 검증 (전 subject 압축해제된 서버에서 실행)**

먼저 서버에서 전체 변환:
```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && python ai_bone/build_unified.py"
```
Expected: `Dataset490_LowerLimb: ~52 cases, 21 classes` (26명 × 2스캔; 일부 결측 시 약간 적을 수 있음).

- [ ] **Step 3: 검증 실행**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && python ai_bone/verify_dataset.py"
```
Expected:
- `[1] 빈 라벨: 0개 (없음 ✓)`
- `[2] geometry 불일치: 0개 (없음 ✓)`
- `[3] CT뼈 정합 낮음(<80%): 0개` 및 겹침율 평균 ≥ 90%
- `[4] 라벨값 이상: 0개`
- `[5]` 21개 클래스 각각 등장 case 수 표시(대퇴/경골 등은 다수, 발뼈도 ≥ 3). 희귀(⚠️) 클래스가 있으면 원인 확인(해당 뼈가 스캔 FOV에 드문 경우).

만약 빈 라벨/기하 불일치가 나오면 Task 1의 size 분기·이름맵을 해당 case로 디버그 후 재변환.

- [ ] **Step 4: Commit**

```bash
git add ai_bone/verify_dataset.py
git commit -m "chore: point dataset verification at unified Dataset490"
```

---

### Task 3: 등방 0.6mm 커스텀 플랜 생성 + 전처리

**Files:**
- Create: `ai_bone/make_iso06_plan.py`

**Interfaces:**
- Consumes: `Dataset490_LowerLimb` raw.
- Produces: `nnUNet_preprocessed/Dataset490_LowerLimb/nnUNetPlans_iso06.json` + 전처리된 3d_fullres 데이터. 후속 학습/추론이 `-p nnUNetPlans_iso06`로 참조.

- [ ] **Step 1: fingerprint 추출 + 기본 플랜 생성 (서버)**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && \
  export nnUNet_raw=/data1/bone/ai_bone/nnunet/raw nnUNet_preprocessed=/home/ubuntu/nnunet_pre nnUNet_results=/data1/bone/ai_bone/nnunet/results && \
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && \
  nnUNetv2_extract_fingerprint -d 490 && nnUNetv2_plan_experiment -d 490"
```
Expected: `nnUNet_preprocessed/Dataset490_LowerLimb/nnUNetPlans.json` 생성.

- [ ] **Step 2: `make_iso06_plan.py` 작성 — 3d_fullres를 등방 0.6mm로 복제**

```python
"""
make_iso06_plan.py — 기본 nnUNetPlans.json의 3d_fullres를 복제해
target spacing을 등방 0.6mm로 바꾼 nnUNetPlans_iso06.json 생성.

실행(서버):
  nnUNet_preprocessed=/home/ubuntu/nnunet_pre python ai_bone/make_iso06_plan.py 490
"""
import sys, os, json, copy

did = int(sys.argv[1]) if len(sys.argv) > 1 else 490
pre = os.environ["nnUNet_preprocessed"]
# 데이터셋 폴더명 자동 탐색
ds = next(d for d in os.listdir(pre) if d.startswith(f"Dataset{did:03d}_"))
base = os.path.join(pre, ds, "nnUNetPlans.json")
plans = json.load(open(base))

cfg = copy.deepcopy(plans["configurations"]["3d_fullres"])
cfg["spacing"] = [0.6, 0.6, 0.6]          # 등방 0.6mm target
plans["plans_name"] = "nnUNetPlans_iso06"
plans["configurations"] = {"3d_fullres": cfg}

out = os.path.join(pre, ds, "nnUNetPlans_iso06.json")
json.dump(plans, open(out, "w"), indent=2)
print(f"저장: {out}  (3d_fullres spacing={cfg['spacing']}, patch={cfg.get('patch_size')})")
```

- [ ] **Step 3: iso06 플랜 생성 실행 (서버)**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && \
  nnUNet_preprocessed=/home/ubuntu/nnunet_pre python ai_bone/make_iso06_plan.py 490"
```
Expected: `저장: .../nnUNetPlans_iso06.json (3d_fullres spacing=[0.6, 0.6, 0.6], patch=...)`.

- [ ] **Step 4: iso06 플랜으로 전처리 (서버)**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && \
  export nnUNet_raw=/data1/bone/ai_bone/nnunet/raw nnUNet_preprocessed=/home/ubuntu/nnunet_pre nnUNet_results=/data1/bone/ai_bone/nnunet/results && \
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && \
  nnUNetv2_preprocess -d 490 -plans_name nnUNetPlans_iso06 -c 3d_fullres -np 8"
```
Expected: `nnUNet_preprocessed/Dataset490_LowerLimb/nnUNetPlans_iso06_3d_fullres/` 에 전처리 .npy/.npz 생성(약 52 case). 오류 없이 완료.

- [ ] **Step 5: Commit**

```bash
git add ai_bone/make_iso06_plan.py
git commit -m "feat: iso 0.6mm custom plan generator for Dataset490"
```

---

### Task 4: 5-fold 학습 스크립트

**Files:**
- Create: `ai_bone/train_unified.sh`

**Interfaces:**
- Consumes: Task 3 전처리 데이터, 서버에 설치된 `nnUNetTrainerNoMirroring_ES`.
- Produces: `nnUNet_results/Dataset490_LowerLimb/nnUNetTrainerNoMirroring_ES__nnUNetPlans_iso06__3d_fullres/fold_{0..4}/checkpoint_best.pth`.

- [ ] **Step 1: 트레이너 설치 확인 (서버)**

`nnUNetTrainerNoMirroring_ES`가 서버 nnunetv2에 이미 설치되어 있어야 한다(476/481에 사용됨). 확인:
```bash
ssh -i "<key>" ubuntu@114.110.134.100 "source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && python -c 'import nnunetv2.training.nnUNetTrainer.nnUNetTrainerNoMirroring_ES as m; print(m.__file__)'"
```
Expected: 트레이너 .py 경로 출력. (없으면 `ai_bone/nnUNetTrainerNoMirroring_ES.py`를 `.../nnunetv2/training/nnUNetTrainer/`에 복사.)

- [ ] **Step 2: `train_unified.sh` 작성 — 단일 데이터셋 490, 5 fold, GPU 0-4**

```bash
#!/bin/bash
# train_unified.sh — 8×H100에 Dataset490 단일 모델 5-fold 분산 학습 (등방 0.6mm iso06)
# fold 0-4를 GPU 0-4에 1개씩. checkpoint 저장되므로 중단/재개 안전.
cd /data1/bone
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate pt210_py312
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
export nnUNet_n_proc_DA=18
export nnUNet_raw=/data1/bone/ai_bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/ai_bone/nnunet/results

LOG=/data1/bone/train_logs; mkdir -p $LOG
PLANS="-p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES"

run() {  # $1=gpu $2=fold
  echo "[start] GPU$1 Dataset490 fold$2 $(date)" >> $LOG/schedule490.log
  CUDA_VISIBLE_DEVICES=$1 nnUNetv2_train 490 3d_fullres $2 $PLANS \
      > $LOG/d490_f$2_gpu$1.log 2>&1
  echo "[done ] GPU$1 Dataset490 fold$2 $(date)" >> $LOG/schedule490.log
}

( run 0 0 ) & ( run 1 1 ) & ( run 2 2 ) & ( run 3 3 ) & ( run 4 4 ) &
wait
echo "=== Dataset490 전체 학습 완료 $(date) ===" >> $LOG/schedule490.log
```

- [ ] **Step 3: 학습 시작 (서버, 백그라운드)**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "nohup bash /data1/bone/ai_bone/train_unified.sh > /data1/bone/train490_main.log 2>&1 &"
```
Expected: 즉시 반환. 몇 분 뒤 `train_logs/d490_f0_gpu0.log` 등에 epoch 로그 시작, `nvidia-smi`에서 GPU 0-4 사용률 상승.

- [ ] **Step 4: 학습 모니터링 (수렴/과적합 확인)**

```bash
ssh -t -i "<key>" ubuntu@114.110.134.100 "bash /data1/bone/ai_bone/live.sh 15"
```
Expected: fold별 epoch/train loss/val loss/EMA dice 표. 게이트: EMA dice가 상승해 대략 0.9±로 수렴, train/val 괴리가 과하지 않음. ES로 개선 정체 시 조기 종료.

- [ ] **Step 5: 학습 완료 확인**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "ls /data1/bone/ai_bone/nnunet/results/Dataset490_LowerLimb/nnUNetTrainerNoMirroring_ES__nnUNetPlans_iso06__3d_fullres/fold_*/checkpoint_best.pth"
```
Expected: fold_0~fold_4 각각 `checkpoint_best.pth` 존재.

- [ ] **Step 6: Commit**

```bash
git add ai_bone/train_unified.sh
git commit -m "feat: 5-fold training script for unified Dataset490"
```

---

### Task 5: 추론 파이프라인 단일모델화 + merge 제거

**Files:**
- Modify: `ai_bone/infer_mako.py:112-141` (predict/run — 단일 모델 490, 5-fold)
- Modify: `ai_bone/postprocess_mako.py:28-34` (입력을 블록별 단일 pred로)
- Delete: `ai_bone/merge_mako.py`

**Interfaces:**
- Consumes: Task 4 체크포인트(fold 0-4).
- Produces: `pred_490/mako_block{0,1,2}.nii.gz`(21라벨 예측). postprocess는 이를 읽어 `clean_block{n}.nii.gz` 생성.

- [ ] **Step 1: `infer_mako.py`의 `predict`를 단일 모델·5-fold로 수정**

`ai_bone/infer_mako.py:112-124`의 `predict` 함수에서 fold와 dataset 처리를 교체. 기존:
```python
    cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir,
           "-d", str(dataset_id), "-c", "3d_fullres",
           "-p", "nnUNetPlans_iso06", "-tr", "nnUNetTrainerNoMirroring_ES",
           "-f", "0", "1", "2", "--save_probabilities"]
```
를:
```python
    cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir,
           "-d", str(dataset_id), "-c", "3d_fullres",
           "-p", "nnUNetPlans_iso06", "-tr", "nnUNetTrainerNoMirroring_ES",
           "-f", "0", "1", "2", "3", "4", "--save_probabilities"]
```

- [ ] **Step 2: `infer_mako.py`의 `run`을 단일 데이터셋 490으로 수정**

`ai_bone/infer_mako.py:127-141`의 `run`에서 476/481 두 모델 루프를 단일 490으로 교체. 기존:
```python
    # 블록별 NIfTI를 각 모델의 입력 폴더에 저장
    for did, name in [(476, "PelvisThighs"), (481, "ShanksFeet")]:
        in_dir = out / f"in_{did}"; in_dir.mkdir(exist_ok=True)
        for i, blk in enumerate(blocks):
            nii = build_nifti(blk)
            nib.save(nii, str(in_dir / f"mako_block{i}_0000.nii.gz"))
        pred_dir = out / f"pred_{did}"; pred_dir.mkdir(exist_ok=True)
        predict(str(in_dir), str(pred_dir), did)
        print(f"=== d{did} 추론 완료 → {pred_dir}")
    print("infer_mako 완료")
```
를:
```python
    # 블록별 NIfTI를 단일 모델 입력 폴더에 저장 후 490으로 추론
    in_dir = out / "in_490"; in_dir.mkdir(exist_ok=True)
    for i, blk in enumerate(blocks):
        nii = build_nifti(blk)
        nib.save(nii, str(in_dir / f"mako_block{i}_0000.nii.gz"))
    pred_dir = out / "pred_490"; pred_dir.mkdir(exist_ok=True)
    predict(str(in_dir), str(pred_dir), 490)
    print(f"=== d490 추론 완료 → {pred_dir}")
    print("infer_mako 완료")
```

- [ ] **Step 3: docstring 갱신(2모델→단일모델)**

`ai_bone/infer_mako.py`의 상단 docstring에서 "476·481 두 모델" 언급을 단일 490으로 정정한다(2번째 흐름 항목):
```python
  2) 각 블록 NIfTI를 단일 통합모델(490)로 nnUNetv2_predict (5-fold 앙상블)
```
그리고 블록↔모델 매칭 문단은 삭제(더 이상 병합 불필요).

- [ ] **Step 4: `postprocess_mako.py` 입력을 블록별 단일 pred로 변경**

`ai_bone/postprocess_mako.py:28-34`에서 병합 결과(`merged_block{b}`) 대신 추론 결과(`pred_490/mako_block{b}.nii.gz`)를 직접 읽도록 수정. 기존:
```python
        mp = f"{pred_dir}/merged_block{b}.nii.gz"
        cp = f"{pred_dir}/in_476/mako_block{b}_0000.nii.gz"
```
를:
```python
        mp = f"{pred_dir}/pred_490/mako_block{b}.nii.gz"
        cp = f"{pred_dir}/in_490/mako_block{b}_0000.nii.gz"
```
(금속 제거·대상다리 추출·L/R 통합·closing 로직은 그대로 유지. 라벨은 이미 통일 21스킴.)

- [ ] **Step 5: `merge_mako.py` 삭제**

```bash
git rm ai_bone/merge_mako.py
```
(단일 모델은 블록마다 모든 뼈를 알고 있어 부위별 병합이 불필요.)

- [ ] **Step 6: import 스모크 (로컬)**

```bash
cd "C:/Users/정현우/Desktop/VPI LAB/stanford medicine" && "/c/ProgramData/anaconda3/envs/ct_env/python.exe" -c "import ast; ast.parse(open('ai_bone/infer_mako.py',encoding='utf-8').read()); ast.parse(open('ai_bone/postprocess_mako.py',encoding='utf-8').read()); print('parse OK')"
```
Expected: `parse OK` (문법 오류 없음).

- [ ] **Step 7: Commit**

```bash
git add ai_bone/infer_mako.py ai_bone/postprocess_mako.py
git commit -m "refactor: single-model inference (Dataset490), drop 2-model merge"
```

---

### Task 6: Mako 환자 End-to-End 추론·검증

**Files:**
- (실행만; 코드 변경 없음. 필요 시 `ai_bone/viz_mako.py` 색/라벨 확인용 재사용.)

**Interfaces:**
- Consumes: Task 4 체크포인트, Task 5 스크립트.
- Produces: `pred_490/clean_block{0,1,2}.nii.gz` + 렌더 확인 이미지. 수용 기준: 대퇴골이 근위~원위 단일 라벨로 연결, 무릎·발목에서 뼈 간 오분류 없음.

- [ ] **Step 1: 추론 실행 (서버)**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && python ai_bone/infer_mako.py /data1/bone/mako/07049679 /data1/bone/mako/pred_490_07049679"
```
Expected: `block0/1/2` 분할 로그 → `d490 추론 완료`. `pred_490/mako_block{0,1,2}.nii.gz` 생성.

- [ ] **Step 2: 후처리 실행 (서버)**

```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && python ai_bone/postprocess_mako.py /data1/bone/mako/pred_490_07049679"
```
Expected: `block0/1/2: 금속 …k, 반대다리·봉 …k 제거, 연결 +…k` → `후처리 완료`. `clean_block{0,1,2}.nii.gz` 생성.

- [ ] **Step 3: 원래 버그 회귀 검증 — 무릎 대퇴골 단일 라벨 확인**

무릎 블록(대퇴↔경골 경계)에서 대퇴골이 위쪽을 지배하는지 z-구간 voxel로 확인(과거엔 위=Tibia 229k로 역전됐음).
```bash
ssh -i "<key>" ubuntu@114.110.134.100 "cd /data1/bone && source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && python -c \"
import numpy as np, nibabel as nib
a=np.asanyarray(nib.load('/data1/bone/mako/pred_490_07049679/clean_block1.nii.gz').dataobj)
nz=a.shape[2]
for lo,hi,nm in [(2*nz//3,nz,'위(대퇴)'),(0,nz//3,'아래(정강)')]:
    s=a[:,:,lo:hi]; print(nm,'Femur(2)',int((s==2).sum())//1000,'k  Tibia(9)',int((s==9).sum())//1000,'k')
\""
```
Expected: `위(대퇴) Femur ≫ Tibia`, `아래(정강) Tibia ≫ Femur` (과거 역전 해소). 대퇴골 voxel의 최대 연결성분이 전체의 대부분(단일 뼈).

- [ ] **Step 4: 렌더 확인 이미지 생성(약한 스무딩) — 로컬**

`clean_block*.nii.gz`를 로컬로 내려받아(`scp`), 확립된 약한 스무딩(gaussian 없이 marching cubes + 약한 taubin)으로 렌더한다. 신규 `ai_bone/render_clean.py`:
```python
"""render_clean.py — clean_block*.nii.gz를 약한 스무딩으로 3D 렌더(육안 QA).
실행: ct_env python ai_bone/render_clean.py <dir> <out.png>
"""
import sys, numpy as np, nibabel as nib, pyvista as pv, matplotlib.pyplot as plt
pv.OFF_SCREEN = True
d = sys.argv[1] if len(sys.argv) > 1 else "mako_pred"
out = sys.argv[2] if len(sys.argv) > 2 else "mako_pred/clean_490.png"
tab = plt.get_cmap("tab20")
pl = pv.Plotter(off_screen=True, window_size=(900, 1300))
for b in [0, 1, 2]:
    p = f"{d}/clean_block{b}.nii.gz"
    try:
        img = nib.load(p)
    except Exception:
        continue
    a = np.asanyarray(img.dataobj); sp = np.abs(np.diag(img.affine))[:3]; oz = img.affine[2, 3]
    for cid in np.unique(a):
        if cid == 0:
            continue
        m = (a == cid).astype(np.float32)
        if m.sum() < 200:
            continue
        g = pv.ImageData(dimensions=m.shape, spacing=sp, origin=(0, 0, oz))
        g.point_data["v"] = m.ravel(order="F")
        s = g.contour([0.5], scalars="v")          # gaussian 없이 바로 등고선
        if s.n_points == 0:
            continue
        s = s.smooth_taubin(n_iter=12, pass_band=0.1)   # 표면만 약하게
        pl.add_mesh(s, color=tab((int(cid) - 1) % 20)[:3], smooth_shading=True, specular=0.3)
pl.camera_position = "xz"
pl.screenshot(out); pl.close()
print(f"저장: {out}")
```
Run(예):
```bash
cd "C:/Users/정현우/Desktop/VPI LAB/stanford medicine" && ct_env python ai_bone/render_clean.py mako_pred_490 mako_pred_490/clean_490.png
```
그 뒤 Read 도구로 PNG를 열어 육안 확인.
Expected: 대퇴골이 근위~원위 **한 색으로 연결**, 무릎/발목 경계에서 여러 색 쪼개짐·뼈 혼재 없음.

- [ ] **Step 4b: Commit 렌더 스크립트**

```bash
git add ai_bone/render_clean.py
git commit -m "feat: light-smoothing render for QA of unified clean blocks"
```

- [ ] **Step 5: 결과를 메모리에 반영**

`bone_ml_plan.md`에 통합 모델 결과(경계 오분류 해소 여부, dice, 남은 이슈)를 기록하고, "미해결 과제(병합정책 수정)" 항목을 해소 처리.

---

## 완료 정의 (Definition of Done)

- `Dataset490_LowerLimb`(21라벨, ~52 case) 변환·검증 통과.
- 5-fold 학습 완료(checkpoint_best 5개), 수렴·과적합 게이트 통과.
- 단일 모델 추론으로 Mako 환자 `clean_block*.nii.gz` 생성, `merge_mako.py` 제거.
- 회귀 검증: 무릎 대퇴골 z-구간 역전 해소(위=Femur 지배), 렌더에서 대퇴골 단일 색.
- 기존 476/481 자산 보존.
