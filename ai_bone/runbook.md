# 전신 뼈 통합 분할 모델 v1 — 서버 실행 런북

> **대상:** 공유 H100 서버(`114.110.134.100`) 위에서 코드를 실제 실행할 때 참조하는
> 복붙(copy-paste) 가능한 단계별 매뉴얼입니다.
> 코드는 이미 완성되어 있으며, 이 문서의 명령을 순서대로 실행하면 됩니다.

---

## 목차

1. [환경 준비](#1-환경-준비)
2. [데이터 다운로드](#2-데이터-다운로드)
3. [통합 빌드 (nnUNet_raw 조립)](#3-통합-빌드-nnunet_raw-조립)
4. [전처리 (iso 0.6mm 계획 + 전처리)](#4-전처리-iso-06mm-계획--전처리)
5. [Stage 1: CADS 사전학습](#5-stage-1-cads-사전학습)
6. [Stage 2: Joint-pooling Baseline](#6-stage-2-joint-pooling-baseline)
7. [Stage 3: MERIT (Conflict-aware 분할 + 병합)](#7-stage-3-merit-conflict-aware-분할--병합)
8. [평가 및 추론](#8-평가-및-추론)
9. [GPU 나눠쓰기 규약](#9-gpu-나눠쓰기-규약)

---

## 1. 환경 준비

**예상 시간:** 1~2분

### 1-A. conda 환경 활성화

매 세션 시작 시 아래를 실행합니다. 서버 로그인 직후 한 번만 실행하면 됩니다.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
```

### 1-B. nnU-Net 환경변수 설정

```bash
export nnUNet_raw=/data1/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
```

> **주의:** `LD_LIBRARY_PATH` 미설정 시 CXXABI 오류가 발생합니다.
> `nnUNet_compile=f` 미설정 시 nnU-Net이 JIT 컴파일을 시도하다 경합을 일으킬 수 있습니다.
> 이 두 줄은 반드시 포함하십시오.

**반복이 싫다면 `~/.bashrc`에 추가하거나, 아래처럼 한 줄로 실행하세요:**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312 && \
  export nnUNet_raw=/data1/bone/nnunet/raw \
         nnUNet_preprocessed=/home/ubuntu/nnunet_pre \
         nnUNet_results=/data1/bone/nnunet/results \
         LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} \
         nnUNet_compile=f
```

### 1-C. 커스텀 Trainer 파일 서버 복사

nnU-Net은 `-tr <TrainerName>` 인자를 설치된 nnunetv2 패키지 내부에서 찾습니다.
커스텀 trainer 두 파일을 **설치된 nnunetv2 패키지 경로**에 복사해야 합니다.

```bash
# 설치 경로 확인
NNUNET_DIR=$(python -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))")
echo "nnunetv2 설치 경로: $NNUNET_DIR"

# Trainer 복사 (저장소 루트에서 실행)
cp ai_bone/train/partial_label_trainer.py \
   "${NNUNET_DIR}/training/nnUNetTrainer/partial_label_trainer.py"

cp ai_bone/train/merit_finetune_trainer.py \
   "${NNUNET_DIR}/training/nnUNetTrainer/merit_finetune_trainer.py"
```

**검증:**

```bash
python -c "
from nnunetv2.training.nnUNetTrainer.partial_label_trainer import nnUNetTrainerNoMirroring_ES_PL
from nnunetv2.training.nnUNetTrainer.merit_finetune_trainer import nnUNetTrainerMERITFinetune
print('OK: 두 trainer 모두 import 성공')
"
```

이 출력이 나오면 다음 단계로 넘어갑니다:
```
OK: 두 trainer 모두 import 성공
```

---

## 2. 데이터 다운로드

**예상 시간:** 데이터셋 크기와 대역폭에 따라 수 시간~1일

### 2-A. 공통 준비

저장 디렉토리를 만듭니다:

```bash
mkdir -p /data1/bone/raw/{totalseg,verse,ctspine1k,ribseg,ctpelvic1k,spinemets,mug500,cads}
```

### 2-B. 데이터셋별 다운로드

각 다운로드 스크립트는 HTTP Range 재개(resume)를 지원합니다.
중단 후 동일 명령을 재실행하면 이어받습니다.

> **다운로드 CLI (배선됨, GPU 불필요):** `python -m ai_bone.download <name|all> --dest <dir> [--force]`.
> 데이터셋별 소스는 `ai_bone/datasets/sources.py`에 등록돼 있습니다.
> **⚠ 소스 ID 미검증:** 모든 소스가 `verified: False`라 `--force` 없이는 다운로드하지 않고 landing URL만 출력합니다. **먼저 landing URL에서 Zenodo record 번호 등 실제 값을 확인**하고 `sources.py`를 고친 뒤 `--force`로 실행하십시오. `method: "manual"`(CADS=HuggingFace, Spine-Mets=TCIA, CTSpine1K=GDrive, MUG500=Figshare)은 자동 다운로드 대신 안내만 출력합니다 — 각 landing에서 해당 클라이언트로 받으십시오.

**TotalSegmentator v2 (앵커 FT, ~40GB, Zenodo)**

```bash
cd /data1/bone/raw/totalseg
python -m ai_bone.download \
  --zenodo-record 10047292 \
  --out /data1/bone/raw/totalseg \
  --parallel 8
```

> Zenodo는 단일 연결 ~1.6MB/s이므로 `--parallel 8`로 병렬 연결을 씁니다.
> **8을 초과하면 Zenodo 서버가 차단합니다** — 최대 8로 유지하십시오.

**VerSe'19/'20 (~3GB, GitHub release)**

```bash
python -m ai_bone.download \
  --github-url https://github.com/anjany/verse \
  --out /data1/bone/raw/verse
```

**CTSpine1K (~10GB, GitHub release)**

```bash
python -m ai_bone.download \
  --github-url https://github.com/MIRACLE-Center/CTSpine1K \
  --out /data1/bone/raw/ctspine1k
```

**RibSeg v2 (~8GB, Zenodo)**

```bash
python -m ai_bone.download \
  --zenodo-record 7205939 \
  --out /data1/bone/raw/ribseg \
  --parallel 8
```

**CTPelvic1K (~15GB, Zenodo)**

```bash
python -m ai_bone.download \
  --zenodo-record 4588403 \
  --out /data1/bone/raw/ctpelvic1k \
  --parallel 8
```

**Spine-Mets-CT (TCIA — 55케이스)**

TCIA는 NBIA Data Retriever 또는 `tcia_utils` 패키지를 사용합니다:

```bash
pip install tcia_utils
python -c "
from tcia_utils import nbia
nbia.downloadSeries(
    series_data=nbia.getSeries(collection='Spine-Mets-CT-SEG'),
    path='/data1/bone/raw/spinemets'
)
"
```

**MUG500+ (~2GB, Figshare)**

```bash
python -m ai_bone.download \
  --figshare-doi 10.6084/m9.figshare.9616168 \
  --out /data1/bone/raw/mug500
```

**CADS 축골격 subset (HuggingFace, 사전학습 전용)**

```bash
python -m ai_bone.download \
  --huggingface-repo StanfordMIMI/CADS \
  --subset axial_skeleton \
  --out /data1/bone/raw/cads
```

> CADS 라이선스: **CC BY-NC-SA** — 사전학습 전용. 상업 배포 불가.

### 2-C. 다운로드 재개

모든 스크립트가 `--resume`(기본값)을 지원합니다. 중단 시 동일 명령을 다시 실행하면 됩니다.

```bash
# 예시: TotalSegmentator 재개
python -m ai_bone.download \
  --zenodo-record 10047292 \
  --out /data1/bone/raw/totalseg \
  --parallel 8
# 이미 받은 파일은 건너뜁니다.
```

### 2-D. 다운로드 확인 게이트

진행 전 파일 수를 확인합니다:

```bash
for ds in totalseg verse ctspine1k ribseg ctpelvic1k spinemets mug500 cads; do
  cnt=$(find /data1/bone/raw/$ds -name "*.nii.gz" -o -name "*.dcm" 2>/dev/null | wc -l)
  echo "$ds: $cnt 파일"
done
```

예상 파일 수(NIfTI 기준, 검증 후 확정):
- `totalseg`: 이미지+세그 합계 ~2,400+
- `verse`: ~600+
- `ctspine1k`: ~2,000+
- `ribseg`: ~1,320+
- `ctpelvic1k`: ~2,300+
- `spinemets`: ~110+
- `mug500`: ~1,000+
- `cads`: 수천 (서브셋 크기 확인 필요)

---

## 3. 통합 빌드 (nnUNet_raw 조립)

**예상 시간:** 데이터셋 크기에 따라 수 시간 (I/O 병목)

### 3-A. CADS → Dataset500_AxialPretrain (사전학습)

> **빌드 CLI (배선됨, GPU 불필요):** `python -m ai_bone.build_raw --pairs <pairs.json> --dataset <name> --out <raw_dir>`.
> end-to-end로 각 케이스에 `harmonize_case`(remap+정합+iso0.6) → `verify_case` 게이트 → `imagesTr/<id>_0000.nii.gz` + `labelsTr/<id>.nii.gz` + present sidecar → `dataset.json`을 씁니다. 게이트 실패 케이스는 건너뛰고 로그로 남깁니다.
> **⚠ pairs 매니페스트는 사용자가 작성:** `--pairs`는 `[[ct_path, seg_path, case_id], ...]` JSON입니다. **CT 파일 선택 버그 방지를 위해 (한 폴더에 여러 부위 CT가 있을 수 있음) 어떤 CT↔seg가 짝인지 명시적으로 나열**합니다. 다운로드 폴더 구조를 확인한 뒤 데이터셋별로 이 목록을 만드십시오(간단한 glob 스크립트로 생성 가능).

```bash
python -m ai_bone.build_raw \
  --stage pretrain \
  --raw-roots /data1/bone/raw/cads \
  --out $nnUNet_raw/Dataset500_AxialPretrain \
  --spacing 0.6
```

### 3-B. FT 데이터셋 → Dataset510_AxialFT (joint pooling fine-tune)

```bash
python -m ai_bone.build_raw \
  --stage ft \
  --raw-roots \
    /data1/bone/raw/totalseg \
    /data1/bone/raw/verse \
    /data1/bone/raw/ctspine1k \
    /data1/bone/raw/ribseg \
    /data1/bone/raw/ctpelvic1k \
    /data1/bone/raw/spinemets \
    /data1/bone/raw/mug500 \
  --out $nnUNet_raw/Dataset510_AxialFT \
  --spacing 0.6
```

> **주의:** `build_raw`는 각 케이스에 대해
> `remap_array` → `align_geometry` → `resample_to_isotropic` 순으로 적용하고,
> 케이스별 `present_labels` 사이드카(`.present.json`)를 함께 저장합니다.
> label_map.json의 `notes` 항목에 "다운로드 후 확정"이라고 표기된 원본 라벨값은
> 다운로드 후 실제 데이터를 확인하고 필요 시 `ai_bone/datasets/<name>/label_map.json`을 수정하십시오.

### 3-C. verify_dataset 게이트 — 통과 필수

```bash
python -m ai_bone.verify_dataset \
  --dataset-dir $nnUNet_raw/Dataset500_AxialPretrain \
  --report /data1/bone/verify_500.json

python -m ai_bone.verify_dataset \
  --dataset-dir $nnUNet_raw/Dataset510_AxialFT \
  --report /data1/bone/verify_510.json
```

출력 예시 (통과 시):
```
Dataset500: 2847 cases checked. PASS (0 empty, 0 size_mismatch, 0 low_overlap)
Dataset510: 4198 cases checked. PASS (0 empty, 0 size_mismatch, 0 low_overlap)
```

**실패 케이스가 있을 경우 중단하고 원인 조사:**

```bash
# 실패 케이스 목록 확인
python -c "
import json
r = json.load(open('/data1/bone/verify_510.json'))
fails = [c for c,v in r['cases'].items() if not v['pass']]
print(f'FAIL {len(fails)} cases:')
for f in fails[:20]: print(' ', f, r['cases'][f])
"
```

실패 원인 대부분은 좌우 매핑 오류(RibSeg `Rib_L_*`/`Rib_R_*` 좌우 번호 규칙) 또는
겹침률(overlap_ratio < 0.5 — CT와 세그 파일 불일치)입니다.
`label_map.json` 수정 후 해당 케이스만 재빌드할 수 있습니다.

---

## 4. 전처리 (iso 0.6mm 계획 + 전처리)

**예상 시간:** Dataset500 ~2시간, Dataset510 ~4~8시간 (CPU 병렬)

### 4-A. 계획(plan) 생성

```bash
nnUNetv2_plan_experiment \
  -d 500 510 \
  -overwrite_target_spacing 0.6 0.6 0.6 \
  -overwrite_plans_name nnUNetPlans_iso06
```

> `-d 500 510`으로 두 데이터셋을 한 번에 계획합니다.

확인:

```bash
ls $nnUNet_raw/Dataset500_AxialPretrain/nnUNetPlans_iso06.json
ls $nnUNet_raw/Dataset510_AxialFT/nnUNetPlans_iso06.json
```

### 4-B. 전처리 실행 (Lustre에서 계산 후 SSD 캐시로 복사)

전처리는 Lustre(`$nnUNet_raw` 위치)에서 실행하되,
학습 I/O는 **로컬 SSD(`/home/ubuntu/nnunet_pre`)**에서 이루어집니다.

```bash
# 전처리 실행 (num_processes는 가용 CPU 코어 수에 맞게)
nnUNetv2_preprocess \
  -d 500 510 \
  -plans_name nnUNetPlans_iso06 \
  -c 3d_fullres \
  --verify_dataset_integrity \
  -np 16
```

전처리 완료 후 SSD로 복사:

```bash
mkdir -p /home/ubuntu/nnunet_pre
rsync -av --progress \
  $nnUNet_raw/../nnunet_pre/ \
  /home/ubuntu/nnunet_pre/
```

> **Lustre `/dev/shm` 불안정 교훈:** 과거 `/dev/shm`을 캐시로 썼을 때 I/O 에러가 발생한 이력이 있습니다.
> 반드시 `/home/ubuntu/nnunet_pre`(로컬 SSD)를 사용하십시오.

### 4-C. 전처리 검증 게이트

```bash
# 전처리된 케이스 수 확인
ls /home/ubuntu/nnunet_pre/Dataset500_AxialPretrain/nnUNetPlans_iso06/3d_fullres/ | wc -l
ls /home/ubuntu/nnunet_pre/Dataset510_AxialFT/nnUNetPlans_iso06/3d_fullres/ | wc -l
```

예상값은 §3-C에서 확인된 케이스 수와 일치해야 합니다.

```bash
# 전처리 파일 손상 여부 확인
python -c "
import os, pickle, glob
base = '/home/ubuntu/nnunet_pre/Dataset510_AxialFT/nnUNetPlans_iso06/3d_fullres'
files = glob.glob(os.path.join(base, '*.pkl'))
errors = []
for f in files:
    try: pickle.load(open(f,'rb'))
    except Exception as e: errors.append((f, str(e)))
print(f'Checked {len(files)} pkl files. Errors: {len(errors)}')
for e in errors[:5]: print(' ', e)
"
```

---

## 5. Stage 1: CADS 사전학습

**예상 시간:** H100 1장 기준 24~72시간 (CADS 규모에 따라)

### 5-A. 실행

```bash
# GPU 0번 사용
bash ai_bone/train/stage1_pretrain.sh 0
```

스크립트 내용 요약:
- Dataset 500, fold "all" (전체 데이터 학습)
- trainer: `nnUNetTrainerNoMirroring_ES_PL`
- plans: `nnUNetPlans_iso06`
- `--c` 플래그 포함 → checkpoint 자동 재개

### 5-B. 진행 상황 모니터링

```bash
# 학습 로그 실시간 확인
tail -f $nnUNet_results/Dataset500_AxialPretrain/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/\
fold_all/training_log_*.txt
```

또는:

```bash
# 마지막 epoch 확인
grep "Epoch " $nnUNet_results/Dataset500_AxialPretrain/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/\
fold_all/training_log_*.txt | tail -5
```

### 5-C. 중단 후 재개

스크립트에 `--c`가 이미 포함되어 있습니다. **동일 명령을 그냥 다시 실행하면 됩니다:**

```bash
bash ai_bone/train/stage1_pretrain.sh 0
# checkpoint_latest.pth에서 자동으로 이어받습니다.
```

### 5-D. 완료 검증 게이트

```bash
ls -lh $nnUNet_results/Dataset500_AxialPretrain/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/\
fold_all/checkpoint_final.pth
```

이 파일이 존재하면 Stage 1 완료입니다.
파일이 없고 `checkpoint_latest.pth`만 있으면 학습이 아직 진행 중이거나 중단된 것입니다.

---

## 6. Stage 2: Joint-pooling Baseline

**예상 시간:** fold당 H100 1장 기준 36~96시간, 5 fold = 총 180~480시간
(GPU가 여러 장이면 fold를 병렬 실행 가능)

### 6-A. 실행 (fold × GPU 조합)

각 fold는 **독립적으로 병렬 실행 가능**합니다.

```bash
# GPU가 1장뿐일 때: fold 순차 실행
for FOLD in 0 1 2 3 4; do
  bash ai_bone/train/stage2_baseline.sh $FOLD 0
done
```

```bash
# GPU가 여러 장일 때: fold를 GPU에 분산 (예: GPU 0~4)
for FOLD in 0 1 2 3 4; do
  bash ai_bone/train/stage2_baseline.sh $FOLD $FOLD &
done
wait
```

```bash
# GPU 2장만 여유일 때: fold 2개씩 순차 실행 (GPU 2, 3번)
bash ai_bone/train/stage2_baseline.sh 0 2 &
bash ai_bone/train/stage2_baseline.sh 1 3 &
wait
bash ai_bone/train/stage2_baseline.sh 2 2 &
bash ai_bone/train/stage2_baseline.sh 3 3 &
wait
bash ai_bone/train/stage2_baseline.sh 4 2
```

### 6-B. 사전 가중치 경로 확인

스크립트가 Stage 1 checkpoint를 자동으로 참조합니다:
```
-pretrained_weights /data1/bone/nnunet/results/Dataset500_AxialPretrain/*_all/checkpoint_final.pth
```

glob(`*_all`)이 실패하면 수동으로 경로를 지정하십시오:

```bash
# 실제 경로 확인
ls $nnUNet_results/Dataset500_AxialPretrain/*/fold_all/checkpoint_final.pth
```

### 6-C. 중단 후 재개

동일 명령 재실행으로 자동 재개됩니다 (`--c` 내장).

### 6-D. 완료 검증 게이트

```bash
for FOLD in 0 1 2 3 4; do
  f="$nnUNet_results/Dataset510_AxialFT/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/\
fold_${FOLD}/checkpoint_final.pth"
  [ -f "$f" ] && echo "fold $FOLD: OK" || echo "fold $FOLD: MISSING"
done
```

5개 fold 모두 OK이면 Stage 2 완료입니다.

---

## 7. Stage 3: MERIT (Conflict-aware 분할 + 병합)

**예상 시간:**
- Step A (gradient 추출): GPU당 ~1~2시간
- Step B (PCA 분할): CPU, 수분
- Step C (파티션별 학습): fold당 GPU 1장 기준 36~96시간
- Step D (병합): CPU, 수분

> **검증 게이트 권고:** MERIT은 K=2, 소규모(2~3 데이터셋)로 먼저 검증 후 전체로 확대하십시오.

### Step A. Gradient Conflict 측정

Stage 2 baseline checkpoint(또는 Stage 1 shared init)를 기준으로
각 FT 데이터셋의 gradient 방향을 추출합니다.

> **⚠ 서버 스텁:** 실제 gradient 추출 로직(nnU-Net predictor 연결 부분)은 서버 nnunetv2 설치 환경에서 확정 후 실행하십시오.

```bash
python -m ai_bone.merit.estimate_conflict \
  --init $nnUNet_results/Dataset500_AxialPretrain/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/fold_all/checkpoint_final.pth \
  --raw $nnUNet_raw \
  --datasets totalseg,verse,ctspine1k,ribseg,ctpelvic1k,spinemets,mug500 \
  --out /data1/bone/merit/g_vectors.npz \
  --dim 512 \
  --batches 8
```

> **GPU 필요:** `estimate_conflict.py`는 서버 GPU에서 실행합니다.
> `--batches 8`은 각 데이터셋에서 8 mini-batch의 gradient를 평균합니다.

완료 확인:

```bash
python -c "
import numpy as np
g = np.load('/data1/bone/merit/g_vectors.npz')
print('데이터셋별 gradient 벡터:')
for k in g.files: print(f'  {k}: shape={g[k].shape}')
"
```

### Step B. PCA 충돌 분할 (partitions.json 생성)

```bash
python -m ai_bone.merit.split \
  --vectors /data1/bone/merit/g_vectors.npz \
  --k 2 \
  --out /data1/bone/merit/partitions.json
```

결과 확인:

```bash
python -c "
import json
p = json.load(open('/data1/bone/merit/partitions.json'))
for pid, datasets in p.items():
    print(f'Partition {pid}: {datasets}')
"
```

예상 출력 (K=2):
```
Partition 0: ['totalseg', 'ribseg', 'mug500']
Partition 1: ['verse', 'ctspine1k', 'ctpelvic1k', 'spinemets']
```
(실제 분할은 gradient 방향에 따라 달라집니다.)

### Step C. 파티션별 Dataset 빌드 + 전처리

`partitions.json`에서 각 파티션의 데이터셋을 추출해 Dataset520, Dataset521 등을 빌드합니다.

```bash
# partitions.json을 읽어 각 파티션의 raw-roots 조합
python -c "
import json, subprocess, sys

p = json.load(open('/data1/bone/merit/partitions.json'))
raw_root = '/data1/bone/raw'
nnunet_raw = '/data1/bone/nnunet/raw'

for pid, datasets in p.items():
    did = 520 + int(pid)
    raw_roots = ' '.join(f'{raw_root}/{ds}' for ds in datasets)
    out_dir = f'{nnunet_raw}/Dataset{did}_MeritPart{pid}'
    cmd = (
        f'python -m ai_bone.build_raw --stage ft '
        f'--raw-roots {raw_roots} '
        f'--out {out_dir} --spacing 0.6'
    )
    print(f'[Partition {pid} → Dataset{did}]')
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
"
```

파티션 Dataset 전처리:

```bash
# partitions.json의 dataset id 목록 (예: 520 521)
nnUNetv2_plan_experiment \
  -d 520 521 \
  -overwrite_target_spacing 0.6 0.6 0.6 \
  -overwrite_plans_name nnUNetPlans_iso06

nnUNetv2_preprocess \
  -d 520 521 \
  -plans_name nnUNetPlans_iso06 \
  -c 3d_fullres \
  -np 16

# SSD 캐시 동기화
rsync -av /data1/bone/nnunet/pre/ /home/ubuntu/nnunet_pre/
```

verify_dataset 게이트:

```bash
python -m ai_bone.verify_dataset --dataset-dir $nnUNet_raw/Dataset520_MeritPart0 --report /data1/bone/verify_520.json
python -m ai_bone.verify_dataset --dataset-dir $nnUNet_raw/Dataset521_MeritPart1 --report /data1/bone/verify_521.json
```

### Step D. 파티션별 Fine-tune

각 파티션과 fold는 독립적으로 실행 가능합니다.

```bash
# Partition 0 (Dataset 520), fold 0, GPU 0
bash ai_bone/merit/merit_train_partition.sh 520 0 0

# Partition 1 (Dataset 521), fold 0, GPU 1 (병렬 가능)
bash ai_bone/merit/merit_train_partition.sh 521 0 1
```

전체 fold:

```bash
# GPU 여유에 맞게 분산 (예: GPU 0~3)
bash ai_bone/merit/merit_train_partition.sh 520 0 0 &
bash ai_bone/merit/merit_train_partition.sh 520 1 1 &
bash ai_bone/merit/merit_train_partition.sh 521 0 2 &
bash ai_bone/merit/merit_train_partition.sh 521 1 3 &
wait
# ... fold 2~4 계속
```

재개: 동일 명령 재실행 (`--c` 내장).

완료 확인:

```bash
for DID in 520 521; do
  for FOLD in 0 1 2 3 4; do
    f="$nnUNet_results/Dataset${DID}_MeritPart*/\
nnUNetTrainerMERITFinetune__nnUNetPlans_iso06__3d_fullres/\
fold_${FOLD}/checkpoint_final.pth"
    ls $f 2>/dev/null && echo "D${DID} fold${FOLD}: OK" || echo "D${DID} fold${FOLD}: MISSING"
  done
done
```

### Step E. Weight 병합

#### Primary: weighted_average (케이스 수 가중)

```bash
python -m ai_bone.merit.merge \
  --mode weighted_average \
  --checkpoints \
    $nnUNet_results/Dataset520_MeritPart0/nnUNetTrainerMERITFinetune__nnUNetPlans_iso06__3d_fullres/fold_0/checkpoint_final.pth \
    $nnUNet_results/Dataset521_MeritPart1/nnUNetTrainerMERITFinetune__nnUNetPlans_iso06__3d_fullres/fold_0/checkpoint_final.pth \
  --weights 0.5 0.5 \
  --out /data1/bone/merit/merged_model_fold0.pth
```

> 가중치 `--weights`는 각 파티션의 케이스 수 비율로 설정하십시오.
> 예: 파티션 0이 2,400케이스, 파티션 1이 1,800케이스 → `--weights 0.571 0.429`

#### Ablation: TIES merge

```bash
python -m ai_bone.merit.merge \
  --mode ties_merge \
  --base $nnUNet_results/Dataset500_AxialPretrain/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/fold_all/checkpoint_final.pth \
  --checkpoints \
    $nnUNet_results/Dataset520_MeritPart0/nnUNetTrainerMERITFinetune__nnUNetPlans_iso06__3d_fullres/fold_0/checkpoint_final.pth \
    $nnUNet_results/Dataset521_MeritPart1/nnUNetTrainerMERITFinetune__nnUNetPlans_iso06__3d_fullres/fold_0/checkpoint_final.pth \
  --weights 0.5 0.5 \
  --density 0.2 \
  --out /data1/bone/merit/merged_model_ties_fold0.pth
```

---

## 8. 평가 및 추론

**예상 시간:** 홀드아웃 세트 크기에 따라 수 시간

### 8-A. Baseline A vs MERIT B 정량 비교

홀드아웃 세트에 대해 두 모델의 예측을 비교합니다.

```bash
# Baseline A (Stage 2 joint pooling) 추론
nnUNetv2_predict \
  -i /data1/bone/holdout/images \
  -o /data1/bone/predictions/baseline_a \
  -d 510 -c 3d_fullres -f 0 1 2 3 4 \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL \
  --save_probabilities

# MERIT B (병합 모델) 추론
nnUNetv2_predict \
  -i /data1/bone/holdout/images \
  -o /data1/bone/predictions/merit_b \
  -d 510 -c 3d_fullres -f 0 \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL \
  --disable_tta \
  -chk /data1/bone/merit/merged_model_fold0.pth
```

### 8-B. Dice + HD95 평가 리포트

```bash
python -m ai_bone.eval.evaluate \
  --gt-dir /data1/bone/holdout/labels \
  --pred-dir /data1/bone/predictions/baseline_a \
  --spacing 0.6 \
  --report /data1/bone/eval/report_baseline_a.json

python -m ai_bone.eval.evaluate \
  --gt-dir /data1/bone/holdout/labels \
  --pred-dir /data1/bone/predictions/merit_b \
  --spacing 0.6 \
  --report /data1/bone/eval/report_merit_b.json
```

비교 요약 출력:

```bash
python -c "
import json
a = json.load(open('/data1/bone/eval/report_baseline_a.json'))
b = json.load(open('/data1/bone/eval/report_merit_b.json'))
print(f'{'Label':<15} {'Baseline_A Dice':>18} {'MERIT_B Dice':>14} {'Delta':>8}')
print('-'*60)
for name in a:
    da = a[name]['dice']; db = b[name]['dice']
    if da != da or db != db: continue  # nan skip
    print(f'{name:<15} {da:>18.4f} {db:>14.4f} {db-da:>+8.4f}')
"
```

### 8-C. 홀드아웃 도메인별 분석

병리(Spine-Mets) vs 정상 케이스 분리 비교:

```bash
python -m ai_bone.eval.evaluate \
  --gt-dir /data1/bone/holdout/spinemets/labels \
  --pred-dir /data1/bone/predictions/baseline_a/spinemets \
  --spacing 0.6 \
  --report /data1/bone/eval/report_baseline_a_spinemets.json
```

### 8-D. Mako 데이터 정성 QC

사내 Mako CT 데이터에 추론 후 오버레이 PNG로 시각적 QC:

```bash
# Mako 추론
python ai_bone/infer_mako.py \
  --model-dir $nnUNet_results/Dataset510_AxialFT/\
nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres \
  --input /data1/bone/mako/raw \
  --output /data1/bone/mako/predictions

# QC 오버레이 생성
python ai_bone/qc_overlay.py \
  --ct-dir /data1/bone/mako/raw \
  --seg-dir /data1/bone/mako/predictions \
  --out-dir /data1/bone/mako/qc_png
```

---

## 9. GPU 나눠쓰기 규약

### 기본 원칙

- **각 train 스크립트 = GPU 1장**. `CUDA_VISIBLE_DEVICES` 인자로 지정.
- **동시 실행 수 = 가용 GPU 수**. 초과하면 학습이 느려지고 다른 사용자 작업에 영향을 줍니다.
- **모든 스크립트에 `--c` 포함** → 중단 후 동일 명령 재실행으로 자동 재개.

### GPU 가용 확인

실행 전 반드시 확인합니다:

```bash
nvidia-smi
```

`Volatile GPU-Util`이 낮은(~0%) GPU만 사용합니다.

```bash
# 여유 GPU 목록 추출 (Util < 5% 기준)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
  | awk -F',' '$2+0 < 5 {print "GPU " $1 ": " $2 " util, " $3 " mem used"}'
```

### 실행 예시 (GPU 2장 여유: 4, 6번)

```bash
# fold 0 → GPU 4, fold 1 → GPU 6 병렬 실행
bash ai_bone/train/stage2_baseline.sh 0 4 &
bash ai_bone/train/stage2_baseline.sh 1 6 &
wait
echo "두 fold 완료"
```

### 중단 시 절차

1. `Ctrl+C` 또는 프로세스 종료.
2. `checkpoint_latest.pth`는 자동으로 저장되어 있습니다.
3. 나중에 동일 명령을 재실행하면 됩니다 (`--c` 덕분에 자동 재개).

```bash
# 예: 중단된 fold 2 재개
bash ai_bone/train/stage2_baseline.sh 2 0
```

### 배경 실행 및 로그 관리

nohup으로 SSH 세션 종료 후에도 계속 실행:

```bash
nohup bash ai_bone/train/stage1_pretrain.sh 0 \
  > /data1/bone/logs/stage1_gpu0.log 2>&1 &
echo "PID: $!"
```

진행 확인:

```bash
tail -f /data1/bone/logs/stage1_gpu0.log
```

### 스크립트 요약표

| 스크립트 | 인자 | GPU 수 | 재개 |
|---|---|---|---|
| `ai_bone/train/stage1_pretrain.sh <GPU>` | GPU ID | 1 | --c 자동 |
| `ai_bone/train/stage2_baseline.sh <FOLD> <GPU>` | fold(0-4), GPU ID | 1 | --c 자동 |
| `ai_bone/merit/merit_train_partition.sh <DID> <FOLD> <GPU>` | 데이터셋ID(52x), fold, GPU | 1 | --c 자동 |

---

## 부록 A. 경로 요약

| 항목 | 경로 |
|---|---|
| 작업 루트 | `/data1/bone` |
| 원본 데이터 | `/data1/bone/raw/<dataset_name>/` |
| nnUNet raw | `/data1/bone/nnunet/raw/` (= `$nnUNet_raw`) |
| 전처리 캐시(SSD) | `/home/ubuntu/nnunet_pre/` (= `$nnUNet_preprocessed`) |
| 학습 결과 | `/data1/bone/nnunet/results/` (= `$nnUNet_results`) |
| MERIT 산출물 | `/data1/bone/merit/` |
| 평가 리포트 | `/data1/bone/eval/` |
| 로그 | `/data1/bone/logs/` |

## 부록 B. 데이터셋 ID 매핑

| Dataset ID | 이름 | 용도 |
|---|---|---|
| 500 | `Dataset500_AxialPretrain` | CADS 사전학습 |
| 510 | `Dataset510_AxialFT` | joint pooling FT |
| 520 | `Dataset520_MeritPart0` | MERIT 파티션 0 |
| 521 | `Dataset521_MeritPart1` | MERIT 파티션 1 |

> 파티션 수(K=2 → 52x 2개)는 `partitions.json` 결과에 따라 조정 가능합니다.
> K=3 확장 시 Dataset522 추가.

## 부록 C. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `CXXABI_1.3.13 not found` | LD_LIBRARY_PATH 미설정 | §1-B 환경변수 재설정 |
| `-tr nnUNetTrainerNoMirroring_ES_PL not found` | Trainer 파일 미복사 | §1-C 복사 재실행 |
| `overlap_ratio < 0.5` (verify 실패) | label_map 좌우 오류 또는 CT 파일 불일치 | `label_map.json` 수정 후 해당 케이스 재빌드 |
| 학습 Loss=NaN | LR 너무 크거나 ignore_label 처리 오류 | checkpoint_latest에서 재개; LR 확인 |
| `checkpoint_final.pth` 없음 | 학습 미완료 또는 ES 미발동 | 로그 확인 후 --c 재실행 |
| Zenodo 다운로드 느림 | 단일 연결 ~1.6MB/s | `--parallel 8` (8 초과 금지) |

---

## 10. 풀 학습 (2 GPU, 최대 가동)

시간이 더 걸려도 전 fold·전 데이터셋을 돌리고 GPU를 거의 풀로 쓰고 싶을 때. 의존성 때문에 **3단계**로 나눠 실행하되, 각 단계에서 2 GPU를 최대한 채운다.

### 학습할 잡 목록 (풀)
- 사전학습 1개: `Dataset500` fold `all`
- baseline 5개: `Dataset510` fold 0–4
- MERIT: 파티션(520/521…) × fold 0–4  (split 결과에 따라 K개)

### Phase 1 — 사전학습: 한 모델에 2 GPU (DDP)
단일 모델이라 큐로 나눌 수 없으니 **nnU-Net DDP로 2장을 한 모델에** 투입 → 두 GPU 모두 가동.
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/bone/nnunet/raw nnUNet_preprocessed=/home/ubuntu/nnunet_pre \
       nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH nnUNet_compile=f nnUNet_n_proc_DA=32
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train 500 3d_fullres all \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL -num_gpus 2 --c
```

### Phase 2 — baseline 5 fold: 2 GPU 잡 큐
`jobs_baseline.txt` (PRETRAINED = Phase1 결과 checkpoint):
```
510 3d_fullres 0 nnUNetTrainerNoMirroring_ES_PL /data1/bone/nnunet/results/Dataset500_AxialPretrain/nnUNetTrainerNoMirroring_ES_PL__nnUNetPlans_iso06__3d_fullres/fold_all/checkpoint_final.pth
510 3d_fullres 1 nnUNetTrainerNoMirroring_ES_PL <같은 checkpoint>
510 3d_fullres 2 nnUNetTrainerNoMirroring_ES_PL <같은 checkpoint>
510 3d_fullres 3 nnUNetTrainerNoMirroring_ES_PL <같은 checkpoint>
510 3d_fullres 4 nnUNetTrainerNoMirroring_ES_PL <같은 checkpoint>
```
```bash
nohup bash ai_bone/train/run_queue.sh "0,1" jobs_baseline.txt > /data1/bone/train_logs/queue_baseline.log 2>&1 &
```
→ fold를 GPU 0/1에 라운드로빈(3+2)으로 순차 실행, 각 GPU는 한 번에 한 모델(=풀 가동), `--c` 재개.

### Phase 3 — MERIT: split 후 파티션×fold 큐
`estimate_conflict.py`→`split.py`로 `partitions.json`을 만들고, 파티션별로 `build_raw`로 `Dataset520/521…`을 만든 뒤, `jobs_merit.txt`에 `520/521 … fold0-4 nnUNetTrainerMERITFinetune <checkpoint>`를 넣어 동일하게:
```bash
nohup bash ai_bone/train/run_queue.sh "0,1" jobs_merit.txt > /data1/bone/train_logs/queue_merit.log 2>&1 &
```

### GPU 이용률(%)을 더 끌어올리는 레버
H100은 nnU-Net 기본 patch/batch에 여유가 남습니다. 시간이 되면:
1. **큰 batch/patch로 재plan**(VRAM·연산 더 사용) — 새 plans로:
   ```bash
   nnUNetv2_plan_experiment -d 500 510 -overwrite_target_spacing 0.6 0.6 0.6 \
     -gpu_memory_target 70 -overwrite_plans_name nnUNetPlans_iso06_big
   nnUNetv2_preprocess -d 500 510 -plans_name nnUNetPlans_iso06_big -c 3d_fullres -np 32
   ```
   이후 스크립트/큐의 `-p`를 `nnUNetPlans_iso06_big`으로 바꿔 사용.
2. **`nnUNet_n_proc_DA` 상향**(기본 24) — GPU가 augmentation 대기로 굶주리면 32~48까지. 124코어라 2 job이면 여유.
3. **완전 학습(ES 미발동)** — 조기종료 없이 1000 epoch를 다 돌리려면 ES 완화가 필요. `nnUNetTrainerNoMirroring_ES_PL`은 `es_min_epochs=200`/`es_patience=75`이므로, 끝까지 돌리려면 커스텀 trainer에서 `es_patience`를 크게(예: 10^9) 두거나 `num_epochs`만 쓰는 non-ES trainer를 만들면 됩니다.

### 시간 감
epoch은 데이터 크기와 무관(250 iter 고정). 잡 1개 ES 수렴 ~12–24h 가정 시, 2 GPU로 baseline 5개≈3라운드, MERIT(K=2)10개≈5라운드 → 대략 **1–2주**면 풀 세트 완주. `nvidia-smi`로 두 GPU가 계속 차 있는지 확인.
