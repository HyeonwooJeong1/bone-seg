# 통합 단일 하지 뼈 분할 모델 — 설계 스펙

- 날짜: 2026-07-03
- 상태: 승인됨 (구현 대기)
- 목적: 부위별 2모델(476 골반-대퇴 / 481 하퇴-발) + 병합 구조를 **하나의 21라벨 단일 nnU-Net 모델**로 대체하여, 무릎·발목 경계에서의 뼈 오분류를 근본 해결한다.

## 1. 배경과 문제

Mako 로봇수술 계획용 하지(고관절~발) CT에서 각 뼈를 3D 딥러닝으로 분할한다. 기존 구조는 VSD 데이터가 부위별 두 스캔(Pelvis-Thighs / Shanks-Feet)으로 제공되어, 두 개의 nnU-Net 모델(`Dataset476`, `Dataset481`)을 따로 학습하고 추론 결과를 `merge_mako.py`로 합쳤다.

**결함:** 무릎 블록은 대퇴골 원위 + 경골 근위가 만나는 경계다. 481(하퇴-발) 모델은 **대퇴골을 학습한 적이 없어**, 무릎의 대퇴골 골간을 경골(Tibia)로 오분류했다. 병합 시 481의 Tibia가 476의 Femur를 덮어써서, 대퇴골이 두 라벨(Femur+Tibia)로 쪼개져 렌더링에서 여러 색으로 보였다.

측정 근거 (무릎 블록 clean, z구간별 voxel):
```
위(대퇴영역):  Femur 28k,  Tibia 229k  ← 대퇴골 골간이 경골로 오분류
중간(관절):    Femur 345k, Tibia 316k
아래(정강):    Femur 3k,   Tibia 246k
```
476 단독은 대퇴골을 완벽히 잡았다(Femur 위 412k 포함, 총 1170k). 즉 **모델 자체가 아니라 부위별 분리 학습 + 병합 구조가 문제**다.

## 2. 목표 / 비목표

**목표**
- 21라벨 전신 하지 뼈를 하나의 모델로 분할.
- 무릎(대퇴↔경골)·발목(경골↔발) 경계를 데이터로 직접 학습 → 병합 로직 제거.
- 최고 정확도 (등방 0.6mm, 5-fold 앙상블).

**비목표**
- 발뼈 개별화(중족골·발가락 낱개 분리): 공개 데이터에 개별 발뼈 라벨이 없어 **묶음 유지**. (TotalSegmentator도 tarsal/metatarsal/phalanges 묶음.)
- 두 스캔을 물리적으로 이어붙인 전신 볼륨 생성(접근 B): 불필요·오류 소지로 제외.

## 3. 데이터

- 출처: VSDFullBody (Zenodo 8302449), cadaver 하지 CT, 26명, CC BY-NC-SA(연구용).
- 각 환자당 부위 스캔 2개:
  - **Pelvis-Thighs**: Sacrum, Hip, Femur, Patella, Tibia, Fibula (좌우)
  - **Shanks-Feet**: Tibia, Fibula, Talus, Calcaneus, Tarsals, Metatarsals, Phalanges (좌우)
- **FOV 검증(016)**: Pelvis-Thighs z 467~1101mm, Shanks-Feet z 34~467mm — 무릎 부근(467mm)에서 **깔끔히 맞닿음, 겹침·누락 없음**. 대퇴골은 전부 Pelvis-Thighs 스캔에서 라벨링되므로, Shanks-Feet 스캔에 **라벨 안 된 대퇴골이 없다** → 통합 학습 시 "배경으로 새는 뼈" 없음. 무릎(대퇴↔경골 전환)은 Pelvis-Thighs 스캔에 온전히 포함되어 단일 모델이 경계를 학습한다.

## 4. 설계

### 4.1 통합 데이터셋 (`Dataset490_LowerLimb`)
- 신규 스크립트 `ai_bone/build_unified.py` (기존 `convert_to_nnunet.py` 확장).
- 각 환자의 **두 스캔을 모두 개별 case**로 변환: `LL_{subj}_PT_0000.nii.gz`, `LL_{subj}_SF_0000.nii.gz` → **약 52 case**.
- 각 스캔의 seg segment **이름 기반**으로 통일 21라벨 id에 remap (부위마다 원본 LabelValue가 달라 이름 기반이 안전).
- CT-라벨 정합: seg와 CT size 같으면 `out.CopyInformation(ct)`, 다르면 `CopyInformation(seg)` 후 물리좌표 `sitkNearestNeighbor` resample (기존 검증된 로직 유지 — 물리좌표 resample을 size 동일 케이스에 쓰면 flip 메타로 라벨 전멸하므로 분기 필수).
- `dataset.json`: 21 labels + background, `numTraining≈52`, channel CT.

### 4.2 라벨 스킴 (21, 기존 UNI 그대로)
```
1 Femur_L    2 Femur_R    3 Hip_L    4 Hip_R    5 Sacrum
6 Patella_L  7 Patella_R  8 Tibia_L  9 Tibia_R  10 Fibula_L  11 Fibula_R
12 Talus_L   13 Talus_R   14 Calcaneus_L 15 Calcaneus_R 16 Tarsals_L 17 Tarsals_R
18 Metatarsals_L 19 Metatarsals_R  20 Phalanges_L 21 Phalanges_R
```
PT 스캔은 1~11 중 존재 라벨, SF 스캔은 8~21 중 존재 라벨만 채워진다.

### 4.3 학습 설정
- 전처리: `nnUNetv2_plan_and_preprocess -d 490` + 커스텀 **등방 0.6mm 플랜**(`nnUNetPlans_iso06`), configuration `3d_fullres`.
- 트레이너: **`nnUNetTrainerNoMirroring_ES`** (L/R 라벨 → 좌우 미러링 증강 금지, + early stopping: es_patience=75, es_min_epochs=200). `__init__`는 명시적 시그니처(plans, configuration, fold, dataset_json, device) 유지(KeyError 방지).
- **5-fold 전부 학습**, 추론 시 5-fold 앙상블. H100 GPU 0~4에 fold 병렬.
- 환경 export: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, `nnUNet_compile=f`, `nnUNet_n_proc_DA=18`, 전처리 데이터는 로컬 SSD.

### 4.4 추론 파이프라인 단순화
- **`merge_mako.py` 제거** (2모델 병합·부위 담당·우선순위 전부 불필요).
- `infer_mako.py`: DICOM → axial('Mako') 선택 → z-gap 3블록 분할 → NIfTI 까지 동일. 각 블록을 **단일 모델 `nnUNetv2_predict -d 490 -f 0 1 2 3 4 --save_probabilities`**로 추론 → 물리 z로 배치.
- `postprocess_mako.py`: 금속(HU≥1900) 제거 + 대상다리 dilate-CC 추출 **유지**. per-bone closing 유지. L/R 통합은 **안전장치로 보존**하되(단일 다리라 L/R 이름 무의미), 통합 모델이 좌우를 일관되게 내면 제거 검토(결과 보고 판단).
- 렌더링: **약한 스무딩** — 라벨 mask → `contour([0.5])` (gaussian 없음) → `smooth_taubin(n_iter≈12, pass_band≈0.1)`. 과한 gaussian+강taubin은 얇은 뼈를 끊으므로 금지.

### 4.5 검증
- `verify_dataset.py`를 `Dataset490`에 실행: 빈 라벨 / 기하 정합 / CT-뼈 overlap / 라벨값 범위 / 21라벨 분포.
- 학습 중: fold별 dice·loss·GPU 모니터(`monitor.py`/`live.sh`). 과적합 체크(train vs val).
- 학습 후: 무릎(대퇴↔경골)·발목 경계 dice 집중 확인.
- **End-to-end 수용 기준**: Mako 환자 추론 결과에서 **대퇴골이 근위~원위 단일 라벨로 연결**되고(원래 버그 해소), 무릎·발목에서 뼈 간 오분류가 없을 것.

## 5. 리스크와 대응
- **부분 라벨 누수**: FOV 검증으로 각 스캔에 라벨 안 된 뼈 없음 확인(016). 전 환자에 대해 `build_unified.py` 변환 후 `verify_dataset.py`로 재확인.
- **좌우(L/R) 모호**: 단일 다리 추론 시 모델이 L/R 중 하나로 라벨. NoMirroring으로 일관성↑. 남으면 postprocess L/R 통합으로 흡수.
- **학습 시간**: 등방 0.6mm·52 case·5 fold. GPU 넉넉(2일+)으로 fold 병렬 → 하루+ 내 완료 예상. ES로 수렴 시 조기 종료.
- **기존 자산 보존**: 476/481 데이터·체크포인트는 삭제하지 않고 백업 유지(롤백 가능).

## 6. 산출물
- `ai_bone/build_unified.py` (신규), `Dataset490_LowerLimb` raw/preprocessed.
- 학습 체크포인트(5 fold), 추론 스크립트 갱신(`infer_mako.py`), `merge_mako.py` 제거.
- Mako 환자 최종 라벨맵 + 렌더 확인 이미지.
