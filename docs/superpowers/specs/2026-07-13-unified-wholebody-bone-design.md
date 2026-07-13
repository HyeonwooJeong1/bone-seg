# 전신 뼈 통합 분할 모델 (v1: 축골격 + 골반대) — 설계 스펙

**작성일:** 2026-07-13
**상태:** 설계 승인됨 → 구현 계획(writing-plans) 대기
**관련 문서:** `docs/unified_bone_taxonomy.md` (데이터셋 조사·taxonomy·MERIT 요약)
**이전 트랙:** VSD 하지 모델(Dataset476/481, 21라벨) — v2에서 합류

---

## 0. 한 줄 요약

부위별로 흩어진 공개 CT 뼈 분할 데이터를 **하나의 통합 라벨 체계**로 묶어,
**CADS 대규모 pseudo-label 사전학습 → 전문가 GT fine-tune** 2단계로 학습하고,
**MERIT(conflict-aware split + weight merge)** 를 joint-pooling baseline과 비교하여
단일 **축골격+골반대 뼈 분할 nnU-Net** 을 만든다.

---

## 1. 목표와 비목표

### 1.1 목표
- 축골격(두개골·척추·늑골·흉골·천골) + 골반대(관골) 를 **개별 뼈 단위**로 분할하는 단일 nnU-Net(iso 0.6mm) 확보.
- 부위별 데이터셋을 통합 taxonomy로 합치는 **재현 가능한 데이터 파이프라인**.
- **joint-pooling(baseline A)** 과 **MERIT(B)** 를 동일 조건에서 비교하는 실험 프레임.

### 1.2 비목표 (v1에서 제외)
- 사지(대퇴·슬개·경골·비골·발, 상완·전완·손) — v2. 기존 VSD 하지 모델은 그대로 유지.
- 손·손목 — 공개 뼈 라벨 부재로 로드맵에서 제외.
- 앱(PyQt) 통합 — 별도 후속(기존 `AiSegmentationMixin` 캐시 로더 재사용).

### 1.3 ★ 실행 제약 (이번 작업의 핵심 조건)
- **H100 서버(114.110.134.100)가 타 사용자와 공유 중** → 이번엔 **코드/스크립트만 완성**하고
  **실제 다운로드·전처리·학습·추론은 사용자가 나중에** GPU 여유가 생길 때 직접 실행.
- 따라서 모든 실행 단위는:
  1. **독립 실행 가능**(단계별 별도 스크립트, 앞 단계 산출물만 입력으로)
  2. **1~N GPU 유연 실행**(전체 8장 점유 가정 금지; `CUDA_VISIBLE_DEVICES`로 가용 GPU만)
  3. **checkpoint 재개**(중단/자원 반납 후 이어서)
  4. **런북(runbook) 문서**로 명령을 순서대로 제공 → 사용자가 복붙 실행
- 코드는 로컬(`ct_env`)에서 **문법·단위 테스트(소형 합성 데이터)** 까지 검증하고, GPU 필요한 부분만 서버 실행으로 미룬다.

---

## 2. 전역 제약 (Global Constraints)

| 항목 | 값 |
|---|---|
| 프레임워크 | nnU-Net v2 (v2.8.1, 서버 pt210_py312) |
| 목표 spacing | 등방 **0.6mm** (기존 `nnUNetPlans_iso06` 규칙 계승) |
| trainer | `nnUNetTrainerNoMirroring_ES` (좌우 라벨 → NoMirroring, EMA dice early stopping) |
| 라벨 정합 규칙 | CT-seg **size 같으면 `CopyInformation(ct)`, 다르면 물리 resample**(NearestNeighbor) — z-flip/CT선택 버그 교훈 |
| 좌우 기준 | LPS(앱과 동일) |
| 라이선스 | **비상업(연구)**. CADS(CC BY-NC-SA)=사전학습만, 상업 재배포 시 CC BY(TotalSegmentator 등)로 GT 재구성 가능하도록 데이터 provenance 태깅 |
| 서버 경로 | 작업 루트 `/data1/bone`, raw/preprocessed/results 동일 계승. 전처리 캐시는 **로컬 SSD `/home/ubuntu/nnunet_pre`**(Lustre `/dev/shm` 불안정 교훈) |
| 환경 함정 | `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`, `export nnUNet_compile=f` 필수(과거 CXXABI/컴파일 경합) |

---

## 3. 통합 Taxonomy (v1)

배경=0, 전경 **53 클래스**. TotalSegmentator 명명 규칙, 좌우 `_L/_R`.

| id | 라벨 | id | 라벨 | id | 라벨 |
|---|---|---|---|---|---|
| 0 | background | | | | |
| 1 | Skull | | | | |
| 2–8 | C1…C7 | 9–20 | T1…T12 | 21–25 | L1…L5 |
| 26 | Sacrum | 27–38 | Rib_L_1…12 | 39–50 | Rib_R_1…12 |
| 51 | Sternum | 52 | Hip_L | 53 | Hip_R |

- 정본은 코드 상수 `ai_bone/taxonomy_v1.py`의 `UNIFIED_V1 = {id: name}` + 역맵.
- **모든 학습 모델의 출력 head = 이 54채널(배경 포함)** — MERIT 병합 전제.
- v2 확장 시 id 54+에 사지 라벨 append(기존 id 불변 → 하위호환).

---

## 4. 데이터셋 인벤토리와 라벨 매핑

### 4.1 역할 분담

| 데이터셋 | 규모 | 단계 | 커버 v1 라벨 | 라벨 성격 | 라이선스 |
|---|---|---|---|---|---|
| **CADS**(축골격 subset) | ~수천~2만 | Pretrain | Skull·척추개별·늑골개별·흉골·천골·관골 | pseudo+QC | CC BY-NC-SA |
| **TotalSegmentator v2** | 1,204 | FT(앵커) | **v1 전 클래스** | 전문가 | CC BY 4.0 |
| **VerSe'19/'20** | ~300 | FT | C1–L5(+일부 천골) | 전문가 | 공개 |
| **CTSpine1K** | 1,005 | FT | C1–L5 | 준전문가 | 공개 |
| **RibSeg v2** | 660 | FT | Rib_L/R_1–12 | 전문가 | 공개(RibFrac 기반) |
| **CTPelvic1K** | 1,184 | FT | Sacrum·Hip_L/R·(요추=grouped) | 전문가 | 공개 |
| **Spine-Mets-CT**(Stanford) | 55 | FT | C1–L5(병리) | 전문가 | TCIA |
| **MUG500+** | 500 | FT | Skull | 전문가 | 공개(Figshare) |

### 4.2 데이터셋별 `label_map.json` 스키마

각 데이터셋 디렉토리에 하나. 원본 정수 라벨값 → 통합 taxonomy 문자열.

```jsonc
// ai_bone/datasets/<name>/label_map.json
{
  "dataset": "verse20",
  "source_format": "nifti_seg",       // nifti_seg | dicom_seg | nrrd | ply(voxelize)
  "provenance_license": "public",     // ccby | ccbync | public | tcia  (라이선스 분리용)
  "map": {                            // 원본 라벨값(str) -> 통합 라벨명
    "1": "C1", "2": "C2", ... "25": "Sacrum"
  },
  "grouped": {                        // granularity 부족 → 개별loss 제외 처리
    // 예) CTPelvic1K: 원본 요추 단일덩어리
    "lumbar_block": {"source_value": 1, "covers": ["L1","L2","L3","L4","L5"]}
  },
  "present_labels": ["C1", ... ],     // 이 데이터셋이 실제로 주석하는 통합 라벨 집합
  "notes": "..."
}
```

- **`present_labels`** = partial-label 학습의 핵심. 스캔별로 override 가능(스캔 FOV가 부위 일부만 담을 때).
- **`grouped`** 항목은 "존재하지만 개별 아님" → 해당 복셀 영역을 **ignore_label**로 마스킹(개별 척추 loss에서 제외, 배경으로 오학습 방지).

### 4.3 partial-label 규약
- nnU-Net 라벨맵에 **`ignore_label`(=255)** 사용. 스캔에 주석 없는 클래스 영역은 배경이 아니라 **ignore**로 두어 loss에서 제외.
- 스캔별 메타 `present_labels`를 `dataset.json`/사이드카에 저장. 학습 loss는 present 채널 + ignore 규약으로 계산.

---

## 5. 데이터 파이프라인

```
download → convert(→NIfTI) → remap(label_map) → geometry 정합 → resample(iso0.6)
        → dedup(환자단위) → partial-label 메타 → nnUNet_raw(DatasetXXX)
```

### 5.1 모듈 (파일 경계)
| 파일 | 책임 |
|---|---|
| `ai_bone/taxonomy_v1.py` | 통합 라벨 상수 + 역맵 + 검증 |
| `ai_bone/datasets/<name>/download.py` | 데이터셋별 다운로드(재개 가능, 병렬 제한) |
| `ai_bone/harmonize.py` | remap + geometry 정합(size 분기) + iso0.6 resample. 입력: 원본, label_map / 출력: 통합라벨 NIfTI |
| `ai_bone/dedup.py` | 환자/스캔 해시(이미지 fingerprint)로 데이터셋 간 중복 제거, 우선순위(전문가>준전문가) |
| `ai_bone/build_raw.py` | 통합 NIfTI → nnUNet_raw DatasetXXX(축골격 FT 통합셋) + present_labels 사이드카 |
| `ai_bone/verify_dataset.py` | 빈라벨 0 / geometry 일치 / 라벨값 범위 / present_labels 정합 다각도 검증 |

### 5.2 geometry 정합 (재실수 방지 — 과거 라벨 전멸 교훈)
- CT와 seg **size 동일 → `seg.CopyInformation(ct)`**(배열 그대로), **size 다름 → 물리 resample(Nearest)**.
- **CT 파일 선택 버그 방지**: 한 폴더에 복수 부위 CT가 있을 수 있음 → 부위 tag 명시 매칭, 애매하면 검증에서 겹침률(라벨∩HU≥200) < 50%면 실패 처리.
- `verify_dataset.py`가 **겹침률·빈라벨·direction**을 리포트. 통과 못 하면 build 중단.

### 5.3 dedup
- CADS·CTPelvic1K·TotalSegmentator가 동일 TCIA 원본 공유 → 이미지 다운샘플 해시로 중복군 검출.
- fine-tune GT 내 중복은 **전문가 라벨 우선**(TotalSegmentator > VerSe/RibSeg > CTSpine1K). CADS는 pretrain 전용이라 FT와 중복돼도 무방.

---

## 6. 학습 설계 (3-stage)

모든 stage는 **독립 스크립트 + 서버 실행**. 로컬은 문법·소형 테스트만.

### 6.1 Stage 1 — Pretrain (CADS)
- CADS 축골격 subset을 통합 taxonomy로 build → `Dataset500_AxialPretrain`.
- iso0.6, `nnUNetTrainerNoMirroring_ES`, 5-fold 불필요(사전학습은 all-train 1 config로 충분) → **fold "all"** 권장.
- 산출물: `checkpoint_final.pth` → **shared init**(Stage2/3의 `-pretrained_weights`).
- 스크립트: `ai_bone/train/stage1_pretrain.sh` (GPU 1장, 재개 가능).

### 6.2 Stage 2 — Baseline A (joint pooling)
- FT 전 데이터셋 통합 → `Dataset510_AxialFT`. present_labels 기반 partial-label + ignore.
- shared init에서 **dataset-balanced random sampling**(데이터셋 크기 편차 보정 오버샘플).
- 5-fold, `-pretrained_weights <stage1>`.
- 스크립트: `ai_bone/train/stage2_baseline.sh <fold> <gpu>` (1 fold=1 GPU, 개별 실행).

### 6.3 Stage 3 — MERIT B
1. **gradient conflict 측정** `ai_bone/merit/estimate_conflict.py`
   - shared init 로드 → 각 FT 데이터셋에서 소량 배치 gradient 계산.
   - **저차원화**: 디코더 말단 N층 + seg head 파라미터만, 또는 랜덤투영(고정 시드). 데이터셋별 평균 gradient 벡터 g_d 저장.
2. **PCA 충돌 분할** `ai_bone/merit/split.py`
   - {g_d} 정규화 → PCA 제1주성분(충돌축) 투영 부호로 K=2 분할(확장 시 상위 주성분으로 K=3).
   - 산출물: `partitions.json` = {partition_id: [dataset,...]}.
3. **파티션별 fine-tune** `ai_bone/merit/train_partition.sh <pid> <fold> <gpu>`
   - 각 파티션 데이터만으로 shared init에서 **짧게·낮은 LR** fine-tune(동일 54채널 head).
4. **weight 병합** `ai_bone/merit/merge.py`
   - **case/voxel 수 가중 평균**(MERIT token-weighted 대응). 대조 ablation: TIES-merging, 단순평균, task arithmetic.
   - 산출물: `merged_model.pth`.
- **★ 검증 게이트**: 먼저 **K=2, 2~3 데이터셋 소규모**로 "병합 모델이 각 파티션 대비 무너지지 않는지" 확인 후 전체 확대.

### 6.4 3D 이식 리스크 대응 (스펙 명시)
- Instance Norm(러닝스탯 없음) → 병합 안전. 확인 코드로 norm 타입 assert.
- 병합은 **동일 아키텍처/plans** 전제 → 전 파티션 `nnUNetPlans_iso06` 고정.
- fine-tune LR/epoch를 shared basin 유지 범위로(예: 초기 LR의 1/10, 200~300 epoch).

---

## 7. 평가

`ai_bone/eval/evaluate.py`:
- **per-class Dice + HD95**(경계 정밀도), 부위별 집계(경추/흉추/요추/천골/늑골/흉골/관골).
- **경계 이행부 별도 리포트**: 경흉·흉요·요천추, 늑골-척추 관절.
- **A vs B 정량 비교표** + 통계(케이스별 paired).
- **홀드아웃 분리**: 병리(Spine-Mets) vs 정상, 데이터셋별 홀드아웃 → 도메인 일반화 확인.
- **정성 QC**: 우리 Mako 데이터에 추론 → `qc_overlay` PNG(기존 `qc_overlay.py` 재사용).

---

## 8. 파일 구조 (신규/수정)

```
ai_bone/
  taxonomy_v1.py                # 통합 라벨 상수
  datasets/
    totalseg/{download.py,label_map.json}
    verse/{...}  ctspine1k/{...}  ribseg/{...}
    ctpelvic1k/{...}  spinemets/{...}  mug500/{...}  cads/{...}
  harmonize.py  dedup.py  build_raw.py  verify_dataset.py
  train/
    stage1_pretrain.sh  stage2_baseline.sh
  merit/
    estimate_conflict.py  split.py  train_partition.sh  merge.py
  eval/
    evaluate.py  qc_overlay.py(기존)
  runbook.md                    # ★ 사용자가 서버에서 순서대로 실행할 명령 모음
tests/
  test_taxonomy.py  test_harmonize.py  test_label_map.py  test_merge.py
docs/superpowers/specs/2026-07-13-unified-wholebody-bone-design.md (본 문서)
```

---

## 9. 런북(runbook) — 사용자가 나중에 실행

`ai_bone/runbook.md`에 아래를 **복붙 가능한 명령**으로 상세화(각 단계 예상시간·재개법·검증법 포함). 요지:

```bash
# (서버) 공통 환경
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/bone/nnunet/raw nnUNet_preprocessed=/home/ubuntu/nnunet_pre \
       nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH nnUNet_compile=f

# 1) 다운로드(부위별, 재개 가능)         python ai_bone/datasets/<name>/download.py
# 2) 통합 변환+정합+검증                 python ai_bone/build_raw.py --stage ft   (+ verify_dataset.py)
# 3) 전처리(iso0.6)                      nnUNetv2_plan_and_preprocess ... -p nnUNetPlans_iso06
# 4) Stage1 pretrain(1 GPU)             bash ai_bone/train/stage1_pretrain.sh
# 5) Stage2 baseline(가용 GPU마다 fold) bash ai_bone/train/stage2_baseline.sh <fold> <gpu>
# 6) MERIT: 충돌측정→분할→파티션학습→병합
#    python ai_bone/merit/estimate_conflict.py ; python ai_bone/merit/split.py
#    bash ai_bone/merit/train_partition.sh <pid> <fold> <gpu> ; python ai_bone/merit/merge.py
# 7) 평가/추론                           python ai_bone/eval/evaluate.py ; (Mako 추론+QC)
```

- **GPU 나눠쓰기 규약**: 각 train 스크립트는 인자로 받은 GPU 1장만 사용. 여러 fold를 동시에 못 돌리면 **하나씩 순차** 실행(각자 checkpoint 재개). 스크립트 상단에 "이 스크립트 = GPU 1장" 명시.

---

## 10. 테스트 전략 (로컬 `ct_env`, GPU 불필요)

- `test_taxonomy.py`: id↔name 왕복, 중복/누락 없음, 54채널.
- `test_label_map.py`: 각 `label_map.json`의 map 값이 전부 통합 taxonomy에 존재, present_labels ⊆ map 값.
- `test_harmonize.py`: 합성 CT+seg(정상/ size불일치/ direction flip)로 geometry 정합 3케이스, 겹침률 검증.
- `test_merge.py`: 소형 2모델 가중평균/ TIES가 파라미터 shape 보존·가중치 합=1 확인(랜덤 텐서).
- 문법: 전 `.py` `ast.parse` + import 스모크.

---

## 11. 위험과 완화

| 위험 | 완화 |
|---|---|
| MERIT 3D 미검증(병합 붕괴) | K=2 소규모 검증 게이트, TIES/단순평균 대조, baseline A 항상 확보 |
| 대용량 다운로드/정합 실패 | 데이터셋별 독립 다운로드·재개, verify_dataset 게이트, dedup |
| partial-label 배경충돌 | ignore_label(255) 규약, present_labels 스캔별 메타 |
| granularity 불일치(요추 grouped) | grouped→ignore 마스킹, 해당 스캔 개별 척추 loss 제외 |
| 서버 공유/중단 | 단계 분리 + checkpoint 재개 + 1GPU 유연 실행 + 런북 |
| 라이선스(CADS 비상업) | provenance 태깅, 상업 시 CC BY GT로 재구성 경로 확보 |
| 좌우/geometry 재실수 | size 분기 정합 + 겹침률 검증 필수 게이트 |

---

## 12. 오픈 이슈 (계획 단계에서 확정)
- CADS 축골격 subset의 실제 라벨 스키마/다운로드 단위 확인(HuggingFace 구조).
- RibSeg v2 라벨값→Rib_L/R_1–12 정확 매핑(좌우·번호 규칙).
- Spine-Mets/CTPelvic1K의 DICOM-SEG → NIfTI 변환 경로.
- MERIT gradient 저차원화 구체 방식(디코더 말단 층 수 vs 랜덤투영 차원).
```
