# 3D CT 뼈 분할 딥러닝 프로젝트

> Mako 로봇수술용 CT에서 뼈를 **밀도 임계값(threshold)** 이 아니라 **학습된 3D 신경망**으로
> 정밀하게 구분하기 위한 프로젝트. 이 문서는 목표·개념·데이터·코드·인프라·진행 상황을 모두 담는다.

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [왜 딥러닝인가 — threshold의 한계](#2-왜-딥러닝인가--threshold의-한계)
3. [핵심 개념](#3-핵심-개념)
4. [데이터](#4-데이터)
5. [인프라 (H100)](#5-인프라-h100)
6. [파이프라인 4단계](#6-파이프라인-4단계)
7. [코드 파일별 상세](#7-코드-파일별-상세)
8. [서버 디렉토리 구조](#8-서버-디렉토리-구조)
9. [실행 방법 (명령어 모음)](#9-실행-방법-명령어-모음)
10. [겪은 문제와 해결 (트러블슈팅)](#10-겪은-문제와-해결-트러블슈팅)
11. [현재 상태 & 완료 예측](#11-현재-상태--완료-예측)
12. [향후 계획](#12-향후-계획)

---

## 1. 프로젝트 개요

- **목적**: 하지(고관절~발) CT에서 각 뼈를 부위별로 정확·고해상 분할하는 3D 딥러닝 모델을 만든다.
- **배경**: 기존 데스크탑 앱은 뼈를 `HU ≥ 150`(밀도 임계값)으로 판정했다. 이 방식은 관절로 붙은 뼈를 못 나누고, 조영·금속을 뼈로 오인하며, 저밀도 뼈를 놓친다.
- **접근**: 공개 라벨 데이터(VSDFullBody)로 **nnU-Net**을 등방 0.6mm 고해상으로 학습 → 우리 환자(Mako) CT에 추론.
- **용도**: 순수 연구/학술 (사용 데이터가 비상업 라이선스).

---

## 2. 왜 딥러닝인가 — threshold의 한계

CT의 각 복셀(3D 픽셀)은 **HU(Hounsfield Unit, 밀도값)** 를 가진다. 뼈는 밀도가 높아 HU가 크다.
가장 단순한 뼈 검출은 **"HU ≥ 임계값이면 뼈"** 다.

### threshold의 3가지 근본 결함
1. **오분류**: 조영제·혈관 석회화·금속 임플란트도 HU가 높아 → 뼈로 오인.
2. **누락**: 골다공증·해면골(뼈 내부 스펀지 구조)은 HU가 낮아 → 놓침.
3. **관절 분리 불가 (가장 치명적)**: 관절에서 맞닿은 두 뼈(예: 대퇴골두 ↔ 골반 관골구)는
   물리적으로 붙어 있어 HU만으로는 **경계를 나눌 수 없다.**

> **실증**: 우리 데이터의 발목뼈 덩어리를 connected-component·침식으로 분리 시도 → 실패.
> 발목뼈들이 관절강이 너무 좁아 하나로 붙어버림. threshold + 고전 영상처리로는 불가능함을 확인.

### 딥러닝은 무엇이 다른가
신경망은 밀도뿐 아니라 **뼈의 3차원 형태·위치·주변 맥락**을 학습한다.
→ "이 복셀은 대퇴골, 저 복셀은 골반"처럼 **복셀마다 어떤 뼈인지** 예측 → 관절로 붙은 뼈도 형태로 구분.

---

## 3. 핵심 개념

### 3.1 nnU-Net (3D U-Net 기반 분할 신경망)
- **U-Net**: 의료영상 분할 표준 구조. 입력을 점점 압축(encoder)해 특징을 뽑고, 다시 펼치며(decoder)
  복셀 단위 예측을 복원. encoder-decoder 사이를 잇는 skip connection이 미세 구조를 보존.
- **nnU-Net**: 데이터셋을 분석해 전처리·네트워크·학습 설정을 **자동 구성**하는 프레임워크.
  의료영상 분할의 사실상 표준 baseline.
- **동작**: CT 입력 → 3D 합성곱 여러 층 → 각 복셀이 어느 뼈일 **확률(score)** 출력 →
  가장 높은 확률의 뼈로 결정(argmax).

### 3.2 Fold · 교차검증 · 앙상블
데이터가 26명뿐이라 최대한 활용하기 위해 **5-fold 교차검증**을 쓴다.
- 데이터를 5조각으로 나눔.
- **fold 0** = 조각 1개를 검증용으로 빼고 나머지 4조각으로 학습. **fold 1** = 다른 조각을 빼고 학습... (5번)
- 각 fold는 **서로 다른 데이터 조합으로 학습된 독립 모델**.
- 여러 fold의 예측을 **평균(앙상블)** 하면 단일 모델보다 정확하고 안정적.
- 이번엔 "균형" 설정으로 **부위당 3 fold**(fold 0,1,2)를 학습.

### 3.3 등방(isotropic) 고해상 spacing
- **spacing**: 복셀 하나의 물리적 크기(mm). 작을수록 고해상.
- **등방**: 가로·세로·높이 spacing이 모두 같음 → 어느 방향에서 봐도 균일 고해상.
- nnU-Net 기본값은 데이터 중앙값으로 **비등방**(한 축만 세밀)을 잡는다. 뼈 단면(관절면·피질골) 디테일이 뭉개짐.
- 그래서 **일부러 등방 0.6mm로 커스텀**(plans 이름 `nnUNetPlans_iso06`)해서 전 방향 고해상 학습.

### 3.4 Early stopping (조기 종료)
- 학습은 **epoch**(전체 학습 데이터 1회 통과)를 반복. 기본 1000 epoch.
- 어느 시점부터 성능(dice)이 더 안 오르면 이후 epoch는 낭비.
- **커스텀 규칙**: 검증 dice(EMA)가 **75 epoch 동안 개선 없으면 자동 종료**. 단 초반 dice=0 구간을
  지나도록 **최소 200 epoch 보장**.
- **dice** = 뼈 구분 정확도. 예측과 정답이 겹치는 정도(0~1, 1이 완벽).

### 3.5 교수님 방법 — score맵 upsampling (추론 단계 적용 예정)
- 신경망은 저해상 격자에서 "각 복셀이 뼈일 확률(score)"을 낸다.
- **argmax(어느 뼈인지 결정)를 저해상에서 하면** 원본으로 늘릴 때 계단(각진 경계)이 생김.
- **확률맵을 먼저 고해상으로 부드럽게 늘린 뒤(resample) → 그다음 argmax** 하면 경계가
  **sub-voxel로 매끄럽게** 나옴.
- nnU-Net도 이 원리를 내부적으로 쓰며(`export_prediction.py`에서 logits를 resample 후 argmax),
  우리는 이를 **원본보다 더 세밀한 격자로** 밀어붙여 표면 품질을 극대화할 예정.

---

## 4. 데이터

### 4.1 학습 데이터 — VSDFullBody
- 출처: Zenodo record **8302449** (Fischer 2023), cadaver(사체) 전신 CT 기반 하지 뼈 분할.
- **26명**, 각 subject에 부위별 스캔 2개:
  - **476 = Pelvis-Thighs** (골반-대퇴): Sacrum, Hip, Femur, Patella, Tibia, Fibula (좌우)
  - **481 = Shanks-Feet** (하퇴-발): Tibia, Fibula, Talus(거골), Calcaneus(종골), Tarsals, Metatarsals, Phalanges (좌우)
- 해상도 고해상(z 0.6mm, 평면 0.75~0.87mm) → 우리 Mako와 유사 → 학습 모델이 잘 맞음.
- 라이선스: **CC BY-NC-SA 4.0 (비상업)** → 연구용 OK.
- **발뼈 개별 한계**: 거골·종골만 개별, 나머지 발목뼈·중족골·발가락은 "묶음". 중족골·발가락은
  물리적으로 떨어져 있어 **학습 후 connected-component로 개별화 가능**, 나머지 발목뼈는 붙어 있어 어려움.

### 4.2 우리 환자 데이터 — Mako
- `11423945/` 하위, **5명 환자** CT. 각자 하지 전체를 **3스테이션**(고관절·무릎·발목)으로 스캔.
- 축상 시리즈 = 'Mako'(512×512, 0.488mm 평면, 0.625mm 슬라이스). SCOUT/SAG/COR는 제외.
- z축으로 떨어진 3블록 구조(하지 프로토콜) → 블록 분할 후 처리 필요.
- **최종 추론 대상**: 학습된 모델을 이 데이터에 적용.

---

## 5. 인프라 (H100)

- **NHN Cloud** 국가 AI데이터센터 VM. **H100 80GB × 8장**, 4일 사용.
- OS Ubuntu 22.04, conda 환경 `pt210_py312`(PyTorch 2.10, Python 3.12).
- 저장소: 로컬 SSD(194G) + NAS `/data1`(10TB Lustre).
- 접속: `ssh -i AD-067.pem ubuntu@114.110.134.100`.
- **학습 데이터는 로컬 SSD**(`/home/ubuntu/nnunet_pre`)에 두고 읽음 (NAS보다 빠르고 안정적).

---

## 6. 파이프라인 4단계

```
[1단계] 초기 탐색: TotalSegmentator로 "AI 분할이 우리 데이터에 되는지" 검증
            │  (관절 붙은 뼈 분리·고해상 하이브리드 실증)
            ▼
[2단계] 데이터 준비: VSD 다운로드 → 압축해제 → nnU-Net 형식 변환(CT-라벨 정합)
            ▼
        nnU-Net 전처리: 등방 0.6mm 리샘플 (nnUNetv2_plan_and_preprocess)
            ▼
[3단계] 학습: nnU-Net 커스텀(등방0.6 + NoMirroring + early stopping), H100 6 fold 병렬
            ▼  ← 지금 여기
[4단계] 모니터링: dice/loss/GPU를 표로 확인
            ▼
[예정]  학습완료 → 우리 Mako 추론 → 교수님 방법(경계 스무딩)
        → 발뼈 개별화(CC 후처리) → 앱 통합(threshold 대체)
```

---

## 7. 코드 파일별 상세

모든 코드는 `ai_bone/`에 있다. `.py`는 처리 로직, `.sh`는 서버 실행·관리 껍데기.

### 1단계 — 초기 탐색 (TotalSegmentator 검증)
| 파일 | 역할 |
|---|---|
| `phase1_segment.py` | Mako DICOM → 축상 시리즈 선별 → z-gap 블록 분할 → NIfTI → TotalSegmentator 분할. DICOM→AI분할 전체 흐름의 원형 |
| `batch_all.py` | 위를 5명 환자 전체에 일괄 실행 |
| `highres_refine.py` | TS 마스크(1.5mm) + 원본 CT(0.6mm) 결합해 경계를 고해상으로 다듬는 하이브리드 |
| `qc_overlay.py` | 분할 마스크를 CT 단면에 색으로 겹쳐 보기 |
| `qc_grid.py` | 여러 축상 슬라이스를 격자로 겹쳐 보기 |
| `qc_surface.py` | 마스크를 3D 표면으로 렌더링(스무딩 비교) |
| `qc_label3d.py` | nnU-Net 라벨맵을 클래스별 색 3D 표면으로 렌더링 |

### 2단계 — 학습 데이터 준비
| 파일 | 역할 |
|---|---|
| `download_vsd.py` | Zenodo VSD 26명(30GB)을 8병렬 다운로드 |
| `prepare_all.py` | zip 압축해제 + convert를 476/481 양쪽 일괄 실행 |
| `convert_to_nnunet.py` | **핵심 변환기**. seg.nrrd(라벨)+CT → nnU-Net 형식. CT-라벨 geometry 정합(라벨을 CT 공간 resample), 부위별 통일 클래스 매핑, dataset.json 생성 |

### 3단계 — 학습
| 파일 | 역할 |
|---|---|
| `nnUNetTrainerNoMirroring_ES.py` | **커스텀 trainer**. 좌우 라벨 보존(NoMirroring) + early stopping(dice 75 epoch 개선 없으면 종료, 최소 200). 서버 nnU-Net 패키지 안에 설치 |
| `train_all.sh` | **학습 진입점**. 6 fold를 GPU 0-5에 분산, SSD 데이터·worker(18)·환경변수·compile off 설정 |
| `kt.sh` | 학습 프로세스 완전 정리 (좀비 방지, GPU 메모리 해제). 스크립트 파일로 실행해 pgrep 자기참조 회피 |
| `wait_and_train.sh` | 전처리 완료를 감지하면 자동으로 train_all 시작 (초기 자동화용) |

### 4단계 — 모니터링
| 파일 | 역할 |
|---|---|
| `monitor.py` | 6 fold의 epoch·dice(현재/최고)·val_loss + GPU 8장을 한 표로 요약 |
| `check.sh` | monitor를 한 번 실행 (conda 활성화 포함) — "그냥 확인" |
| `live.sh` | monitor를 N초마다 갱신 — 실시간 대시보드 |

> 데이터 다운로드·변환·전처리는 이 스크립트들과, nnU-Net 기본 CLI
> (`nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, `nnUNetv2_predict`)를 함께 사용해 수행.

---

## 8. 서버 디렉토리 구조

```
/data1/bone/                          (NAS, 10TB)
├── ai_bone/                          ← 코드 전부 (로컬에서 전송)
│   └── data/vsd/                     ← VSD 원본 (26명 CT+라벨 압축해제)
├── nnunet/
│   ├── raw/                          ← 변환된 입력 (Dataset476_*, Dataset481_*)
│   ├── preprocessed/                 ← 전처리 결과(등방 0.6mm) [원본 보관]
│   └── results/                      ← 학습 결과 (모델·로그·체크포인트)
└── train_logs/                       ← fold별 학습 stdout 로그

/home/ubuntu/nnunet_pre/              (로컬 SSD) ← 전처리 데이터 사본, 학습이 여기서 읽음(빠름)
~/.totalsegmentator/                  ← TotalSegmentator 가중치 (1단계용)
```

---

## 9. 실행 방법 (명령어 모음)

> 접속 키: `C:\Users\정현우\Desktop\H100 접속\AD-067.pem`, 서버: `ubuntu@114.110.134.100`
> 환경 활성화(원격): `source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312`

### 학습 상태 확인 (한 번)
```bash
ssh -i "<key>" ubuntu@114.110.134.100 "bash /data1/bone/ai_bone/check.sh"
```
### 학습 상태 실시간 (15초 갱신, Ctrl+C 종료)
```bash
ssh -t -i "<key>" ubuntu@114.110.134.100 "bash /data1/bone/ai_bone/live.sh 15"
```
### GPU 실시간
```bash
ssh -i "<key>" ubuntu@114.110.134.100 "nvidia-smi -l 2"
```
### 학습 시작 / 재시작
```bash
ssh ... "nohup bash /data1/bone/ai_bone/train_all.sh > /data1/bone/train_main.log 2>&1 &"
```
### 학습 완전 정리 (재시작 전)
```bash
ssh ... "bash /data1/bone/ai_bone/kt.sh"
```

### nnU-Net 핵심 환경변수 (학습/추론 시)
```bash
export nnUNet_raw=/data1/bone/ai_bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre       # SSD
export nnUNet_results=/data1/bone/ai_bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # scipy libstdc++ 충돌 방지
export nnUNet_compile=f                                    # torch.compile 끔
```

---

## 10. 겪은 문제와 해결 (트러블슈팅)

| # | 문제 | 원인 | 해결 |
|---|---|---|---|
| 1 | CT-라벨 무결성 검사 실패 | subject마다 CT와 seg의 크기·방향 다름 | 라벨을 CT 공간으로 `sitk.Resample`(nearest) |
| 2 | scipy import 실패 (CXXABI) | conda scipy가 시스템 libstdc++와 충돌 | `conda install libstdcxx-ng>=13` |
| 3 | 위가 학습에서 재발 | 시스템 libstdc++ 우선 로드 | `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 우선 |
| 4 | GPU가 놀고 학습 안 됨 | 8 job이 torch.compile 동시 컴파일 → CPU 경합 | `nnUNet_compile=f` (컴파일 끔) |
| 5 | 프로세스 정리 안 됨 / 유령 GPU 메모리 | job이 `;`체인으로 재spawn, worker 잔존 | bash 먼저 kill → GPU PID kill (kt.sh) |
| 6 | Lustre NAS IO 병목 | 데이터 읽기 느림 | RAM디스크 시도 → 불안정 → **로컬 SSD** |
| 7 | pkill이 SSH를 끊음 | 명령줄에 패턴 있어 pgrep 자기참조 | kill 로직을 스크립트 파일(kt.sh)로 |
| 8 | 전처리 계획이 비등방 | nnU-Net 자동값 | `-overwrite_target_spacing 0.6 0.6 0.6` |
| 9 | GPU 사용률 낮음 (근본) | H100이 너무 빨라 CPU augmentation이 못 따라감 | batchgeneratorsv2는 GPU 증강 미지원 → 구조적 한계, 학습엔 지장 없음 |

---

## 11. 현재 상태 & 완료 예측

- **학습 중**: 476 fold0-2 + 481 fold0-2 = **6 fold 병렬** (H100 GPU 0-5).
- **설정**: 등방 0.6mm, NoMirroring, early stopping, epoch time ≈ **210~230초**.
- **진행**: loss 정상 하강, dice가 fold별로 오르기 시작 (초기 단계).

### 완료 예측 (epoch ≈ 220초, 6 fold 병렬 동시)
| 수렴 시점 | 종료 epoch | 소요 | 완료 예상 |
|---|---|---|---|
| 빠름 | ~200 (최소보장) | ~12시간 | 익일 새벽~오전 |
| 보통 | ~300 | ~18시간 | 익일 오전~점심 |
| 느림 | ~400 | ~24시간 | 익일 오후 |

> **다 안 기다려도 됨**: dice가 충분히 높으면(예 0.9+) 그 시점 체크포인트로 바로 추론 가능.
> nnU-Net은 50 epoch마다 checkpoint + best 저장.

---

## 12. 향후 계획

1. **학습 완료/수렴** → best 체크포인트 확보
2. **우리 Mako 추론**: 5명 환자 CT를 서버로 전송 → `nnUNetv2_predict`
3. **교수님 방법 적용**: 확률맵을 원본보다 세밀한 격자로 resample 후 argmax → 표면 스무딩
4. **발뼈 개별화**: 중족골·발가락 묶음을 connected-component로 개별 분리 (후처리)
5. **앱 통합**: 기존 threshold 뼈 판정을 AI 라벨맵으로 교체 (사전계산+캐시 방식)

### 미완료/보류 트랙
- 481 fold3,4 (균형 설정에서 제외 — 필요시 추가 학습)
- 발 개별화 후처리 (학습 후)
- 앱 통합 (모델 확정 후)

---

*작성: 프로젝트 진행 기록 기반. 상세 진행 로그는 Claude 메모리(`bone_ml_plan.md`)에 별도 관리.*
