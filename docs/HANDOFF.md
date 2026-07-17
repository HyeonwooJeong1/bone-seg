# 전신 뼈 통합 분할 (Unified Whole-Body Bone Segmentation) — 프로젝트 핸드오프

> 이 문서는 **다른 Claude 인스턴스가 이어받아 작업**할 수 있도록 프로젝트의 모든 맥락·경로·명령·상태·다음 단계를 최대한 자세히 담은 인수인계 문서다. 요약이 아니라 "그대로 따라 하면 되는" 수준으로 작성했다.

날짜 기준: 2026-07-17. 저장소 최신 커밋 `d65bc15`.

---

## 0. 한 줄 요약 / 현재 위치

- **목표**: 부분 라벨(partial-label) 공개 CT 데이터셋 여러 개를 통합해 **54클래스 전신 뼈(축골격+골반대) 3D 분할 모델**을 만들고, **MERIT(conflict-aware split + weight merge)** 방식으로 학습한 뒤 **논문**을 쓴다.
- **접근**: 2단계 = ① CADS 의사라벨 **사전학습** → ② 6개 전문가 GT 데이터셋 **fine-tuning**(부분라벨 marginal loss + MERIT).
- **지금까지 완료**: 데이터 파이프라인 **전부 완성** — 다운로드 자동화 → 6개 데이터셋 ETL/build → 통합 병합(3255 케이스) → ignore 라벨 정리 → **nnU-Net 전처리(iso 0.6mm)까지 끝. 학습 직전 상태.**
- **다음**: **GPU 학습**(사용자가 "GPU 사용 가능" 신호를 줄 때까지 절대 시작 금지). 그 전에 CADS 사전학습 데이터 정비.

**⚠️ 절대 규칙(사용자 지시, 반드시 지킬 것):**
1. **GPU 학습은 사용자가 명시적으로 허락하기 전까지 절대 실행하지 않는다.** 지금까지 실행한 것은 전부 CPU 데이터 처리다.
2. **모든 서버 작업은 Docker 컨테이너 안에서** 한다(로컬은 코드 작성·테스트만). GPU 학습도 준비되면 Docker(GPU 옵션)로만.
3. **공유 서버**다. 다른 사용자의 컨테이너·이미지·프로세스를 함부로 죽이거나 지우지 않는다. worker 수를 과하게 잡지 않는다(124코어지만 16 정도로 배려).
4. 커밋 메시지 끝에 반드시: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
5. 콘솔/NHN 접속 비밀번호는 저장·기록 금지. SSH는 키 전용.
6. (보안) HF 토큰 `hf_...`이 과거 채팅에 노출됨 → 사용자에게 폐기 권고 완료. 새 토큰 필요 시 사용자에게 요청.

---

## 1. 환경 / 접속 / 경로

### 1.1 로컬 (Windows, 사용자 PC)
- 작업 폴더(git repo): `C:\Users\정현우\Desktop\VPI LAB\stanford medicine`
- git 원격: `https://github.com/HyeonwooJeong1/bone-seg.git`, 브랜치 `master` (push 인증 캐시됨)
- **로컬 테스트 파이썬(ct_env)**: `C:\ProgramData\anaconda3\envs\ct_env\python.exe`
  - 있음: `SimpleITK 2.5.5`, `pydicom 3.0.2`, numpy 등 → **전체 테스트 스위트(98 tests) 실행 가능**
  - 없음: `scipy`, `scikit-image`, `pydicom_seg`, `nnunetv2` → 이들에 의존하는 코드는 **Docker에서만** 실행/테스트
  - 실행 예: `"/c/ProgramData/anaconda3/envs/ct_env/python.exe" -m pytest tests/ai_bone/ -q` → `98 passed`
- Bash 도구는 Git Bash(POSIX). 경로에 한글(`정현우`)/공백 있으니 따옴표 필수.

### 1.2 서버 (공유 H100, Ubuntu)
- 접속: `ssh -i ~/ad067.pem ubuntu@114.110.134.100` (키 `ad067.pem`은 로컬 홈 `~/`에 있음). `sudo`는 NOPASSWD.
- **디스크 구조 (중요)**:
  - `/` (`/dev/vda1`, 194G) = **Docker 루트(`/var/lib/docker`)가 여기 있어 잘 참**. 2026-07-17 기준 73%(53G 여유). 여기에 큰 파일 쓰지 말 것.
  - `/data1` (Lustre 네트워크 FS, 10T, 약 4T 여유) = **모든 데이터/전처리/결과는 여기**. 소파일 많으면 `du`/`cp`가 느림(정상).
- **주요 경로**:
  - `/data1/bone/raw/<dataset>` — 원본 다운로드
  - `/data1/bone/nnunet/raw/DatasetNNN_*` — 빌드된 nnU-Net raw 데이터셋
  - `/data1/bone/nnunet/preprocessed/` — 전처리 결과 (Dataset520 + 옛 476/481/490)
  - `/data1/bone/nnunet/results/` — 학습 결과(예정)
  - `/data1/bone/build/` — **로컬 코드(ai_bone, docker)를 여기로 전송해서 Docker가 마운트해 실행**. run 스크립트들도 여기.
  - `/data1/bone/tmp/` — 컨테이너 HOME/임시
  - `/data1/bone/miniforge3/` — conda 설치본(원래 `/home/ubuntu/miniforge3`였으나 디스크 확보로 이동, **심볼릭 링크 `/home/ubuntu/miniforge3 → /data1/bone/miniforge3`**). 학습 env `pt210_py312` (Python 3.12, **torch 2.10.0+cu126**), 그리고 `tf220_py312`.
    - ⚠️ 이 conda는 심볼릭 링크로 작동한다. `/home/ubuntu/miniforge3/...` 절대경로가 그대로 유효. 옮기거나 링크 건드리지 말 것.
  - `/data1/shared/gpu/gpu_monitor.py` — **공유 GPU 모니터**(팀 공용, 다른 사용자도 봄). ubuntu가 miniforge3 base python으로 실행 중. **죽이지 말 것.** (미니포지 이동 시 잠깐 재시작한 이력 있음.)

### 1.3 Docker
- 이미지: **`bone-pipeline:latest`** = `FROM bone-nnunet:2.8.1` + `docker/requirements.txt` 추가설치 + `COPY ai_bone` + 커스텀 트레이너 nnunetv2에 복사.
  - 베이스에 이미 있음: SimpleITK, scipy, numpy, nibabel, **pydicom 3.x**, **nnunetv2 CLI**(`nnUNetv2_*`), torch(GPU).
  - `docker/requirements.txt` 추가분: `requests, huggingface_hub, hf_transfer, scikit-image, matplotlib, gdown, osfclient, boto3`.
  - (주의) `pydicom-seg`는 pydicom 3.x와 비호환이라 **제거**함. DICOM-SEG는 pydicom 직접 파싱으로 처리(§4.6).
- **표준 실행 패턴** (호스트 코드가 이미지 코드를 덮어쓰도록 `-w /data1/bone/build` 사용, 소유권 위해 `--user`, `/`를 안 건드리게 HOME/tmp를 /data1·RAM으로):
  ```bash
  docker run -d --name <NAME> --user "$(id -u):$(id -g)" \
    -e HOME=/data1/bone/tmp --tmpfs /tmp:size=32g \
    -v /data1:/data1 -w /data1/bone/build \
    bone-pipeline:latest <command>
  ```
  - `-w /data1/bone/build` 로 두면 `python -m ai_bone.*` 가 **전송된 최신 코드**를 씀(이미지 재빌드 불필요, 새 pip 의존성이 없을 때).
  - 새 pip 의존성을 넣었을 때만 이미지 재빌드 필요.
- **코드 전송 방법**(로컬 → 서버 빌드 디렉토리). ⚠️ `ai_bone/data`(30G)·`ai_bone/nnunet`을 반드시 제외(안 하면 tar가 hang/디스크 폭발):
  ```bash
  KEY=~/ad067.pem
  tar czf - --exclude 'ai_bone/data' --exclude 'ai_bone/nnunet' \
    --exclude '__pycache__' --exclude '*.pyc' ai_bone docker \
  | ssh -i "$KEY" ubuntu@114.110.134.100 'cd /data1/bone/build && tar xzf -'
  ```
- **이미지 재빌드**(requirements 바뀔 때만):
  ```bash
  ssh -i ~/ad067.pem ubuntu@114.110.134.100 \
   'cd /data1/bone/build && cp docker/.dockerignore .dockerignore \
    && timeout 500 docker build -t bone-pipeline:latest -f docker/Dockerfile . >/data1/bone/rebuild.log 2>&1 && echo OK'
  ```
  - ⚠️ `.dockerignore`는 **빌드 컨텍스트 루트(`/data1/bone/build/.dockerignore`)**에 있어야 적용됨(`docker/`에 있으면 무시됨). 내용: `ai_bone/data`, `ai_bone/nnunet`, `**/__pycache__`, `*.pyc`, `*.zip`, `*.tar.gz`.

### 1.4 SSH 실행 팁(겪은 이슈)
- 각 SSH 호출은 도구 타임아웃(기본 2분, 최대 10분)이 있음. **긴 작업은 `docker run -d`(detached)로 띄우고 폴링**한다.
- 백그라운드 쉘 작업은 `nohup ... & ` + 완료 플래그 파일 패턴 사용(예: `&& touch /data1/bone/x.done`), `pkill`은 SSH 세션을 끊을 수 있으니 스크립트-파일 패턴 권장.
- 거대한 로그(전처리 config 한 줄이 수백 KB) 때문에 `docker logs | tail`이 잘림 → **`grep -c PREPROCESS_DONE` 같은 표적 검색**으로 확인.
- Lustre에서 `du`/`find`가 느려 타임아웃 나면 완료 플래그·`pgrep`만 가볍게 확인.

---

## 2. 방법론 / 설계 (논문 컨셉)

### 2.1 통합 taxonomy v1 — 54 클래스
`ai_bone/taxonomy_v1.py` (`UNIFIED_V1`, `NAME_TO_ID`, `NUM_CLASSES=54`, `IGNORE_LABEL=54`, `FG_NAMES`).

| id | 이름 | id | 이름 | id | 이름 |
|----|------|----|------|----|------|
| 0 | background | 21–25 | L1–L5 | 39–50 | Rib_R_1–12 |
| 1 | Skull | 26 | Sacrum | 51 | Sternum |
| 2–8 | C1–C7 | 27–38 | Rib_L_1–12 | 52 | Hip_L |
| 9–20 | T1–T12 | | | 53 | Hip_R |

- 전경 1..53, 배경 0.
- **`IGNORE_LABEL = 54`** (⚠️ 매우 중요). 처음엔 255였으나 **nnU-Net은 ignore 라벨이 "가장 큰 라벨 값(= max_fg+1)"이어야** 함 → 255를 거부. fg가 1..53이라 ignore=54. (커밋 `d65bc15`에서 전 코드 255→54 반영, 그리고 이미 빌드된 Dataset520 라벨 파일도 255→54로 relabel 완료.)

### 2.2 부분 라벨 처리 (marginal loss)
- 각 데이터셋은 taxonomy 중 **일부 클래스만** 라벨링돼 있다(예: RibSeg는 늑골만, CTSpine1K는 척추만).
- 각 케이스에 **`<case>.present.json`** 사이드카(`{"present_labels": [...]}`) 를 붙여 "이 케이스에 실제 라벨된 클래스"를 기록.
- 학습 시 **marginal loss(Shi et al. 2021)**: annotate 안 된 전경 채널을 배경으로 접어(collapse) 손실 계산 → 부분 라벨을 안전하게 섞어 학습.
- 구현: `ai_bone/train/marginal_loss.py`(`collapse_to_present`, `present_mask_from_ids`, `MarginalDiceCELoss`), `ai_bone/train/marginal_trainer.py`(`nnUNetTrainerMarginal`, 서버 GPU에서 train_step 배선 마무리 필요).
- taxonomy에 없는 뼈(L6, T13, coccyx, 병변 등)는 **IGNORE(54)** 로 칠해 손실에서 제외.

### 2.3 MERIT (Decentralized Instruction Tuning) — conflict-aware split + weight merge
논문 원본: `C:\Users\정현우\Desktop\VPI LAB\논문\stanford medical\Decentralized Instruction Tuning...pdf` (arXiv 2606.01717, ICML 2026). 요지:
1. merge-ready 초기화 θ0 (공통 사전학습)에서 시작.
2. 각 데이터셋의 gradient를 뽑아 **cosine 충돌 행렬 C** 계산(≤200 calib 샘플, 파라미터 subsample s=5).
3. **열-중심화 PCA** → 재귀적 sample-balanced-median split으로 K=2^r 개 브랜치로 분할(비슷한/충돌 적은 데이터끼리 묶음).
4. 각 브랜치 **독립 학습** → **token-weighted merge**로 하나의 모델로 병합.
- 이론: merging = 곡률 가중 분산 감소; PCA split이 이득 최대화(λ1³로 스케일, spectral gap과 함께 증가); merging = spectral filtering + implicit norm regularization.
- baseline: joint(전부 한꺼번에), random-split, conflict-split, model soups, K-means.
- 구현: `ai_bone/merit/`
  - `split.py` — `pca_conflict_split`
  - `merge.py` — `weighted_average`, `ties_merge` (torch/numpy 겸용)
  - `conflict_analysis.py` — `cosine_matrix`, `pca_embed`, ARI/NMI, plots, CLI
  - `estimate_conflict.py` — 실제 nnU-Net gradient 추출(`make_projection`, `reduce_grad`, `select_params`, `_random_patch`) — **GPU 필요, 서버 실행**
  - `merge_diagnostics.py` — `displacement_report`, `lmc_loss_barrier`, `perturbation_robustness`, `make_nnunet_eval_fn`

### 2.4 평가 지표 (Metrics Reloaded, Nature Methods 2024 기반)
`ai_bone/eval/`:
- `metrics.py` — DSC, NSD@τ(3mm), HD95, ASSD, clDice(얇은 늑골용, skeletonize)
- `instance_metrics.py` — Panoptic Quality(PQ=RQ×SQ, VerSe/panoptica), identification rate + centroid localization(VerSe), confusion pairs, **L/R swap rate**, rib recall(>0.7), label accuracy
- `bone_groups.py` — REGION_GROUPS, LR_PAIRS, TRANSITION_ZONES(경추/흉추 경계 등), DIFFICULTY_STRATA, RIB_FIRST/INTERMEDIATE/TWELFTH → **복잡한 뼈일수록 세분 분석**
- `evaluate.py` — `evaluate_dir`, `region_summary`, `difficulty_summary`, `evaluate_instances_dir`, CLI

경쟁 연구 대비 포지셔닝: Bonnet(ISBI'26), Skellytour(Radiology:AI 2025), CADS(22k/167구조), CL-Net(continual), U-Net Transplant(MICCAI'25). 관련 문서: `docs/related_work_gap.md`, `docs/paper_positioning.md`, `docs/experiment_design.md`, `docs/metrics_justification.md`, `docs/unified_bone_taxonomy.md`, `docs/project_log.md`.

---

## 3. 데이터셋 (전부 빌드 완료)

빌드된 nnU-Net raw 데이터셋 (`/data1/bone/nnunet/raw/`):

| Dataset | 케이스 | 내용 | 소스/방식 | build overlap-thr |
|---|---|---|---|---|
| Dataset512_TotalSeg | 1227 | 전신 뼈(per-structure) | Zenodo 10047292 | 0.25 |
| Dataset511_CTPelvic1K | 99 | 천추·좌우 골반(요추는 grouped→ignore) | Zenodo 4588403 | 0.5 |
| Dataset513_RibSeg | 496 | 늑골 24개(좌우 각 12) | RibSeg v2 마스크(gdrive) + RibFrac CT(Zenodo) | 0.3 |
| Dataset514_CTSpine1K | 1005 | 척추 C1–L5 | HuggingFace `alexanderdann/CTSpine1K` | 0.3 |
| Dataset515_VerSe | 374 | 척추(벤치마크) | bonescreen S3 zip(http) | 0.3 |
| Dataset516_SpineMets | 54 | 전이성 척추(DICOM-SEG) | IDC 공개 S3(boto3) | 0.25 |
| **Dataset520_UnifiedFT** | **3255** | **위 6개 통합 병합(최종 학습용)** | merge_raw | — |
| Dataset510_AxialFT | 1326 | (레거시) TotalSeg+CTPelvic1K 옛 병합 | — | — |

- **Dataset520_UnifiedFT = 최종 fine-tuning 데이터셋.** case_id는 `<dataset>__<원래id>` 형식(예: `verse__sub-verse010`, `ribseg__RibFrac1`). `case_datasets.json`에 출처 기록.
- 각 케이스: `imagesTr/<cid>_0000.nii.gz`, `labelsTr/<cid>.nii.gz`, `labelsTr/<cid>.present.json`. `dataset.json`의 labels에 `"ignore": 54`.

### 3.1 각 데이터셋 라벨 매핑 (label_map.json, `ai_bone/datasets/<name>/label_map.json`)
- **totalseg**: IDENTITY(값이 이미 unified id). `combine.py`가 per-structure 바이너리를 unified id로 결합.
- **ctpelvic1k**: 1(요추 그룹)→ignore(grouped), 2→Sacrum, 좌우 hip. (요추는 개별 라벨 없어 ignore.)
- **ribseg**: raw 1–24 → **1–12=오른쪽(Rib_R), 13–24=왼쪽(Rib_L)**. 기하 검증으로 확정(axcodes L,P,S에서 world-x 부호). CT는 RibFrac(`RibFracXXX-image.nii.gz`).
- **ctspine1k**: 1–24=C1–L5, **25=L6→ignore**(1005개 중 18개), 26+ 없음. (전체 스캔으로 확정.)
- **verse**: 1–24=C1–L5, **25=L6→ignore(50개), 28=T13→ignore(6개)**, 26(천추)/27 없음. (374개 전체 스캔.)
- **spinemets**: DICOM-SEG SegmentLabel('T1 vertebra'..)을 파싱 → C/T/L=taxonomy, 그 외(L6/T13/S*)=IGNORE(54). combine이 unified id 직접 생성 → label_map은 IDENTITY + 54 passthrough(grouped source_value 54→ignore).
- **mug500**: **SKIP 권장**(STL 표면메쉬 210GB, Skull 1라벨은 TotalSeg가 이미 커버).

### 3.2 다운로드 방식 (`ai_bone/download.py` + `ai_bone/datasets/sources.py`)
`download_dataset(name, dest_root, ...)` 지원 method: `zenodo`, `huggingface`, `gdrive`(gdown), `osf`(osfclient), **`http`**(직접 URL, resumable), `manual`.
- 예: `docker run ... bone-pipeline python -m ai_bone.download <name> --dest /data1/bone/raw [--max-workers 4] [--allow <glob>]`
- **verse**: osfclient가 403(OSF 큰 파일 외부 스토리지) → **bonescreen S3 직접 zip 6개**(http)로 전환. URL: `https://s3.bonescreen.de/public/VerSe-complete/dataset-verse{19,20}{training,validation,test}.zip`. 총 48G(압축), 374 스캔.
- **spinemets**: metadata.csv의 `s3://idc-open-data/<uuid>/*`를 boto3 익명(`UNSIGNED`)으로 다운로드. 스크립트 `/data1/bone/build/idc_download.py`(ThreadPoolExecutor 24). 35582 dcm, 19G.

---

## 4. 코드 구조 (`ai_bone/`) — 파일별 역할

### 4.1 taxonomy / 라벨
- `taxonomy_v1.py` — 54클래스 정의, `IGNORE_LABEL=54`.
- `label_map.py` — `LabelMap` dataclass, `load_label_map`, `remap_array`(원본값→unified id LUT; grouped→ignore; **float dtype 입력도 int 캐스팅해 처리** — VerSe dir-iso 리샘플 대응, 커밋 `e6844af`).

### 4.2 데이터셋 ETL (`ai_bone/datasets/`)
- `registry.py` — `DATASETS` (name→DatasetSpec, `label_map_path`). 등록: cads, totalseg, verse, ctspine1k, ribseg, ctpelvic1k, spinemets, mug500.
- `sources.py` — 다운로드 소스 레지스트리(method/URL/record/repo). `_ALLOWED_METHODS`.
- `combine.py` — per-structure 바이너리(TotalSeg/CADS) → 단일 unified-id seg. `TS_NAME_TO_UNIFIED`, `combine_arrays`, `combine_case`, CLI.
- `make_pairs.py` — build_raw가 먹을 `[ct, seg, case_id]` 페어 매니페스트 생성. 함수: `match_by_token`, `totalseg_pairs`, `ctpelvic1k_pairs`, **`ribseg_pairs`**, **`ctspine1k_pairs`**, **`verse_pairs`**(BIDS, VerSe19/20 중복 dedup, `__MACOSX` 제외), `write_pairs`. CLI `--dataset {totalseg,ctpelvic1k,ribseg,ctspine1k,verse,generic}`.
- **`dicom_seg.py`** (Spine-Mets 전용, §4.6) — DICOM CT 시리즈 + DICOM-SEG → (CT, unified seg).
- `merge_raw.py` — `merge_raw({name:raw_dir}, out_dir, link=True)`: 여러 raw를 하드링크로 하나로. case_id에 `<dataset>__` prefix, present sidecar 병합, `case_datasets.json`+`dataset.json` 생성. CLI `--sources name=path`.

### 4.3 빌드 파이프라인
- `nifti_io.py` — `read_sitk(path)`: SimpleITK 읽기 + **nibabel fallback + 방향코사인 정규화(SVD polar decomposition, RAS→LPS)**. TotalSeg의 비직교(2e-4) 방향 대응.
- `harmonize.py` — `harmonize_case(ct, seg, lm, spacing_mm)`: 라벨 remap + 정렬 + 등방 리샘플.
- `verify_dataset.py` — `verify_case`(overlap = fg∩(HU≥thr)/fg), `is_pass(report, overlap_thr)`. **뼈 마스크가 실제 골밀도와 겹치는지 게이트.**
- `build_raw.py` — `build_from_pairs(pairs, lm, raw_dir, spacing=0.6, hu_thr=200, workers=16, overlap_thr=0.5)`: 페어별 읽기→harmonize→verify→`imagesTr/<cid>_0000.nii.gz`+`labelsTr/<cid>.nii.gz`+present sidecar 쓰기. `write_dataset_json`(ignore=54). CLI `--pairs --dataset --out --workers --overlap-thr`.

### 4.4 전처리 / 플랜
- `make_iso06_plan.py` — 등방 0.6mm 플랜 `nnUNetPlans_iso06.json` 생성. Dataset476의 검증된 iso06 3d_fullres 설정(patch [224,80,128], batch 2)을 계승, 없으면 spacing만 0.6 override. **476 템플릿이 `/data1/bone/nnunet/preprocessed/Dataset476_PelvisThighs/`에 있어 정식 계승됨.**

### 4.5 MERIT / eval / train — §2.3, §2.4 참고. train/:
- `partial_label_trainer.py`, `merit_finetune_trainer.py`, `marginal_loss.py`, `marginal_trainer.py`(GPU에서 train_step 배선 마무리), `docker_train.sh`, `run_queue.sh`, `stage1_pretrain.sh`, `stage2_baseline.sh`.
- `nnUNetTrainerNoMirroring_ES.py` — 좌우 대칭 뼈라 **미러링 증강 끔** + early stopping.

### 4.6 DICOM-SEG 디코더 (`dicom_seg.py`) — 직접 파싱(pydicom-seg 미사용)
- `segment_label_to_unified(label)` — 'T1 vertebra'→'T1', C/T/L 범위 밖(L6/T13/S*/'sacrum')→`'__ignore__'`, 비-척추(병변 등)→`None`.
- `combine_segments(seg_binaries)` — 세그먼트별 바이너리 → unified id 배열. **real id가 IGNORE보다 우선(겹칠 때)**. non-taxonomy→IGNORE(54).
- `decode_seg_volume(ds)` — pydicom `pixel_array` + `PerFrameFunctionalGroupsSequence`(ReferencedSegmentNumber + ImagePositionPatient를 slice normal에 투영)로 볼륨 복원, IOP/PixelSpacing으로 geometry 구성.
- `seg_to_unified_image(seg_path, ref_ct)` — 위 결과를 sitk 이미지로 만들고 **CT 그리드에 nearest 리샘플**(size 일치 보장).
- `read_ct_series`, `convert_case`, `find_cases(root)`(레이아웃 무관 study 페어링), `main()` CLI(`--root --staging --pairs`).

---

## 5. 재현: 각 데이터셋 build 명령 (전부 완료됐지만 재실행용)

전제: 코드 전송(§1.3) + `/data1/bone/raw/<원본>` 존재. 전부 detached Docker.

```bash
# 공통 헤더
KEY=~/ad067.pem; H='ssh -i '"$KEY"' ubuntu@114.110.134.100'
U='--user "$(id -u):$(id -g)" -e HOME=/data1/bone/tmp -v /data1:/data1 -w /data1/bone/build bone-pipeline:latest'

# --- RibSeg (Dataset513) : /data1/bone/build/run_ribseg.sh ---
#  unzip RibFrac 이미지 → make_pairs(ribseg) → build_raw(overlap 0.3)
#  ct-root=/data1/bone/raw/ribfrac_ct/images, mask-root=.../ribseg/extracted/ribseg_v2/seg

# --- CTSpine1K (Dataset514) : run_ctspine.sh ---
#  make_pairs(ctspine1k) ct-root=raw_data/volumes mask-root=raw_data/labels → build_raw(0.3)

# --- VerSe (Dataset515) : run_verse.sh ---
#  make_pairs(verse) --root /data1/bone/raw/verse/extracted → build_raw(0.3)
#  (추출은 verse_extract_scan.py: *_ct.nii.gz + *_seg-vert_msk.nii.gz만 풀고 __MACOSX 제외)

# --- Spine-Mets (Dataset516) : run_spinemets.sh ---
#  dicom_seg --root .../spinemets/dicom --staging .../staging --pairs ... → build_raw(0.25)

# --- 통합 병합 (Dataset520) ---
docker run -d --name merge_ft $U python -m ai_bone.datasets.merge_raw \
  --out /data1/bone/nnunet/raw/Dataset520_UnifiedFT \
  --sources totalseg=/data1/bone/nnunet/raw/Dataset512_TotalSeg \
    ctpelvic1k=/data1/bone/nnunet/raw/Dataset511_CTPelvic1K \
    ribseg=/data1/bone/nnunet/raw/Dataset513_RibSeg \
    ctspine1k=/data1/bone/nnunet/raw/Dataset514_CTSpine1K \
    verse=/data1/bone/nnunet/raw/Dataset515_VerSe \
    spinemets=/data1/bone/nnunet/raw/Dataset516_SpineMets
```
서버에 실제 run 스크립트들이 `/data1/bone/build/` 에 남아있음: `run_ribseg.sh`, `run_ctspine.sh`, `run_verse.sh`, `run_spinemets.sh`, `run_preprocess.sh`, `idc_download.py`, `relabel_ignore.py`, `verse_extract_scan.py`, `scan_ctspine.py`.

---

## 6. 전처리 (완료) — Dataset520 iso 0.6mm

스크립트 `/data1/bone/build/run_preprocess.sh` (핵심: **nnUNet_preprocessed를 /data1로**, `/tmp`는 RAM tmpfs로 → `/`를 안 건드림):
```bash
export nnUNet_raw=/data1/bone/nnunet/raw
export nnUNet_preprocessed=/data1/bone/nnunet/preprocessed
export nnUNet_results=/data1/bone/nnunet/results
export nnUNet_compile=f HOME=/data1/bone/tmp MPLCONFIGDIR=/data1/bone/tmp
cd /data1/bone/build
nnUNetv2_extract_fingerprint -d 520 --verify_dataset_integrity -np 16
nnUNetv2_plan_experiment -d 520
python ai_bone/make_iso06_plan.py 520          # → nnUNetPlans_iso06.json (476 템플릿 계승)
nnUNetv2_preprocess -d 520 -plans_name nnUNetPlans_iso06 -c 3d_fullres -np 16
```
실행:
```bash
docker run -d --name preprocess --user "$(id -u):$(id -g)" --tmpfs /tmp:size=32g \
  -e HOME=/data1/bone/tmp -v /data1:/data1 -w /data1/bone/build \
  bone-pipeline:latest bash /data1/bone/build/run_preprocess.sh
```
**결과(검증됨)**: `PREPROCESS_DONE`, 에러 0. `/data1/bone/nnunet/preprocessed/Dataset520_UnifiedFT/`:
- `nnUNetPlans_iso06_3d_fullres/` 에 **6510 `.b2nd`(3255케이스 × data+seg) + 3255 `.pkl`**, 총 **1.1T**.
- 설정: patch **[224,80,128]**, spacing **[0.6,0.6,0.6]**, batch **2**, CTNormalization, PlainConvUNet 6-stage, InstanceNorm3d.

**⚠️ 만약 재빌드/재병합해서 라벨 파일이 다시 255를 갖게 되면** nnU-Net이 거부한다. `relabel_ignore.py`처럼 255→54로 바꾸고 `dataset.json`의 `labels.ignore=54`로 맞춰야 함. (지금 코드는 IGNORE_LABEL=54라 새 빌드는 자동으로 54. 단, 이미 빌드된 소스 513–516은 255를 가진 케이스가 있으니 재병합 시 relabel 필요.)

---

## 7. 다음 단계 (GPU 학습 — **사용자 허락 전 금지**)

### 7.0 학습 전 준비 상태 (2026-07-17 갱신)
GPU-free 준비물은 **완료**됨(커밋 `f19485a`, 108 테스트):
- ✅ **marginal 부분라벨 trainer 완성** (`nnUNetTrainerMarginal`): loss 벡터화 + **ignore(54) 복셀 마스킹** + present.json을 **RAW labelsTr**에서 로드 + nnU-Net DS 래퍼 재사용 + train/validation_step에서 present stash. **이미지에서 import·discovery 검증됨**(MRO: Marginal→NoMirroring_ES_PL→ES→NoMirroring; `recursive_find_python_class`로 `-tr nnUNetTrainerMarginal` 발견 확인). Dockerfile에 등록 완료 → **이미지 재빌드 완료**(marginal_trainer가 nnunetv2에 baked-in).
- ✅ **CV splits 생성 완료**: `ai_bone/train/make_splits.py` → `/data1/bone/nnunet/preprocessed/Dataset520_UnifiedFT/splits_final.json`(+`test_ids.json`). **trainpool 2721 / test 534**(Spine-Mets 54 전부 도메인 홀드아웃 + 나머지 15% test, fold별 val은 데이터셋 stratified).

**남은 것 = GPU 필요**:
- ⏳ **1-epoch 스모크 테스트**(GPU): `batch['keys']`가 case id를 담는지, marginal loss+present mask+DS가 end-to-end 도는지, whole-body 0.6mm patch [224,80,128]·batch2 **VRAM** 확인. `train_step`/`validation_step`이 nnU-Net 2.8.1 API와 정확히 맞는지 최종 확인(runbook §7).
- ⏳ **CADS 사전학습 θ0**(아래).

준비되면 순서:
1. **CADS 사전학습 데이터 정비**: `/data1/bone/raw/cads`에 52G/24814파일(SAROS/CT-ORG/AMOS subset) 있음(다운로드가 hung이라 kill함). CADS는 per-structure 바이너리 → `part_NNN`→구조 매핑 표가 필요(아직 미정, `combine.py` 확장 필요). 부족하면 Docker로 `python -m ai_bone.download cads --allow <shard>` 재개.
2. **사전학습(Stage1)**: CADS 의사라벨로 θ0 학습 → `Dataset500_AxialPretrain`(예정). `stage1_pretrain.sh` 참고.
3. **Fine-tuning(Stage2)**: Dataset520, plan `nnUNetPlans_iso06`, trainer는 **marginal 부분라벨**(`nnUNetTrainerMarginal`) + NoMirroring. 예시(stage2_baseline.sh, 단 baseline은 joint):
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> nnUNetv2_train 520 3d_fullres <FOLD> \
     -p nnUNetPlans_iso06 -tr nnUNetTrainerMarginal \
     -pretrained_weights /data1/bone/nnunet/results/Dataset500_.../checkpoint_final.pth --c
   ```
   - **반드시 Docker(GPU 옵션)로.** `train/docker_train.sh` 참고(bone-pipeline, bind-mount 없이).
   - `marginal_trainer.py`의 train_step(deep supervision/AMP)을 GPU에서 마무리.
4. **MERIT 실험** (오케스트레이션 코드 완비, GPU-free 검증됨):
   - **분할 3전략** `ai_bone/merit/orchestrate.py`: `assign_conflict`(gradient PCA, estimate_conflict 필요=GPU) / `assign_anatomy`(척추·늑골·골반·전신, GPU 불필요) / `assign_random`. `build(strategy,...)` → `branch_folds`(entry b = branch b, nnU-Net `fold=b`로 그 브랜치 데이터만 학습) + `merge_weights`(train case 수).
   - **생성된 아티팩트**(서버 `/data1/bone/nnunet/merit/`): `anatomy.json`(4브랜치: 골반67·늑골337·척추937·전신834), `random.json`(2브랜치). conflict는 θ0 있어야 생성.
   - **브랜치 학습**: `branch_folds`를 splits로 써서 각 브랜치 학습(GPU). **병합**: `ai_bone/merit/merge_checkpoints.py`로 N개 .pth를 case-가중 평균(dtype-safe) 또는 TIES 병합 → 단일 체크포인트. CLI가 orchestrate 아티팩트의 merge_weights 사용.
   - baseline(joint/anatomy/random/soups) 대비 비교. `merge_diagnostics.py`로 진단(LMC/displacement/perturbation).
5. **평가**: `eval/evaluate.py`로 DSC/NSD/HD95/PQ/L-R swap 등, region/difficulty별. VerSe 등은 instance metric.

---

## 8. 겪은 문제 & 해결 (재발 방지)

| 문제 | 원인 | 해결 |
|---|---|---|
| nnU-Net "ignore must be highest label" | ignore=255인데 fg 0–53 | **IGNORE_LABEL=54**, 빌드된 520 라벨 255→54 relabel |
| VerSe 빌드 43% ERROR(`arrays used as indices`) | dir-iso 마스크가 float dtype | `remap_array`에서 인덱스 int 캐스팅 |
| DICOM-SEG import 실패 | `pydicom-seg`가 pydicom 3.x 비호환 | pydicom 직접 파싱(`decode_seg_volume`), pydicom-seg 제거 |
| VerSe osfclient 403 | OSF 큰 파일 외부 스토리지 | bonescreen S3 직접 zip(http method) |
| CTSpine1K "Fetching 0 files" | top dir이 `raw_data`(언더스코어) | allow_patterns `raw_data/**` |
| CADS/CTSpine1K HF 429/403 | 익명 rate-limit / gated | HF 토큰 로그인 + `--max-workers` 낮춤 |
| TotalSeg combine ITK 오류 | 비직교 방향코사인(2e-4) | `nifti_io.read_sitk` nibabel fallback+정규화 |
| TotalSeg 다수 SKIP | 전체뼈 마스크에 저HU 골수 포함 | `--overlap-thr 0.25` |
| 컨테이너 root 소유 파일 | 컨테이너 root 실행 | `--user $(id -u):$(id -g)` |
| Docker "No module ai_bone.*" | `-w /data1/bone/bone`가 호스트 코드로 이미지 shadow | `-w /data1/bone/build` |
| Zenodo 대용량 zip 중단 | 연결 끊김 | `download_file` retry+resume+size verify |
| `/` 100% full | nnunet_pre(30G)+miniforge3(23G)가 `/home`(=`/`)에 | /data1로 이동(miniforge는 심볼릭 링크), 공유 이미지(115G)는 남의 것이라 못 지움 |
| tar 전송 hang | `ai_bone/data`(30G) 포함됨 | tar에서 `ai_bone/data`,`ai_bone/nnunet` 제외 |

---

## 9. 테스트

- 전체: ct_env에서 `"/c/ProgramData/anaconda3/envs/ct_env/python.exe" -m pytest tests/ai_bone/ -q` → **98 passed**.
- scipy/skimage/SimpleITK 필요한 테스트는 로컬 시스템 파이썬(Python312)엔 없으니 **반드시 ct_env**로.
- 테스트 파일: `tests/ai_bone/test_{sources,download_dataset,build_pairs,label_map,build_raw,merge_raw,taxonomy,dicom_seg,metrics,instance_metrics,harmonize,verify,geometry,dedup}.py` 등.
- 새 코드는 항상 TDD로 테스트 추가하고 ct_env에서 녹색 확인 후 커밋.

---

## 10. 커밋 히스토리 (최근, master)

```
d65bc15 fix: ignore label 255 -> 54 (nnU-Net requires ignore == max_fg+1)
d6f57f3 fix: decode DICOM-SEG with pydicom directly (drop pydicom-seg)
ba6414e feat: Spine-Mets DICOM-SEG ETL (per-vertebra) + identity label_map
e6844af fix: remap_array casts gather indices to int (float-typed masks)
430306a fix: VerSe label_map — L6(25)/T13(28)->ignore, drop unused sacrum(26)
e354190 feat: verse_pairs — BIDS CT<->seg match, dedup VerSe19/20 splits
aa45397 feat: VerSe via direct S3 (http method)
30487b6 feat: CTSpine1K ETL — ctspine1k_pairs + L6->ignore label_map
89c6cad fix: CTSpine1K allow_patterns raw_data/** (top dir underscore)
52fa791 feat: automate CTSpine1K via HuggingFace (alexanderdann/CTSpine1K)
f9c57c9 feat: automate RibSeg (gdrive) + VerSe (osf) downloads
cd64d1d feat: marginal loss for partial-label multi-dataset training
938a45d feat: merge_raw — assemble per-dataset raws into one FT dataset
```

---

## 11. 빠른 상태 점검 체크리스트 (인수 직후 실행 권장)

```bash
KEY=~/ad067.pem; SSH="ssh -i $KEY ubuntu@114.110.134.100"
# 1) 빌드된 데이터셋
$SSH 'for d in 511 512 513 514 515 516 520; do echo -n "Dataset$d: "; ls /data1/bone/nnunet/raw/Dataset${d}_*/labelsTr/*.nii.gz 2>/dev/null | wc -l; done'
# 2) 전처리 산출물
$SSH 'ls /data1/bone/nnunet/preprocessed/Dataset520_UnifiedFT/nnUNetPlans_iso06_3d_fullres/*.b2nd | wc -l'   # 6510 기대
# 3) 디스크
$SSH 'df -h / /data1 | grep -vE "tmpfs|udev"'
# 4) conda(심볼릭 링크) 정상
$SSH '/home/ubuntu/miniforge3/envs/pt210_py312/bin/python -c "import torch;print(torch.__version__)"'
# 5) 공유 GPU 모니터 살아있나(죽이지 말 것)
$SSH 'pgrep -f gpu_monitor.py >/dev/null && echo UP || echo DOWN'
# 6) 이미지
$SSH 'docker images bone-pipeline:latest'
# 7) 로컬 테스트
"/c/ProgramData/anaconda3/envs/ct_env/python.exe" -m pytest tests/ai_bone/ -q
```

---

**요약**: 데이터 파이프라인(다운로드→ETL→build→병합→전처리)은 **전부 완료**. `Dataset520_UnifiedFT`(3255 케이스)가 iso 0.6mm로 전처리되어 **학습 직전 상태**. 남은 것은 CADS 사전학습 정비 후 **GPU에서 marginal fine-tuning + MERIT 실험**(사용자 허락 후, Docker GPU로). 모든 코드는 커밋됨(`d65bc15`), 98 테스트 통과.
