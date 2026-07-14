# 프로젝트 진행 기록 (Project Log)

사용자가 전 과정을 검토할 수 있도록 정리한 연대기. 상세 산출물은 각 문서/커밋 참조.
(SDD 태스크 세부 원장은 `.superpowers/sdd/progress.md`.)

---

## Phase A — 3D 뼈 시각화 앱 개선 (PyQt5/PyVista)
- 뼈 표면 스무딩(Taubin, 부피보존, n_iter=20) — 울퉁불퉁 완화(구멍 안 뚫리게).
- AI 분할을 기본 동작으로: 앱 로드 시 AI 뼈만, 레거시 HU/threshold UI 숨김(ai_first_mode).
- UI 영어화(해외 협업), 3D 클릭 뼈 선택(토글), Hide/Show 자동 해제, 랜드마크 선택 색 변경(노랑), Selection Info 패널, 랜드마크 메모.
- 세션 저장/로드: **AI 뼈만** 저장·복원(레거시 HU 뼈 제외), 아이보리 볼륨 숨김(흰 뼈 문제 해결).
- **개발 버전 관리**: `APP_VERSION`(app/constants.py) → 창 제목·UI 라벨·세션 기록, git 태그 `v1.0.0`.
- **3D 크롭 수정**: AI 뼈에 크롭이 안 걸리던 문제 → `_apply_crop_to_ai_bones`로 실시간 clip.

## Phase B — 전신 뼈 통합 모델: 연구·설계
- **데이터셋 조사**: TotalSegmentator, CADS(22k/167), Skellytour, VerSe/CTSpine1K, RibSeg(v2)/RibFrac,
  CTPelvic1K, VSDFullBody(하지), MUG500(두개), Spine-Mets(Stanford). → `docs/unified_bone_taxonomy.md`.
- **전략 확정**: 2단계 = **CADS pseudo 사전학습 → 전문가 부분라벨 fine-tune**.
- **MERIT 채택**: conflict-aware split + weight merge (사용자가 논문 지정) → taxonomy 문서에 학습법 반영.
- **통합 taxonomy v1**: 배경+53 축골격(두개·척추 C1–L5 개별·천골·늑골 L/R 1–12·흉골·관골 L/R).
- **스펙**: `docs/superpowers/specs/2026-07-13-unified-wholebody-bone-design.md`(코드-only, 서버 실행은 런북).
- **계획**: `docs/superpowers/plans/2026-07-13-unified-wholebody-bone.md`(17 TDD 태스크).

## Phase C — 구현 (Subagent-Driven, 브랜치 feat/bone-v1 → master 병합)
- 17개 TDD 태스크(구현자+리뷰어 서브에이전트, 태스크당 리뷰). 로컬 ct_env에서 **전 42~45 테스트 통과**(GPU 불필요).
- 모듈(`ai_bone/`): `taxonomy_v1`·`label_map`(+datasets/*/label_map.json 8개)·`geometry`·`harmonize`·
  `dedup`·`build_raw`·`verify_dataset`·`download`·`eval/metrics`+`evaluate`·`merit/split`+`merge`+
  `estimate_conflict`(서버 스텁)·`train/`(partial-label·MERIT trainer + stage/queue/docker 스크립트)·`runbook.md`.
- 최종 whole-branch 리뷰(opus) 통과 + cleanup(BOM, download resume, .gitattributes 등). master 병합.

## Phase D — 실행 CLI·서버 운영 (GPU-free)
- **다운로드 CLI**: `sources.py`(데이터셋 소스 레지스트리) + `python -m ai_bone.download`.
  - 검증 record: TotalSegmentator 10047292, CTPelvic1K 4588403, RibFrac CT 3893508/3893498/3893496.
  - VerSe=OSF(osf.io/nqjyw·t98fz), RibSeg 마스크=GDrive, MUG500=Figshare, Spine-Mets=TCIA(수동).
  - **CADS=HuggingFace**(gated) — 사용자 HF 토큰 로그인 후 다운로드. subset(0040_saros·0008_ctorg·0038_amos).
  - **download_file 재시도+resume+크기검증**(대용량 zip 연결 끊김 대응).
- **build_raw CLI**: 명시적 pairs 매니페스트 → harmonize→verify 게이트→nnUNet_raw. **멀티프로세싱 `--workers 16`**(공유서버 배려, 124코어 독점 금지).
- **상태 확인**: `ai_bone/dl_status.sh`(표 형식: size/status/files/disk).
- **서버 학습=Docker**(정책): 이미지 `bone-nnunet:2.8.1` 존재, `docker run --gpus` OK.
  `ai_bone/train/docker_train.sh`(커스텀 trainer bind-mount). merit trainer import 경로 버그 수정.
  **GPU는 사용자 지시 전까지 미실행.** 여유 GPU=2·3·4·5(0·1·7 사용중).
- **서버 접속**: `ssh -i ~/ad067.pem ubuntu@114.110.134.100`, 작업루트 `/data1/bone`(9TB 여유), conda `pt210_py312`.

## Phase E — 문헌·논문 준비
- **문헌 조사**(ML+의학): `docs/related_work_gap.md`. 경쟁작 = Bonnet(62라벨), Skellytour(60), CADS,
  CL-Net(235구조 continual), U-Net Transplant(3D merging+pretraining), MedSAMix, MERIT.
- **MERIT 원문 정독** → `docs/paper_positioning.md`: MERIT 5-stage·이론, 경쟁작 문장단위 차별점,
  Intro/Contributions 초안, 사용자 아이디어("비슷한 뼈끼리 묶어 merge"=해부학-prior 분할 baseline) 평가.
- **핵심 gap**: "conflict-aware split+merge를 **3D 부분라벨 뼈 분할**에" = 미점유.
- **실험 설계**: `docs/experiment_design.md`(RQ·baseline M0–M7·지표 DSC/NSD/HD95/instance-id·**복잡 뼈 난이도 층화**·
  병합 진단·conflict 시각화·ablation·통계).

---

## 현재 상태 (2026-07-14)
- **다운로드 진행 중**: totalseg/ribfrac_ct resume 재개(연결 끊김 → 수정 후 이어받는 중), ctpelvic1k 사실상 완료,
  CADS subset 진행(HF 429 백오프로 느림). `dl_status.sh`로 확인.
- 코드 파이프라인·문서 완비. **GPU 학습은 대기**(사용자 지시 필요, Docker 경로 준비됨).

## 다음 단계(예정)
1. 다운로드 완료 확인 → 데이터셋별 **pairs 매니페스트** 작성 → `build_raw`로 통합·verify.
2. iso0.6 전처리(-np 16).
3. (GPU 승인 시) Stage1 CADS 사전학습 → merge-readiness 진단 → conflict 분할 → K=2 게이트 → M1–M6 학습.
4. 평가(NSD·instance-id·난이도 층화 지표 **evaluate.py에 추가 필요**) → ablation → 논문 표/그림.

## 주요 문서 색인
- 데이터/taxonomy: `docs/unified_bone_taxonomy.md`
- 스펙/계획: `docs/superpowers/specs/…design.md`, `docs/superpowers/plans/…bone.md`
- 문헌 gap: `docs/related_work_gap.md`
- 논문 포지셔닝(MERIT 정독·차별점·Intro): `docs/paper_positioning.md`
- 실험 설계: `docs/experiment_design.md`
- 서버 실행 런북: `ai_bone/runbook.md`
- SDD 태스크 원장: `.superpowers/sdd/progress.md`
