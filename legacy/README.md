# legacy/ — 보관용 (현재 파이프라인 미사용)

여기 파일들은 **과거 하지(lower-limb) 뼈 모델 시절**의 스크립트로, 전신(whole-body)
통합 모델 파이프라인으로 확장되며 **대체됨**. 2026-07-21에 `ai_bone/`에서 이곳으로
아카이브함(삭제 아님 — 필요하면 되살릴 수 있음).

**중요:** 현재 전신 파이프라인(`ai_bone/` + `tests/`)은 아래 파일들을 **import/참조하지
않음**을 확인하고 옮김. 따라서 이동해도 학습/빌드/테스트는 영향 없음.

## lowerlimb/
- **데이터 준비(하지)**: `build_unified.py`, `convert_to_nnunet.py`, `prepare_all.py`,
  `batch_all.py`, `download_vsd.py` — VSD 하지 데이터 → nnU-Net 변환/빌드. 전신에서는
  `ai_bone/download.py` + `ai_bone/datasets/*` + `ai_bone/build_raw.py`로 대체.
- **학습 모니터(하지)**: `monitor.py`, `watch_train.py`, `web_monitor.py` — Dataset490
  학습 진행 모니터.
- **학습 실행 쉘(하지)**: `train_all.sh`, `wait_and_train.sh`, `wait_and_train2.sh`,
  `check.sh`, `live.sh`, `dl_status.sh`, `kt.sh` — 옛 `/data1/bone/ai_bone/nnunet`
  레이아웃 기준. 전신에서는 `ai_bone/train/*.sh`, `ai_bone/train/docker_train.sh`로 대체.
- **docker/**: 옛 하지 도커 설정(`Dockerfile`, `run_all.sh`). 전신에서는 최상위
  `docker/`로 대체.

## 아카이브 안 한 것 (현재도 유효하거나 별개 프로젝트라 `ai_bone/`에 유지)
- `ai_bone/gpu_monitor.py` — **서버 공유 GPU 대시보드 소스**(`/data1/shared/gpu/`에 배포·실행 중). 건드리지 말 것.
- `ai_bone/gpu_keepalive.py` — GPU 유틸(임시 사용 가능).
- `ai_bone/infer_mako.py`·`viz_mako.py`·`postprocess_mako.py` — MAKO 추론(데이터 `/data1/hyeonwoo/bone/mako` 잔존).
- `ai_bone/infer_app.py`·`phase1_segment.py`·`highres_refine.py`·`render_clean.py`·`viz_smooth.py`·`qc_*.py` — 원본 3D 시각화 앱 관련.
