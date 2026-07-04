# AI 뼈 분할 통합 — 사용 & 배포 안내

기존 3D CT 앱에 **학습된 nnU-Net 통합모델(Dataset490)** 기반 AI 뼈 분할을 추가했습니다.
기존 기능(threshold 렌더, 랜드마크 점찍기, 뼈 분리/관리, 크롭, fusion, 세션 저장 등)은 **전부 그대로** 유지되고, AI는 토글 버튼으로 얹혔습니다.

## 사용법
1. 앱 실행 → 환자 선택 → **Load CT & Render** (기존과 동일)
2. 왼쪽 패널 **"AI 뼈 분할 (학습 모델)"** → **"AI 뼈 분할 실행"** 클릭
   - 처음 1회는 추론(수 분, GPU면 빠름). 결과는 `<환자>_ai_labels.npz`로 **자동 캐시** → 다음엔 즉시 표시.
   - 뼈마다 **실제 이름·색**(대퇴골/경골/슬개골/비골/골반/거골/종골/발목뼈…)으로 표시됩니다.
3. AI 뼈 위에서 **랜드마크 점찍기·이름변경·숨기기·세션저장** 등 기존 도구 그대로 사용 가능.
4. **"AI 끄기"** → threshold 렌더로 복원.

## 새 CT 추가 (바로바로)
1. 새 환자 DICOM 폴더를 **`11423945/<환자ID>/`** 에 복사
2. 앱에서 그 환자 선택 → Load → **AI 뼈 분할 실행**
   - 모델이 번들돼 있어 **오프라인**으로 그 자리에서 분할합니다.

## GPU / CPU
- **자동 감지**: NVIDIA GPU가 있으면 GPU(5-fold 앙상블, 빠름), 없으면 CPU(단일 fold, 느림).
- 수동 지정하려면 콘솔에서:
  `python -m ai_bone.infer_app <dicom_dir> <out.npz> --device cpu --folds 0`

## 다른 PC로 넘길 때 (배포)
넘길 폴더에 아래가 모두 포함돼야 합니다:
```
├── app/  main.py  dicom_utils.py       (앱)
├── ai_bone/infer_app.py + nnUNetTrainerNoMirroring_ES.py
├── models/Dataset490_LowerLimb/…       (번들 모델 — 필수)
├── requirements_ai.txt
├── setup_and_run.bat
└── 11423945/<환자들>/                   (CT 데이터, 필요분만)
```
받는 사람은 **Python 3.11 설치 후 `setup_and_run.bat` 더블클릭** → 가상환경 생성 + 의존성 설치(torch 최초 ~2.5GB 다운로드) + 앱 실행. **인터넷은 최초 설치 때만 필요**, 이후 추론은 오프라인.

## 동작 원리 (요약)
- `ai_bone/infer_app.py`: DICOM → z-gap 스테이션 분할 → 번들 모델로 21라벨 예측 → 후처리(금속·대상다리·좌우통합·closing) → 앱용 `npz`.
- `app/mixins/ai_segmentation.py`: npz를 현재 시리즈에 매칭(shape/z-range) → 뼈별 marching cubes + 약한 스무딩 → 의미론적 색·이름 → 기존 `separated_bones`로 표시.
- ES 트레이너는 추론 시 자동으로 nnU-Net 패키지에 설치됩니다(자체완결).
