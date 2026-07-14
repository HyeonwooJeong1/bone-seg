# 실험 설계 — Conflict-Aware Merge for Partial-Label Whole-Body Skeletal Segmentation

관련: `paper_positioning.md`(포지셔닝), `related_work_gap.md`(gap), `unified_bone_taxonomy.md`(라벨/데이터).

---

## 0. 연구 질문 (RQ) → 실험 매핑

| RQ | 질문 | 검증 실험 |
|---|---|---|
| **RQ1** | conflict-aware split+merge가 **joint pooling**을 이기는가? | M4 vs M1 (전 지표) |
| **RQ2** | **데이터로 발견한 conflict 분할**이 **해부학-prior 분할(사용자 아이디어)**·random을 이기는가? | M4 vs M2 vs M3 |
| **RQ3** | CADS 사전학습 init이 **merge-ready**인가(병합이 안 무너지나)? | 병합 진단(§5) + K=2 게이트 |
| **RQ4** | 이득이 **복잡한 뼈·경계 이행부**에 집중되는가? | 난이도 층화 분석(§4) |
| **RQ5** | conflict 구조가 **해부학을 복원**하나, **라벨-규칙 충돌**을 드러내나? | conflict 행렬·PCA·분할 시각화(§5) |
| **RQ6** | merge법·K·사전학습 regime의 영향? | ablation(§6) |

---

## 1. 비교 방법 (Baselines & Ours)

모두 **동일 CADS 사전학습 init θ⁰**, 동일 아키텍처(nnU-Net iso0.6, 통합 54채널 head), 동일 총 예산.

| ID | 방법 | 설명 |
|---|---|---|
| **M0** | Region specialists | 데이터셋/부위별 개별 nnU-Net (참조·앙상블 재료) |
| **M1** | **Joint pooling** (★주 baseline) | 전 데이터셋 한 풀 + balanced sampling + partial-label(ignore) |
| **M2** | **Anatomical-prior split + merge** (★사용자 아이디어) | 부위(척추/늑골/골반/하지/두개)로 그룹→독립 학습→가중 병합 |
| **M3** | Random split + merge | MERIT의 control(무작위 분할) |
| **M4** | **MERIT conflict split + weighted merge** (★ours) | gradient-conflict PCA 분할(K=2,4)→독립 학습→case/voxel 가중 병합 |
| **M5** | No-split task-arithmetic merge | 데이터셋별 모델의 task vector 산술 병합 (U-Net Transplant식) |
| **M6** | Probability-space ensemble | 데이터셋별 모델 확률맵 평균+argmax (추론 융합) |
| **M7** | Public zero-shot | TotalSegmentator·Skellytour·Bonnet·CADS 공개모델 그대로 |
| (opt) | Continual (CL-Net식) | 재현 어려우면 인용만 |

> 공정성: M1–M6은 **같은 init·같은 총 GPU/토큰 예산**. M4/M2/M3는 분할만 다름(브랜치 학습·병합 동일).

---

## 2. 데이터 분할 & 평가 세트

- **데이터셋별 train/val/test**(케이스 단위, 환자 누수 없이). 통합 test = 각 데이터셋 test 합집합.
- **외부 일반화 홀드아웃**: 학습에 안 쓴 공개셋(예: 한 데이터셋을 통째로 external로) → 도메인 이동.
- **병리 홀드아웃**: **Spine-Mets(전이성 척추)** — 정상 vs 병리 분리 보고(로버스트니스).
- **우리 Mako CT**(임상 정성 QC): 정량 GT 없으니 정성 오버레이 + 표면 품질.
- 시드 **3~5개**(주 비교는 5시드), 평균±표준편차.

---

## 3. 평가 지표 (핵심)

### 3.1 오버랩 / 표면 (per-class + mean)
- **DSC**(Dice) — per-class + 부위별 매크로 평균.
- **NSD**(Normalized Surface Dice, 뼈별 허용오차 τ; 얇은 뼈는 작은 τ) — **경계 품질 핵심 지표**.
- **HD95**, **ASSD**(평균 대칭 표면거리).
- **상대 부피오차** RVE.

### 3.2 개별-뼈(instance) 특화 — 복잡한 뼈일수록 필수
- **Instance detection F1**: 각 늑골/척추가 검출됐나(존재/누락/과검출).
- **라벨 식별 정확도**(identification rate): 척추 번호가 맞나(VerSe 표준). **off-by-one enumeration error rate**(경계 이행부에서 흔함).
- **좌/우 혼동률**(L↔R swap rate) — 좌우 라벨 데이터의 핵심.
- **인접 구조 혼동행렬**: 어떤 뼈를 어떤 뼈로 오분류하나(예: T12↔L1, 인접 늑골, 대퇴골두↔관골).

### 3.3 병합/최적화 진단 (학습 중·후 관찰) — RQ3
- **Merging gain**: DSC(merged) − Σ wₖ·DSC(branchₖ), 그리고 − DSC(joint).
- **Linear Mode Connectivity**: 브랜치쌍·브랜치↔merged 보간 경로의 **loss barrier**(0이면 이상적).
- **Weight displacement** ‖θₖ−θ⁰‖, **merged가 θ⁰에 더 가까운 비율**(MERIT Table1식).
- **Perturbation robustness**: Gaussian σ∈{.01,.05,.1} 교란 시 loss 증가(merged가 더 평평한가).
- **train-loss vs held-out**: merged가 train-loss 높아도 test 좋은가(암묵 정규화 서명).

---

## 4. ★ 복잡한 뼈 — 난이도 층화 심층 분석 (RQ4)

"복잡할수록 더 자세히"를 위해 **뼈를 난이도 축으로 층화**하고 각 축에 특화 지표를 붙인다.

### 4.1 난이도 층(bone strata)
| 층 | 예시 뼈 | 왜 어려움 | 특화 지표 |
|---|---|---|---|
| **얇음/작음** | 늑골 11–12(floating), 비골, 쇄골, 견갑골, 횡돌기 | 부분부피·저대비 | NSD(작은 τ), 두께방향 오차, 검출 F1 |
| **다중 인접(혼동)** | 개별 척추(인접 유사), 개별 늑골 | 번호 밀림·경계 붙음 | identification acc, off-by-one, 인접 혼동행렬 |
| **좌우 대칭** | 늑골 L/R, 관골·대퇴 L/R, 견갑/쇄골 | 좌우 뒤집힘 | L↔R swap rate |
| **관절 접촉면** | 대퇴골두↔관골, 대퇴↔경골, 척추↔늑골, 인접 척추 | 접촉면 분리 실패 | **접촉면 국소 NSD/HD95**(경계 band 마스크) |
| **이행부** | C7–T1, T12–L1, L5–S1 | 열거 오류 집중 | 이행부 vertebra id acc, 열거오류율 |
| **해부 변이** | 이행추(transitional), 늑골수 변이 | 분포 밖 | 변이 케이스 별도 DSC/id acc |

### 4.2 분석 산출물
- **부위·난이도층별 DSC/NSD 표** (전 방법 M1–M7 나란히).
- **뼈별 Δ(우리−joint) 히트맵** — 어떤 뼈에서 이득이 큰지 한눈에.
- **인접 혼동행렬**(척추/늑골) — 오분류 패턴.
- **이행부 열거오류 사례 그림**(정성).
- **난이도 vs 이득 상관**: "복잡한 뼈일수록 conflict-merge 이득이 큰가?"를 수치로.

---

## 5. 학습 중 관찰·기록할 값 (monitoring & logging)

### 5.1 매 학습(브랜치/joint) 로깅
- epoch, **train loss / val loss**, **pseudo-Dice(EMA)**, **per-class val Dice**, learning rate, epoch time, GPU-util.
- best/last checkpoint, 조기종료(ES) 시점.
- (nnU-Net `progress.png` + `training_log` 파싱 → CSV로 수집)

### 5.2 conflict 분석 산출물 (RQ5)
- **gradient conflict 행렬 C**(T×T 코사인) — heatmap.
- **PCA 임베딩 산점도**(데이터셋 라벨) + **분할 경계**.
- **분할 결과표**: 어떤 데이터셋이 같은 그룹인가 → **해부학 그룹과 얼마나 일치**하나(ARI/NMI로 정량).
- conflict PCA축 ↔ 해부/라벨규칙 해석(정성 + top 충돌쌍 나열: 예 CTSpine1K↔VerSe 척추번호).

### 5.3 자원/실용 지표
- **총 GPU-시간**, wall-clock, **통신량**(decentralized=0), 모델 크기, 추론 시간/스캔.
- "데이터셋 추가 시 비용": 새 부위 1개 추가 = 새 브랜치 1개 학습 + 병합(재학습 불요) — 시간 측정.

---

## 6. Ablation 매트릭스

| 축 | 값 |
|---|---|
| **분할법** | conflict-PCA(ours) · 해부-prior · random · K-means(같은 gradient 표현) |
| **K(브랜치 수)** | 2 · 4 · (8) |
| **병합법** | case-가중 평균(ours) · 단순평균 · **TIES** · task arithmetic · (Fisher) |
| **gradient 저차원화** | 디코더 말단층 · 랜덤투영 차원(256/512/1024) |
| **사전학습 regime** | CADS pseudo · (SSL init) · scratch → merge-readiness 비교(U-Net Transplant "stable vs plastic" 연결) |
| **partial-label 처리** | ignore-label · marginal loss (대조) |

---

## 7. 통계 & 보고

- **케이스 단위 paired Wilcoxon signed-rank**(M4 vs 각 baseline), per-class.
- 다중비교 보정(Holm/Bonferroni), 95% CI, effect size.
- 주 비교(M4 vs M1, M4 vs M2)는 **5시드**.
- 표: per-class DSC/NSD/HD95(±SD) + 부위 매크로 + 난이도층 + 이행부.
- 그림: 뼈별 Δ 히트맵, conflict 행렬/PCA, merge 진단(barrier·displacement), 이행부 정성.

---

## 8. 실행 순서(체크리스트)
1. 데이터 통합·전처리(iso0.6) 완료 → 데이터셋별 present-label 메타.
2. **Stage1 CADS 사전학습**(θ⁰) → merge-readiness 진단(§5.3) 먼저.
3. **conflict 추정**(estimate_conflict) → C·PCA·분할(§5.2) 산출.
4. **K=2 소규모 게이트**: 병합이 안 무너지는지 확인(RQ3) 후 확대.
5. M1–M6 학습·병합(동일 예산, 5시드) + M7 zero-shot 추론.
6. 평가(§3) + 난이도 층화(§4) + 진단(§5) + ablation(§6) + 통계(§7).
7. 우리 Mako 정성 QC.

> 코드 훅: 평가=`ai_bone/eval/evaluate.py`(현재 DSC/HD95; **NSD·instance-id·혼동행렬·이행부 band 지표 추가 필요**). conflict=`ai_bone/merit/estimate_conflict.py`+`split.py`. 병합=`merit/merge.py`.
