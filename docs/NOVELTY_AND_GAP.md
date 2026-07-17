# 우리 연구의 차별점 & 연구 Gap — 핸드오프용 (자립형)

> 다른 Claude가 이 문서 **하나만 읽어도** "우리가 뭐가 다른가"를 완전히 파악하도록 자립형으로 정리했다.
> 더 깊은 근거·문헌 지도는 같은 폴더의 `related_work_gap.md`(문헌 전수조사), `paper_positioning.md`(MERIT 정독 + 문장단위 차별화 + Intro 초안)에 있다. 프로젝트 전체 맥락은 `HANDOFF.md`.

작업 제목(가안): ***Conflict-Aware Dataset Merging for Partially-Labeled Whole-Body Skeletal Segmentation***

---

## 0. 한 문장 요약 — 우리만의 것

> **부분라벨로 흩어진 여러 뼈 CT 데이터셋을, "데이터셋 간 gradient 충돌(conflict)"을 측정해 conflict-aware로 분할한 뒤, 공유 사전학습(CADS) init에서 브랜치별로 독립 학습하고 case-가중치로 weight-merge 하여 단일 3D 전신-뼈 분할 모델을 만드는 최초의 프레임워크.**

이 "**3D 뼈 분할 × 부분라벨 × gradient-conflict 인지 분할 + weight merge**"의 **교집합**은 현재까지 **아무도 안 했다**. 이게 novelty의 핵심.

⚠️ 결정적 주의: **"전신 개별-뼈 분할 모델을 만들었다"는 것 자체는 novelty가 아니다** (Bonnet·Skellytour·CADS가 이미 함). 우리 기여는 **결과물(모델/라벨셋)이 아니라 학습 방법론(conflict-aware merge)**이다. 리뷰어가 제일 먼저 때리는 지점이므로, 논문 전체에서 이 프레이밍을 절대 흔들면 안 된다.

---

## 1. 이미 존재하는 것 (= 우리가 "새롭다"고 주장하면 안 되는 것)

| 영역 | 대표 연구 | 이미 한 것 |
|---|---|---|
| 전신 개별-뼈 분할 결과물 | TotalSegmentator, **Skellytour**(60라벨), **Bonnet**(축골격 62라벨), **CADS**(167구조) | 개별 뼈 전신 분할 "모델"은 성숙 |
| 부분라벨 다중데이터 통합 | DoDNet, MultiTalent, UniSeg, **CL-Net**(235구조), Continual Segment | 이질 라벨 통합·부분라벨 학습은 성숙 |
| 3D 의료분할 model merging | **U-Net Transplant**(MICCAI'25), MedSAMix | 3D 분할에서 "가중치 병합이 된다"는 이미 보임 |
| conflict-aware split+merge | **MERIT**(ICML'26, LLM) | 방법 자체는 존재 — 단 **텍스트/멀티모달에서만** |
| CT 사전학습/파운데이션 | VoCo, STU-Net, CT-FM, SuPreM, VISTA3D … | 사전학습이 좋다는 것도 새롭지 않음 |

**핵심 함의**: 위 다섯 축은 각각 성숙했다. 우리는 어느 한 축을 새로 발명하는 게 아니라, **아무도 안 겹쳐본 두 축(=부분라벨 3D 뼈 분할 + MERIT식 conflict 병합)을 교차**시킨다.

---

## 2. 비어있는 Gap (= 우리가 점유하는 정확한 교집합)

> **3D(뼈) 분할**에서, **여러 부분라벨 데이터셋을 gradient-conflict로 인지·분할(MERIT식 코사인행렬→PCA→균형분할)** 한 뒤, **공유 사전학습 init에서 파티션별 fine-tune → case/voxel 가중 weight merge**로 단일 모델을 만들고, 이를 **joint-pooling / 해부학-prior 분할 / 무분할 task-arithmetic 병합(U-Net Transplant식) / 앙상블 / 공개모델 zero-shot** 과 정량 비교.

이 교집합에서 파생되는 **미해결 질문 = 논문 기여점 후보**:
1. **LLM에서 온 conflict-aware split+merge(MERIT)가 3D dense prediction(분할)에서도 성립하는가?** (dense pixel loss·deep supervision·instance norm·거대 3D gradient에서 gradient-conflict 신호가 유효한가)
2. **부분라벨 다중 뼈 데이터셋의 conflict 구조는 무엇인가?** — 서로 다른 부위(척추 vs 늑골)는 gradient가 거의 **직교(저충돌)**일 수 있고, **같은 부위 다른 라벨규칙**(척추 번호/좌우 규칙: CTSpine1K↔VerSe↔Spine-Mets)은 **강한 충돌**일 수 있다. 이 충돌의 **해부학적 해석**은 의료영상 특유의 신규 관찰.
3. **conflict-aware 분할이 뻔한 해부학 분할·joint·무분할 병합을 이기는가?** — 특히 **경계 이행부**(무릎/발목, 경추-흉추, 흉요 이행부, 늑골-척추 접합)에서. 이걸 정량 비교한 연구가 없음.
4. **탈중앙(privacy)**: 데이터셋별 독립 학습→병합이라 **원본 데이터 공유 없이** 통합 가능 + **재학습 없이 데이터셋 추가**(새 부위=새 브랜치 학습 후 병합).

---

## 3. 가장 가까운 경쟁작 vs 우리 — 문장 단위 차별화

가까운 순서대로. (각 경쟁작을 **baseline으로 실험에 포함**해야 함.)

### Bonnet (ISBI'26 oral, arXiv 2601.22576) — 결과물이 가장 겹침
- **그들**: 축골격+사지 **62라벨**(우리 taxonomy와 거의 동일). 그러나 **단일 데이터셋(TotalSegmentator 911장)에서 from-scratch**, **병합·다중데이터 통합·부분라벨 전혀 없음**. 기여 = **속도**(HU threshold + sparse-conv, 2.69s).
- **우리 차별**: 개별 축골격 라벨이라는 **결과물은 겹치지만**, 우리는 **여러 전문가 부분라벨 데이터셋을 conflict-aware로 통합·병합**한다. 축이 다름(그들=추론 속도, 우리=다중데이터 학습법). → Bonnet 62라벨을 **참조 taxonomy + zero-shot baseline**으로 사용.

### Skellytour (Radiology:AI 2025, ryai.240050)
- **그들**: 전신 **60라벨** + 피질/해면골 subseg, **단일 소스 90 WBCT(골수종)**. 병합/다중데이터 통합 없음. 논지 = "정밀 소수 데이터가 weak-label 대규모를 능가".
- **우리 차별**: 단일 정밀 데이터 vs **다수 부분라벨 데이터 통합+병합**. → **강 baseline**으로 포함.

### CL-Net (arXiv 2503.12698, 2025) — "통합" 계열 최강 경쟁
- **그들**: frozen universal encoder + organ별 pruned decoder head(Lottery Ticket) + **body-part-guided 디코더 출력 병합**. **235 구조(장기 중심, 개별 뼈가 타깃 아님)**. **weight 병합·gradient-conflict 없음** — 논문에 "separate client model을 aggregate 하지 않는다"고 **명시(=병합을 회피)**. continual(순차) 확장. 앙상블 대비 86.1 vs 83.9.
- **우리 차별(강함)**: (i) CL-Net=**순차 continual·단일모델·디코더 헤드**; 우리=**병렬 독립 브랜치 + weight 병합**. (ii) CL-Net=**출력(logit) 병합 by 부위**; 우리=**weight 병합 by conflict**. (iii) CL-Net=**장기 중심(뼈 아님)**; 우리=**뼈 특화**. (iv) CL-Net은 병합을 **회피**한다고 명시 → 우리는 병합을 **채택**하고 그 근거(탈중앙·재학습불요·프라이버시)를 MERIT 이론으로 뒷받침. → 정면 충돌 회피 + 두 패러다임 비교로 포지셔닝.

### U-Net Transplant (MICCAI 2025) — 방법론 최근접
- **그들**: **독립 학습된 task vector를 task arithmetic으로 병합**. **gradient-conflict 분할 없음**(태스크가 미리 주어짐: BTCV 4, ToothFairy2 4). 기여 = **"wide-minima 사전학습 regime이 병합을 잘 되게 한다"**. 소수 태스크, 뼈/부분라벨 초점 아님.
- **우리 차별**: (i) 그들=**사전 정의된 태스크 병합**; 우리=**gradient-conflict로 분할을 능동 발견**(MERIT식 active partitioning). (ii) task arithmetic vs **conflict-aware split + case 가중 병합**. (iii) 그들=**언제** 병합이 되나(사전학습 regime); 우리=**어떻게 나눠야** 부분라벨 다중-뼈에서 최적 병합인가. → **상보적**: 그들의 "사전학습이 merge-readiness를 좌우"를 **인용·계승**해 우리 CADS init을 정당화.

### MERIT (ICML 2026, arXiv 2606.01717) — 우리가 이식하는 방법
- **그들**: LLM/멀티모달에서 shared init → 데이터셋 gradient 코사인행렬 → PCA → sample-balanced median 재귀분할(K=2^r) → 브랜치 독립학습 → token 가중 병합. 이론: 병합=곡률가중 분산감소, conflict-aware PCA 분할이 이득 최대화, 병합=spectral filtering+암묵 norm 정규화.
- **우리 차별(단순 이식이 아님을 입증하는 4가지)**:
  1. **3D dense·부분라벨로의 비자명한 이식**: disjoint 라벨 채널, ignore-loss와 conflict 신호의 상호작용, deep supervision·거대 3D gradient의 저차원화(디코더 말단층/랜덤투영).
  2. **병합 head 통일 + Instance Norm 활용**: nnU-Net은 러닝스탯 없는 IN이라 model-soup의 BN 병합 문제를 원천 회피(우리 이점).
  3. **해부학적 conflict 해석**: "gradient-conflict PCA가 해부 부위 그룹을 복원하나, 라벨-규칙 충돌을 드러내나?"는 의료영상 신규 관찰.
  4. **token 가중 → case/voxel 가중** 병합 재정의.

---

## 4. 우리의 신규 기여 (Contributions, C1–C4)

- **C1 (프레임워크)**: 부분라벨 전신-뼈 분할을 위한 **conflict-aware 데이터셋 분할 + 가중 weight 병합** 파이프라인. CADS pseudo 사전학습 → 부위별 전문가 GT 부분라벨 fine-tune → 병합. 3D dense·ignore-label(=54)·instance-norm에 맞춘 gradient-conflict 추정과 case/voxel 가중 병합.
- **C2 (분석)**: 3D 분할에서 CADS init의 **merge-readiness**를 LMC(loss barrier)/displacement/perturbation robustness로 검증하고, **부분라벨 gradient-conflict 구조의 해부학적 해석**(부위 직교성 vs 라벨-규칙 충돌: 척추 번호/좌우)을 최초 제시.
- **C3 (실험)**: joint-pooling · **해부학-prior 분할(=사용자 아이디어)** · random 분할 · **무분할 task-arithmetic 병합(U-Net Transplant식)** · 확률맵 앙상블 · 공개모델(Bonnet/Skellytour/CADS) zero-shot 대비 **per-class Dice/HD95 + 경계 이행부** 정량 우위. 병합법(가중평균/TIES/task arithmetic)·K(파티션 수)·사전학습 regime ablation.
- **C4 (실용)**: **탈중앙·프라이버시**(데이터 미공유 통합) + **재학습 없는 데이터셋 추가** 확장성.

---

## 5. 우리 세팅에서만 나오는 "신규 관찰" (논문의 차별적 매력)

MERIT를 그대로 옮기면 안 되고, **의료·부분라벨 특유의 반전**이 있다:

1. **부위 유사 ≠ gradient 정렬**: "비슷한 뼈끼리 묶기"(사용자의 직관, 해부학-prior 분할)는 예컨대 척추 3종(CTSpine1K·VerSe·Spine-Mets)을 한 그룹에 넣는데, **같은 척추라도 라벨 규칙(번호 매김·L6/T13 처리·좌우)이 달라 오히려 gradient가 충돌**할 수 있다. → 순진한 해부학 그룹핑이 최적이 아닐 수 있다.
2. **반대로 유리할 수도**: 부분라벨이라 **다른 부위 데이터셋은 다른 출력 채널만 갱신** → 충돌이 적어, 부위별 그룹핑이 사실상 conflict-aware 분할과 가까울 수 있다.
3. **→ 핵심 실험 질문**: **"데이터로 발견한 conflict 분할이, 뻔한 해부학 분할을 이기는가?"**
   - 이기면 → 방법의 가치 입증.
   - 비슷하면 → "해부학 prior로 충분"이라는 실용적 발견.
   - **어느 결과든 논문거리**(win-win 실험 설계).

---

## 6. 리뷰어가 때릴 지점 & 방어

| 공격 | 방어 |
|---|---|
| "결과물 모델이 Bonnet/Skellytour와 중복" | 기여는 **모델이 아니라 방법(conflict-aware merge)**. 그들을 **baseline**으로 넣어 비교 |
| "MERIT를 그대로 이식한 것 아닌가" | §3 MERIT의 4가지(3D dense·부분라벨 이식 / IN·head 통일 / 해부학 conflict 해석 / U-Net Transplant 대비 실측 이득) |
| "CL-Net이 이미 36 nnU-Net 앙상블을 이겼다" | CL-Net=**continual·출력병합·장기**; 우리=**병렬·weight병합·뼈**. 축이 다름 + 비교 포함 |
| "3D에서 merge가 실제로 안 무너지나" | **K=2 소규모 게이트** 먼저 — LMC loss barrier≈0, merge가 붕괴 안 함을 먼저 입증하고 확장 |
| "사전학습이 좋다는 건 새롭지 않다" | 맞다. 우리는 "**어떤 사전학습이 conflict-aware merge에 유리한가**"로 U-Net Transplant 발견을 계승·확장 |

---

## 7. 차별점을 "증명"하는 최소 실험 세트 (novelty는 결국 실험으로 증명됨)

- **필수 baseline**: (i) joint-pooling(공유 init) (ii) 해부학-prior 분할(사용자 아이디어) (iii) random 분할+병합 (iv) 무분할 task-arithmetic 병합(U-Net Transplant식) (v) 확률맵 앙상블 (vi) 공개모델 Bonnet/Skellytour/CADS **zero-shot** (vii) 가능하면 CL-Net식 continual.
- **필수 지표**: per-class **Dice/HD95** + **경계 이행부 별도**(경추-흉추, 흉요, 늑골-척추, 무릎/발목) + **L/R swap rate** + **VerSe식 instance PQ/identification** + 도메인 홀드아웃(**병리 Spine-Mets**).
- **ablation**: 병합법(가중평균 vs TIES vs task arithmetic) · K(2/4/8) · 사전학습 regime(stable vs plastic) · conflict 신호 저차원화 방식.
- 구현 위치(이미 있음): 지표 `ai_bone/eval/`, MERIT `ai_bone/merit/`(split/merge/estimate_conflict/merge_diagnostics). 데이터 `Dataset520_UnifiedFT`(3255, iso06 전처리 완료).

---

## 8. 후보 투고처
- 방법 강조: **MICCAI / IPMI / MedIA / IEEE TMI**.
- ML 색채 강조: **NeurIPS/ICLR (Datasets & Benchmarks 또는 workshop)**, merging 커뮤니티(**MERGE workshop @ NeurIPS**).
- 임상 강조: **Radiology:AI**.

---

## 부록: 핵심 출처 (자세한 링크는 `related_work_gap.md` §핵심 출처)
- TotalSegmentator(ryai.230024) · CADS(arXiv 2507.22953) · Skellytour(ryai.240050) · Bonnet(arXiv 2601.22576)
- CL-Net(arXiv 2503.12698) · U-Net Transplant(MICCAI'25 Paper0752) · MedSAMix(arXiv 2508.11032)
- MERIT(arXiv 2606.01717) · TIES/Task Arithmetic/Model Soups/AdaMerging(2310.02575)/CABS(2503.01874)
- MultiTalent·UniSeg·MO-CTranS(2503.22557) · marginal loss(Shi et al.) · VISTA3D(CVPR'25)
