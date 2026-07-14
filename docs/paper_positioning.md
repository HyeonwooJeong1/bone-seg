# 논문 포지셔닝 — MERIT 정독 + 갱신 gap + 차별점 + Intro 초안

원문 정독: `논문/stanford medical/Decentralized Instruction Tuning...pdf` (MERIT, ICML 2026, arXiv 2606.01717).

---

## A. MERIT 키포인트 (우리가 빌려오는 것)

**문제.** 이질적 데이터 혼합을 centralized joint 학습하면 (1) gradient 간섭(negative transfer, LR 제약),
(2) 무거운 동기화(all-reduce) 비용. 둘은 결합돼 있음.

**아이디어.** "모두 한 궤적에 묶지 말고, **conflict로 나눠 독립 학습 후 파라미터 공간에서 한 번에 병합**."

**5-stage 알고리즘 (Algorithm 1):**
1. **Gradient conflict 추정** — merge-ready init θ⁰에서 데이터셋별 대표 gradient(캘리브레이션 ≤200샘플 평균, 정규화) → **코사인 유사도 행렬 C (T×T)**. 파라미터 1/s(기본 s=5, 20%)만 subsample해도 full과 상관 0.98+. 증분 확장 O(Tm).
2. **PCA 분해** — C에 PCA → 데이터셋별 r차원 임베딩 (r∈{1,2,3} → K=2^r∈{2,4,8} 브랜치). 코사인-PCA 사용.
3. **균형 분할** — PCA축을 따라 **sample-balanced median으로 재귀 50/50 분할**(데이터량 균형 유지).
4. **통신 없는 브랜치 학습** — 각 그룹을 θ⁰에서 독립 학습(같은 backbone/하이퍼, 데이터만 다름). 총 예산 = joint와 동일.
5. **token 가중 병합** — θ̄ = Σ wₖθₖ, wₖ = Nₖ/ΣN (Nₖ=그룹 토큰수). 균형이면 단순평균.

**이론(flat basin 2차 근사) 3결과:**
- **병합 = 곡률가중 분산 감소**: L(병합) ≤ ΣwᵢL(θᵢ), 이득 Gvar = ½Σλℓ·Varw(uℓ·δᵢ) ≥ 0. 분산이 **고곡률 방향**에 몰릴수록 큼.
- **conflict-aware PCA 분할이 Gvar 최대화**: 이득이 spectral gap λ₁/λ₂로 커지고 λ₁³로 스케일(곡률이 gradient 충돌 gt=−HΔt와 병합식에 이중 진입). random 분할보다 우월.
- **병합 = spectral filtering + 암묵적 norm 정규화**: 유효 조건수 κ 낮춤 + θ⁰로 수축(‖병합−θ⁰‖²≤Σwᵢ‖θᵢ−θ⁰‖²) → PAC-Bayes 일반화. **개별 브랜치 train loss가 더 높아도 held-out은 좋음**(암묵 정규화 서명).
- **merge-ready init 진단**: linear mode connectivity(loss barrier 0), displacement(병합이 θ⁰에 2.4–2.9× 더 가까움), perturbation robustness(더 평평).

**그들이 쓴 baseline(=우리도 반드시 재현):** joint(0.5/1/2ep), **random 분할+병합**, **conflict-induced 분할**, **uniform model soup**, K-means 클러스터 분할(부록). 핵심 관찰: **random 분할조차 joint-1ep을 이김**(flat basin 병합 이득), **conflict 분할이 최고**. 결과 3B/136 task 54.3→57.0.

**MERIT가 스스로 강조한 차별점(우리도 같은 논리로 우리 것을 지켜야):**
- vs model soup: soup=같은 데이터·다른 seed 평균(분산감소); MERIT=이질 혼합의 **disjoint 분할** 병합, **분할 선택이 병합 품질을 결정**.
- vs Federated/Local SGD: FL=프라이버시로 **고정 분할**; MERIT=데이터 중앙 보유 하에 **분할 자체를 conflict로 최적화**(핵심 이득원).
- vs gradient 멀티태스크(PCGrad/GradNorm): 그들은 매 step 동기화된 per-task gradient 필요; MERIT는 **학습 전에** 충돌 처리 → step 동기화 제거.

---

## B. 3D 부분라벨 뼈 분할로 옮길 때 — 무엇이 유지/변형되나 (=우리 분석적 novelty)

1. **Merge-ready init**: MERIT=instruction-tuned LLM. 우리=**CADS-사전학습 nnU-Net**. 3D 분할에서 merge-ready인지 **LMC/displacement/perturbation로 검증** 필요. U-Net Transplant가 "사전학습 regime(wide minima)이 병합을 좌우"를 보였으니 **그 위에 서서** CADS init의 merge-readiness를 정당화.
2. **★ Conflict 구조가 다르다(핵심 관찰거리)**: MERIT=공유 어휘 → dense 충돌. 우리=**부분라벨 = disjoint/overlap 라벨 채널**. 서로 다른 부위(척추 vs 늑골) 데이터셋은 gradient가 **거의 직교(저충돌)**일 수 있고, **같은 부위 다른 규칙**(척추 번호/좌우 규칙 CTSpine1K↔VerSe↔Spine-Mets)은 **강한 충돌**일 수 있음. → **"gradient-conflict PCA가 해부학적 부위 그룹을 복원하나, 아니면 라벨-규칙 충돌을 드러내나?"** 는 의료영상 특이 신규 질문.
3. **Gradient 추정 비용**: 3D U-Net은 파라미터 큼 → MERIT의 subsample(s=5) 대신 **디코더 말단층/랜덤투영**으로 저차원화(우리 estimate_conflict.py 방향).
4. **token 가중 → case/voxel 가중** 병합.
5. **Instance Norm**(러닝스탯 없음) → model-soup의 BN 병합 문제 회피 → 병합에 유리(우리 이점).

---

## C. 사용자 아이디어 평가 — "비슷한 뼈 데이터셋끼리 묶어 학습 후 마지막에 merge"

**핵심: 이건 MERIT의 파이프라인 그 자체(브랜치 학습→병합)이고, 다만 분할을 gradient-conflict가 아니라
'해부학적 유사성(도메인 prior)'으로 하는 변형이다.** 좋은 직관이고 **강력한 해석가능 baseline**이 된다.

**단, MERIT 이론이 주는 반전 주의:** MERIT는 "**충돌하는 걸 떼어놓고, 정렬된 걸 묶어라**"이다.
"비슷한 뼈끼리 묶기"는 예컨대 **척추 데이터셋 3종(CTSpine1K·VerSe·Spine-Mets)을 한 그룹**에 넣는데,
이들은 **같은 척추라도 라벨 규칙이 달라 오히려 gradient가 충돌**할 수 있다. 즉 "부위 유사 ≠ gradient 정렬".
→ 순진한 해부학 그룹핑이 **최적이 아닐 수** 있다.

**그러나 우리 setting에선 반대로 유리할 수도:** 부분라벨이라 **서로 다른 부위 데이터셋은 다른 출력 채널을 갱신**
→ 충돌이 적어, 부위별 그룹핑이 사실상 conflict-aware 분할과 가까울 수 있다.

**→ 결론: 사용자 아이디어를 "해부학-prior 분할" 이라는 이름의 baseline/변형으로 정식 포함**하고,
**gradient-conflict 분할 vs 해부학 분할 vs joint**를 비교하는 것이 바로 논문의 핵심 실험이자 기여.
질문: **"데이터로 발견한 충돌 분할이, 뻔한 해부학 분할을 이기는가?"** — 이기면 방법의 가치 입증,
비슷하면 "해부학 prior로 충분"이라는 실용적 발견. **어느 결과든 논문거리.**

---

## D. (a) 경쟁작 대비 차별점 — 문장 단위

**Bonnet (ISBI'26 oral, arXiv 2601.22576).**
- 그들: 축골격+사지 **62라벨**, 그러나 **단일 데이터셋(TotalSegmentator 911장)만으로 from-scratch**, **병합/다중데이터 통합/부분라벨 없음**. 기여 = **속도**(HU threshold + sparse-conv, 2.69s). 일반화는 HU+z-score로.
- 우리 차별: "개별 축골격 라벨"이라는 **결과물은 겹치지만**, 우리는 **여러 전문가 부분라벨 데이터셋을 conflict-aware로 통합·병합**. 축이 다름(그들=속도, 우리=다중데이터 학습법). Bonnet의 62라벨을 **참조 taxonomy·zero-shot baseline**으로 활용.

**Skellytour (Radiology:AI 2025).**
- 그들: 전신 **60라벨** + 피질/해면, **단일 소스 90 WBCT(골수종)**, 병합/다중데이터 통합 없음. 논지="정밀 소수데이터가 weak-label 대규모 능가".
- 우리 차별: 단일 정밀 데이터 vs **다수 부분라벨 데이터 통합+병합**. Skellytour를 **강 baseline**으로.

**CL-Net (arXiv 2503.12698, 2025) — 통합 계열 최강 경쟁.**
- 그들: **frozen universal encoder + organ별 pruned decoder head(Lottery Ticket)** + **body-part-guided로 디코더 출력 병합**. **235 구조(193 organ+33 lymph+9 lesion) — 개별 뼈가 타깃 아님**. **weight 병합·gradient-conflict 없음**(오히려 "separate client model을 aggregate 하지 않는다"고 명시=병합 회피). continual(순차) 확장. 앙상블 대비 86.1 vs 83.9.
- 우리 차별(강함): (i) CL-Net=**순차 continual·단일모델·디코더 헤드**; 우리=**병렬 독립 브랜치 + weight 병합**. (ii) CL-Net=**출력(logit) 병합 by 부위**; 우리=**weight 병합 by conflict**. (iii) CL-Net=**장기 중심(뼈 아님)**; 우리=**뼈 특화**. (iv) CL-Net은 병합을 **회피**한다고 명시 → 우리는 **병합을 채택**하고 그 이유(탈중앙·재학습불요·프라이버시)를 이론으로 뒷받침. → 정면 충돌 회피 + 두 패러다임 비교로 포지셔닝.

**U-Net Transplant (MICCAI 2025) — 방법론 최근접.**
- 그들: **독립 학습된 task vector를 task arithmetic으로 병합**, **gradient-conflict 분할 없음**(태스크가 미리 주어짐: BTCV 4, ToothFairy2 4). 기여=**"wide-minima 사전학습이 병합 잘 되게 한다"**. 소수 태스크, 뼈/부분라벨 초점 아님.
- 우리 차별: (i) 그들=**사전 정의된 태스크 병합**; 우리=**gradient-conflict로 분할을 발견**(MERIT의 active partitioning). (ii) task arithmetic vs **conflict-aware split + 가중 병합**. (iii) 그들=**언제** 병합이 되나(사전학습); 우리=**어떻게 나눠야** 부분라벨 다중-뼈에서 최적 병합인가. → **상보적**: 그들의 사전학습 발견을 **인용·계승**해 우리 CADS init의 merge-readiness를 정당화.

---

## E. (b) Intro / Contributions 초안

**작업 제목(가안):** *Conflict-Aware Dataset Merging for Partially-Labeled Whole-Body Skeletal Segmentation*

**Intro 논리 흐름(4문단):**
1. **동기**: 임상은 전신 개별-뼈 분할을 원하지만, 라벨은 **부위별·기관별로 흩어진 부분라벨 데이터셋**에 파편화돼 있다(척추=VerSe/CTSpine1K, 늑골=RibSeg, 골반=CTPelvic1K, 하지=VSD, 두개=MUG500 …). 전신 통합 모델(TotalSegmentator, Skellytour, Bonnet, CADS)이 있으나 대개 **단일(약)라벨 소스** 또는 **대규모 pseudo-label**에 의존한다.
2. **문제**: 이 데이터를 하나로 합쳐 학습하는 표준 방식(joint pooling)은 **데이터셋 간 gradient 간섭**과 **데이터 중앙집중(프라이버시·거버넌스)** 부담을 진다. 부분라벨 통합 기존해법(DoDNet, MultiTalent, UniSeg, CL-Net)은 **단일 궤적/단일 모델**을 가정한다.
3. **통찰 & 방법**: LLM에서 **conflict-aware split + weight merge**(MERIT)가 joint를 이겼고, 3D 분할에서 **모델 병합 자체는 가능**함이 보였다(U-Net Transplant). 우리는 이 둘을 잇는다 — **부분라벨 뼈 데이터셋들을 gradient-conflict로 분할해 공유 사전학습(CADS) init에서 독립 학습한 뒤 case-가중 병합**하는 최초의 프레임을 제안하고, **해부학-prior 분할·joint·앙상블·무분할 병합(task arithmetic)**과 비교한다.
4. **관찰의 신규성**: 부분라벨(**disjoint 채널**) 세팅에서 **gradient-conflict가 해부학 구조를 복원하는지/라벨-규칙 충돌을 드러내는지**를 최초로 분석하고, **경계 이행부**에서의 이득을 정량화한다.

**Contributions (bullet):**
- **C1 (프레임워크).** 부분라벨 전신-뼈 분할을 위한 **conflict-aware 데이터셋 분할 + 가중 weight 병합** 파이프라인(CADS pseudo 사전학습 → 부위별 전문가 GT 부분라벨 fine-tune → 병합). 3D dense·ignore-label·instance-norm에 맞춘 gradient-conflict 추정(저차원화)과 case/voxel 가중 병합.
- **C2 (분석).** 3D 분할에서 CADS init의 **merge-readiness**를 LMC/displacement/perturbation로 검증하고, **부분라벨 gradient-conflict 구조의 해부학적 해석**(부위 직교성 vs 라벨-규칙 충돌: 척추 번호/좌우)을 제시.
- **C3 (실험).** joint-pooling · **해부학-prior 분할(=사용자 아이디어)** · random 분할 · **무분할 task-arithmetic 병합(U-Net Transplant식)** · 확률맵 앙상블 · 공개모델(Bonnet/Skellytour/CADS) zero-shot 대비, **per-class Dice/HD95 + 경계 이행부** 정량 우위. 병합법(가중평균/TIES/task arithmetic)·K·사전학습 regime ablation.
- **C4 (실용).** **탈중앙·프라이버시**(데이터 미공유 통합)와 **재학습 없는 데이터셋 추가**(새 부위 → 새 브랜치 학습 후 병합)라는 확장성.

**필수 방어(리뷰어 대비):**
- "결과물이 Bonnet/Skellytour와 중복" → 기여는 **방법(conflict-merge)**, 그들은 **baseline**.
- "MERIT 단순이식" → **3D dense·부분라벨·disjoint 채널**로의 비자명 이식 + **해부학 conflict 분석** + **U-Net Transplant(무분할) 대비 실측 이득**.
- "CL-Net이 앙상블 이김" → CL-Net=continual·출력병합·organ; 우리=병렬·weight병합·뼈. 축 다름 + 비교 포함.
- K=2 소규모 **merge 붕괴 안 함** 게이트 먼저.
