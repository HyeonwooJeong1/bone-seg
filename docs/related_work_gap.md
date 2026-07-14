# 관련 연구 & 연구 Gap 분석 — 통합 뼈 분할 + MERIT식 conflict-aware merge

**목적:** 우리 아이디어(부위별 부분라벨 CT를 통합 → CADS pseudo 사전학습 → 전문가 GT fine-tune →
MERIT식 conflict-aware split + weight merge)로 논문을 쓰기 위한 문헌 지도와 **비어있는 gap**.

**요약 결론(먼저):** "전신 개별-뼈 분할 모델" **자체는 이미 여러 팀이 했다**(Bonnet·Skellytour·CADS).
부분라벨 다중데이터 통합도 성숙했다(CL-Net·MultiTalent·UniSeg). 3D 분할 model merging도 나왔다
(U-Net Transplant·MedSAMix). **아직 아무도 안 한 조합 = "데이터셋 gradient-conflict 인지 분할 +
weight merge를 3D (뼈) 분할에 적용"** — 여기가 우리 방법론적 novelty의 핵심.

---

## 1. 전신/개별 뼈 분할 (우리의 "결과물"과 직접 경쟁 — novelty 약함)

| 연구 | 무엇 | 우리와 관계 |
|---|---|---|
| **TotalSegmentator** (Radiology:AI 2023) | 104구조(뼈 59), weak-label 대규모 | 기준 베이스라인 |
| **CADS** (arXiv 2507.22953, 2025) | 22,022 CT / 167구조, 머리~무릎, 공개 | 우리 **사전학습 소스** = 우리가 쓰는 그 데이터셋 |
| **Skellytour** (Radiology:AI 2025, ryai.240050) | 전신 골격 **60라벨** + 피질/해면골 subseg, 90개 정밀 라벨 WBCT로 학습, DSC 0.94, TotalSeg 능가 | **개별-뼈 전신 분할 이미 함**. "정밀 소수 데이터"로 weak-label 능가 논지 |
| **Bonnet** (ISBI'26 oral, arXiv 2601.22576, 2026) | **축골격 62라벨**(늑골24·척추25·골반3·대퇴/상완/견갑/쇄골 L·R·두개·흉골) + sparse-conv 초고속(2.69s) | ★**우리 taxonomy와 거의 동일**. "개별 축골격 라벨" 자체는 이미 점유됨 → 우리는 라벨셋으로 차별화 불가 |
| **CL-Net** (arXiv 2503.12698, 2025) | 부분라벨 **20 public+16 private = 13,952 CT**, **235 fine-grained** 구조, continual learning, universal encoder+pruned decoders, **specialist nnU-Net 36개 앙상블 상한 능가**(5% 크기) | ★**부분라벨 다중데이터 통합**을 continual learning으로 이미 함. 우리 "통합" 논지의 최강 경쟁 |
| Continual Segment (arXiv 2302.00162) | 143 whole-body organs 단일 continual 모델 | 통합 계열 선행 |
| VSDFullBody(하지)·CTPelvic1K·VerSe·RibSeg·Spine-Mets | 부위별 개별 뼈 데이터/모델 | 우리 fine-tune GT 소스 |

**함의:** "전신 개별 뼈를 분할하는 통합 모델을 만들었다"만으로는 **Bonnet/Skellytour/CADS/CL-Net에 이미
선점**되어 리뷰어가 novelty 부족으로 볼 것. → 데이터셋/결과물이 아니라 **학습 방법론**으로 승부해야 함.

---

## 2. 부분라벨·이질 다중데이터 통합 (성숙 분야 — 우리의 배경)

- **DoDNet**(dynamic head), **MultiTalent**(MICCAI'23, task-specific heads), **UniSeg**(MICCAI'23, prompt-driven),
  **marginal/exclusion loss**(Shi et al.), **Conditional nnU-Net**(partial labels), **CLIP-driven Universal Model**.
- **MO-CTranS**(arXiv 2503.22557, 2025): 이질 라벨 통합, task token, 라벨 충돌 처리.
- **Federated multi-organ, inconsistent labels**(arXiv 2206.07156).
- CADS·CL-Net는 **LLM 기반 라벨 harmonization**으로 라벨 스키마 충돌을 정리.

**함의:** partial-label 통합 자체는 novelty 아님. 우리의 `ignore_label` + present-label 마스킹은 표준.
→ 이것도 배경일 뿐, 기여점 아님.

---

## 3. Model Merging (우리 방법론의 뿌리 — 여기서 gap을 판다)

### 3a. 일반(주로 LLM/2D 분류)
- **Model Soups**(가중평균), **Task Arithmetic**(task vector 산술), **TIES-merging**(trim→sign-elect→merge),
  **DARE**, **Fisher-weighted averaging**, **AdaMerging**(arXiv 2310.02575), **CABS**(ICML'25, conflict-aware
  balanced sparsification), **Representation Surgery**.
- **MERIT**(arXiv 2606.01717, 2026): shared init → **데이터셋 gradient conflict PCA 분할** → 파티션별 독립 학습
  → **token 가중 평균 병합**. ← 우리가 이식하려는 방법. **텍스트/멀티모달에서만 검증됨.**

### 3b. 의료/3D 분할에 적용된 merging (가장 가까운 선행 — 필수 차별화 대상)
- **U-Net Transplant**(MICCAI'25, 0752): ★**3D 의료 분할 model merging의 첫 분석**. **task arithmetic**으로
  독립 학습된 task vector 병합. 핵심 발견 = **"stable(wide minima) 사전학습 regime이 병합 잘 되게 한다"**.
  → **그러나 데이터셋을 conflict로 분할하지 않음**(그냥 독립 task를 병합). ← **우리의 결정적 차별점.**
- **MedSAMix**(arXiv 2508.11032, 2025): **training-free** SAM 병합(의료 25태스크). 뼈/3D nnU-Net 아님.
- 독립 nnU-Net **확률맵 앙상블 융합**(SKM-TEA/OAIZIB)·**Mixture of Modality Experts**(뇌병변): 추론시 융합이지
  weight merge 아님.
- **VISTA3D**(CVPR'25): 통합 3D 분할 foundation(프롬프트 기반) — merging 아님.

**함의:** 3D 분할 merging은 U-Net Transplant가 문을 열었지만 **"독립 task를 병합"**에 그친다.
**"데이터셋 간 gradient 충돌을 측정해 conflict-aware로 분할한 뒤 병합"**은 3D 분할/의료/뼈에서 **미발표**로 보임.

---

## 4. CT 사전학습/파운데이션 (우리 Stage1의 배경)
- **STU-Net**, **VoCo**, **CT-FM**, **Merlin**, **SuPreM**, **CT-CLIP**, **Curia**, **TAP-CT**, **CoralBay** — 대규모
  self-supervised/contrastive/vision-language CT 사전학습. 저데이터에서 nnU-Net 능가 보고.
- 우리는 **supervised pseudo-label(CADS) 사전학습 → 전문가 fine-tune** = curriculum 관점. SSL 파운데이션과
  경쟁이 아니라 **직교**(원하면 SSL init을 Stage0로 얹을 수 있음).

**함의:** "사전학습이 좋다"는 새롭지 않음. U-Net Transplant가 이미 **"사전학습 regime이 merging을 좌우"**를
보였으니, 우리는 그 위에서 **어떤 사전학습이 conflict-aware merge에 유리한가**로 연결하면 스토리가 강해짐.

---

## 5. ★ 비어있는 Gap (우리가 주장할 수 있는 novelty)

정확히 아래 **교집합**이 미점유:

> **3D (뼈) 분할**에서, **여러 부분라벨 데이터셋을 gradient-conflict로 인지·분할(MERIT식 PCA)** 한 뒤,
> **공유 사전학습 init에서 파티션별 fine-tune → 데이터량 가중 weight merge**로 단일 모델을 만들고,
> 이를 **joint-pooling** 및 **conflict-무시 merge(U-Net Transplant식)**·**앙상블**과 비교.

구체적 미해결 질문(=논문 기여점 후보):
1. **LLM에서 온 conflict-aware split+merge(MERIT)가 3D dense prediction(분할)에서도 성립하는가?**
   (dense pixel loss·deep supervision·instance norm·거대 3D gradient에서 gradient-conflict 신호가 유효한가)
2. **부분라벨 다중 뼈 데이터셋에서 conflict 구조는 무엇인가?** (부위 겹침 vs 라벨규칙 충돌 — 예: 척추 번호
   규칙 CTSpine1K↔VerSe↔Spine-Mets, 좌우 규칙 차이). 이 conflict의 해부학적 해석은 새로운 관찰.
3. **conflict-aware split+merge vs joint-pooling vs U-Net Transplant(무분할 merge) vs CL-Net(continual)**
   — 경계 부위(무릎/발목/경추-흉추/흉요 이행부, 늑골-척추) 성능을 정량 비교한 연구가 없음.
4. **탈중앙(privacy) 이점**: 데이터셋별 독립 학습→병합이라 데이터 공유 없이 통합 가능(의료 프라이버시 스토리).
   U-Net Transplant도 이 점을 들지만 conflict-aware 분할은 안 함.

---

## 6. 정직한 리스크(리뷰어가 때릴 지점)와 방어
- **"결과물 모델은 Bonnet/Skellytour와 중복"** → 기여를 **모델이 아니라 방법(conflict-aware merge)**으로 명확히.
  Bonnet/Skellytour/CL-Net를 **베이스라인**으로 넣어 비교.
- **"MERIT를 그대로 이식한 것 아닌가"** → (a) 3D dense·부분라벨로의 비자명한 이식(gradient 저차원화, ignore-loss와
  conflict 신호의 상호작용), (b) 병합 head 통일·instance norm 활용, (c) 해부학적 conflict 해석, (d) U-Net
  Transplant 대비 conflict-aware 분할의 실측 이득 — 이 4가지로 "단순 이식"이 아님을 입증.
- **"CL-Net이 이미 36 nnU-Net 앙상블을 이겼다"** → CL-Net은 **continual(순차)+단일모델**, 우리는 **병렬 학습+병합**
  (탈중앙·재학습 없이 데이터셋 추가). 축을 다르게 잡아 직접 경쟁 회피 + 두 방식 비교로 포지셔닝.
- **소수(K=2) 검증 게이트 필수** — 3D merge가 실제로 안 무너지는지 먼저 보이고 확장.

---

## 7. 추천 포지셔닝 & 최소 실험 세트
- **제목 축**: "Conflict-Aware Dataset Merging for Partially-Labeled 3D Skeletal Segmentation"
  (핵심어: conflict-aware split, weight merging, partial labels, decentralized, bone).
- **필수 베이스라인**: (i) joint-pooling(공유 init) (ii) U-Net Transplant식 무분할 task-vector merge
  (iii) 확률맵 앙상블 (iv) 가능하면 CL-Net/continual (v) Bonnet/Skellytour/CADS(공개 모델) zero-shot.
- **필수 비교 지표**: per-class Dice/HD95 + **부위 경계 이행부** 별도 + 도메인 홀드아웃(병리 Spine-Mets).
- **ablation**: 병합법(가중평균 vs TIES vs task arithmetic), K(파티션 수), 사전학습 regime(stable vs plastic,
  U-Net Transplant 연결), conflict 신호 저차원화 방식.
- **스토리 강화**: 탈중앙/프라이버시(데이터 미공유 통합) + "데이터셋 추가 시 재학습 없이 병합"(확장성).

## 8. 후보 투고처
- 방법 강조: **MICCAI / IPMI / MedIA / IEEE TMI**. ML 색채 강하면 **NeurIPS/ICLR Datasets&Benchmarks 또는
  workshop**, merging 커뮤니티(**MERGE workshop @ NeurIPS**). 임상 강조: **Radiology:AI**.

---

## 핵심 출처
- TotalSegmentator https://pubs.rsna.org/doi/full/10.1148/ryai.230024 ·
  CADS https://arxiv.org/abs/2507.22953 (github.com/murong-xu/CADS)
- Skellytour https://pubs.rsna.org/doi/10.1148/ryai.240050 (editorial 10.1148/ryai.250057) ·
  Bonnet https://arxiv.org/abs/2601.22576 (github.com/HINTLab/Bonnet)
- CL-Net https://arxiv.org/abs/2503.12698 · Continual Segment https://arxiv.org/abs/2302.00162
- U-Net Transplant (MICCAI'25) https://papers.miccai.org/miccai-2025/0971-Paper0752.html
  (github.com/LucaLumetti/UNetTransplant) · MedSAMix https://arxiv.org/abs/2508.11032
- MERIT https://arxiv.org/abs/2606.01717 (github.com/naver-ai/merit) · CABS(ICML'25) https://arxiv.org/abs/2503.01874
- AdaMerging https://arxiv.org/abs/2310.02575 · Model merging review https://arxiv.org/abs/2503.08998
- MultiTalent(MICCAI'23) · UniSeg(MICCAI'23) · MO-CTranS https://arxiv.org/abs/2503.22557 ·
  Federated inconsistent labels https://arxiv.org/abs/2206.07156
- CT 파운데이션: VoCo · STU-Net · CT-FM · Merlin · SuPreM · CT-CLIP · TAP-CT · VISTA3D(CVPR'25)
