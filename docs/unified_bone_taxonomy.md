# 통합 전신 뼈 분할 — 데이터셋 + Taxonomy 매핑

**목표:** 부위별로 흩어진 공개 CT 뼈 분할 데이터를 하나의 통합 라벨 체계로 묶어
**(1) 대규모 pseudo-label 사전학습 → (2) 전문가 GT fine-tune** 2단계로 단일 전신 뼈 모델 학습.

---

## 1. 학습 전략 (2단계)

```
[Stage 0 (선택)] SSL 사전학습
    unlabeled 대규모 CT (autoPET 1,014 + AbdomenAtlas/FLARE 이미지)
    └ self-supervised (MAE/contrastive) 로 backbone 초기화

[Stage 1] 지도 사전학습 (Pretrain)  ← 규모 우선, 라벨 품질은 pseudo
    CADS (22,022 스캔 / 167 구조, 머리~무릎, pseudo-label + 자동 QC)
    └ 통합 taxonomy 로 remap 후 whole-body 라벨로 학습

[Stage 2] Fine-tune  ← 품질 우선, 전문가 GT
    TotalSegmentator + VerSe + CTSpine1K + RibSeg v2 + CTPelvic1K
    + VSDFullBody(하지·발) + MUG500+(두개) + Spine-Mets-CT(Stanford, 병리)
    └ 부분 라벨(partial-label) 학습: 각 스캔에 존재하는 라벨만 loss 계산
```

- **왜 이 순서인가:** CADS는 규모(22k)는 압도적이지만 라벨이 모델 생성(pseudo)이라 균일하지 않음 → 넓은 표현을 먼저 배우는 **사전학습**에 적합. 전문가 GT는 상대적으로 소규모지만 정밀 → **fine-tune**에서 라벨 정확도를 끌어올림.
- **부분 라벨 처리:** 한 스캔에 일부 뼈만 라벨된 경우가 많음. nnU-Net 위에 **marginal loss / DoDNet / CLIP-driven Universal Model / MultiTalent** 계열 기법으로 "있는 라벨만" 학습.

---

## 2. 통합 Taxonomy (목표 라벨 체계)

TotalSegmentator v2 명명 규칙을 기준(좌/우 = `_L`/`_R`). 세분화 단계는 데이터별 최대공약수에 맞춰 조정 가능.

### HEAD
| id 그룹 | 라벨 | 세분화 옵션 |
|---|---|---|
| head | `Skull` | (선택) `Cranium`, `Mandible` 분리 |

### SPINE (개별 척추뼈)
| 그룹 | 라벨 |
|---|---|
| cervical | `C1`…`C7` (7) |
| thoracic | `T1`…`T12` (12) |
| lumbar | `L1`…`L5` (5) |
| sacral | `Sacrum` (+선택 `Coccyx`) |

### THORAX
| 그룹 | 라벨 |
|---|---|
| ribs | `Rib_L_1`…`Rib_L_12`, `Rib_R_1`…`Rib_R_12` (24) |
| sternum | `Sternum` |

### SHOULDER / UPPER LIMB
| 라벨 |
|---|
| `Clavicle_L/R`, `Scapula_L/R`, `Humerus_L/R` |
| `Radius_L/R`, `Ulna_L/R` *(공개 데이터 희소)* |
| `Hand_L/R` (carpal+metacarpal+phalanx 통합) *(거의 라벨 없음 — 갭)* |

### PELVIS / LOWER LIMB
| 라벨 |
|---|
| `Hip_L/R` (os coxae), `Femur_L/R`, `Patella_L/R` |
| `Tibia_L/R`, `Fibula_L/R` |
| `Talus_L/R`, `Calcaneus_L/R`, `Tarsals_L/R`, `Metatarsals_L/R`, `Phalanges_L/R` |

> 기존 `Dataset490_LowerLimb`(21라벨: Femur/Hip/Sacrum/Patella/Tibia/Fibula/Talus/Calcaneus/Tarsals/Metatarsals/Phalanges 좌우 + Sacrum)은 이 통합 체계의 **PELVIS/LOWER LIMB 부분집합** → 그대로 편입 가능.

---

## 3. 데이터셋 → 통합 Taxonomy 매핑 표

| 데이터셋 | 커버 부위 (통합 라벨) | 원본 granularity | remap 시 주의 | 라벨성격 | 단계 |
|---|---|---|---|---|---|
| **CADS** (22,022) | 머리~무릎 전 구간: Skull, C1–L5 개별, Rib 개별, Sternum, Clavicle/Scapula/Humerus, Hip/Femur, Sacrum | 167 구조(뼈+장기+혈관) | 뼈 라벨만 추출, 이름 정규화. 무릎 아래(경골·발)는 커버 약함 | **pseudo+QC** | **Pretrain** |
| **TotalSegmentator v2** (1,204) | Skull, C1–L5 개별, Rib 개별, Sternum, Clavicle/Scapula/Humerus, Hip/Femur; (appendicular task) Patella/Tibia/Fibula/Tarsal/Metatarsal/Phalanges | 59 뼈 개별 | 가장 표준. 통합 taxonomy의 **기준 라벨명**으로 사용 | **전문가 검수** | **Fine-tune (핵심 GT)** |
| **VerSe'19/'20** (~300) | `C1…L5`, `Sacrum` | 척추 개별 + centroid | 척추 번호 라벨 규칙을 통합 기준으로 채택 | 전문가 | Fine-tune |
| **CTSpine1K** (1,005) | `C1…L5` | 척추 개별 | 일부 자동생성 후 검수 → 품질 편차 확인 | 준-전문가 | Fine-tune(척추 보강) |
| **RibSeg v2** (660) | `Rib_L/R_1…12` | 늑골 24개 개별 | RibFrac 원본은 **골절 라벨**이므로 반드시 RibSeg 마스크 사용 | 전문가 | Fine-tune(늑골) |
| **CTPelvic1K** (1,184) | `Sacrum`, `Hip_L`, `Hip_R`, (요추 일부) | 4-class (요추/천골/좌·우 관골) | 좌우 hip 분리 규칙 통일. 요추는 개별 아님 → 병합 처리 | 전문가 | Fine-tune(골반) |
| **VSDFullBody** (20명, 기존 사용) | `Femur/Patella/Tibia/Fibula/Talus/Calcaneus/Tarsals/Metatarsals/Phalanges_L/R`, `Hip`, `Sacrum` | 하지·발 전 뼈 개별 | **무릎 아래·발**을 채우는 유일한 고품질 소스 | 전문가(우리 검증됨) | Fine-tune(하지·발) |
| **MUG500+** (500 두개골) | `Skull` (+정상/craniotomy) | 두개골 마스크 | 하악/치아 분리 라벨은 제한적 | 전문가 | Fine-tune(두개) |
| **Spine-Mets-CT (Stanford)** (55) | `C1…L5`, `Sacrum` (+병변) | 척추 개별 + 병변 분류 | **병리 케이스** → 도메인 다양성/로버스트니스 | 전문가 | Fine-tune(척추, 하드케이스) |
| autoPET (1,014) | — (뼈 라벨 없음) | 종양 병변만 | 이미지만 사용 | 없음 | Stage0 SSL / pseudo 생성용 |
| AbdomenAtlas·FLARE | — (장기 위주) | 장기 | 이미지 풀 | 뼈 없음 | Stage0 SSL |

---

## 4. 통합 시 반드시 해결할 것 (체크리스트)

1. **라벨명·id 정규화**: 각 데이터셋 라벨을 위 통합 taxonomy 문자열로 1:1 매핑하는 dict 작성 (dataset별 `label_map.json`).
2. **granularity 불일치**: 예) CTPelvic1K는 요추가 개별 아님 → "L1–L5 병합" 클래스로 fallback 하거나 해당 스캔에서 요추 loss 제외.
3. **좌/우 규칙 통일**: 환자 좌우(L/R) 기준을 LPS 기준으로 통일(우리 앱과 동일). 데이터별 뒤집힘 확인.
4. **부분 라벨 마스킹**: 스캔별 "존재 라벨 집합"을 메타로 저장 → 없는 라벨은 loss/배경에서 제외(background collision 방지).
5. **중복 원본 제거**: CADS·CTPelvic1K·TotalSegmentator가 동일 TCIA 원본 일부 공유 → 환자 단위 dedup.
6. **spacing/orientation**: 기존 파이프라인처럼 등방 0.6mm 재샘플 + RAS/LPS 정합.
7. **라이선스 분리**: CADS 라벨=CC BY-NC-SA(비상업). 상업 배포 시 fine-tune GT는 CC BY(TotalSegmentator 등)만으로 구성하고 CADS는 사전학습에만 사용.

---

## 5-B. 학습 방법: MERIT (Conflict-Aware Splitting + Weight Merging) 적용

**논문:** MERIT — *Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging*
(NAVER AI, arXiv 2606.01717, github.com/naver-ai/merit). LLM instruction tuning용이나,
"이질적 다중 데이터셋을 한 모델로" 라는 문제 구조가 우리와 동일 → 3D 분할로 이식.

### 핵심 개념 — "다 모아 random 학습" vs MERIT
- **Joint pooling(다 모아 random 셔플):** 모든 데이터셋을 한 풀에 넣고 무작위 샘플. 단순하지만
  데이터셋 간 **gradient 간섭(conflict)** 으로 서로의 성능을 깎을 수 있음. → 이게 MERIT가 이기려는 **baseline**.
- **순차(dataset별로 차례로) fine-tune:** catastrophic forgetting → **금지**.
- **MERIT:** shared init → 데이터셋별 gradient conflict 측정 → **conflict 낮은 그룹으로 분할** →
  그룹별 독립 학습 → **데이터량 가중 weight 병합**. "충돌 나는 것끼리 억지로 섞지 않고, 나눠 학습 후 합침."

### MERIT 4단계 → 우리 파이프라인 매핑
1. **Shared init:** Stage1의 **CADS-사전학습 backbone**을 공통 시작점으로 사용(논문의 필수 조건).
2. **Gradient conflict 추정:** 각 전문가 데이터셋(TotalSeg / VerSe / CTSpine1K / RibSeg v2 /
   CTPelvic1K / VSDFullBody / MUG500+ / Spine-Mets)에서 소량 배치의 gradient 벡터 계산.
3. **PCA conflict-axis 분할:** gradient 벡터들에 PCA → 주 충돌축에 투영 → 부호 반대(충돌) 데이터셋을
   다른 파티션으로. 예상 결과: **척추 라벨 규칙 충돌**(CTSpine1K↔VerSe↔Spine-Mets) 분리, 부위가 겹치지
   않는 것(발 VSD ↔ 두개 MUG500)은 같은 그룹으로 묶여도 무방.
4. **파티션별 독립 fine-tune → 병합:** K개 nnU-Net을 각자 학습 후,
   **case/voxel 수 가중 평균**으로 1개 모델 병합("token-weighted"의 분할 문제 대응물).

### 3D 분할 이식 시 필수 적응 (중요)
- **병합 가능하려면 출력 head가 전 파티션 동일**해야 함 → 모든 파티션 모델은 **통합 taxonomy 전체 클래스**를
  출력. 각 파티션은 자기 데이터에 있는 라벨만 **partial-label loss**로 학습.
- **nnU-Net은 Instance Norm**(running stat 없음) → model-soup류의 BN 통계 병합 문제 회피 → **병합에 유리**.
- **gradient PCA 비용:** 3D U-Net 전체 파라미터 gradient는 큼 → **디코더 마지막 몇 층 / 인코더 일부만** 쓰거나
  gradient sketch(랜덤투영)로 저차원화.
- **병합은 shared basin 가정:** 파티션 fine-tune은 CADS-init에서 **짧고 낮은 LR**(scratch 아님)로 → MERIT의
  variance-reduction 이론 성립 조건과 우리 fine-tune 성격이 일치.
- **병합 방식 비교(ablation):** MERIT의 데이터량 가중 평균을 기본으로, **TIES-merging**(부호충돌 해소)·
  **task arithmetic**을 대조군으로.

### 권장 실험 순서
1. **Baseline A:** CADS-init에서 전 데이터셋 joint pooling + dataset-balanced random sampling(partial-label).
2. **MERIT B:** 위 4단계(K=2~3 파티션부터).
3. A vs B를 통합 taxonomy per-region Dice로 비교. 특히 **경계 부위(무릎/발목/척추 이행부)** 개선 여부 확인.

## 5. 출처
- CADS: https://arxiv.org/abs/2507.22953 · https://github.com/murong-xu/CADS · https://huggingface.co/datasets/mrmrx/CADS-dataset
- TotalSegmentator: https://pubs.rsna.org/doi/full/10.1148/ryai.230024 · https://github.com/wasserth/TotalSegmentator
- VerSe / CTSpine1K: https://github.com/MIRACLE-Center/CTSpine1K
- RibSeg v2: https://arxiv.org/abs/2210.09309 · https://github.com/HINTLab/RibSeg
- CTPelvic1K: https://github.com/MIRACLE-Center/CTPelvic1K
- VSDFullBody: https://www.nature.com/articles/s41597-023-02669-z · https://zenodo.org/records/8316967
- MUG500+: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8591340/
- Spine-Mets-CT (Stanford): https://www.medrxiv.org/content/10.1101/2024.10.14.24314447v1 · https://github.com/rouge1616/Spine-Mets-CT-SEG · https://doi.org/10.7937/kh36-ds04
- autoPET: https://autopet.grand-challenge.org/
- 부분라벨 학습 참고: DoDNet, CLIP-driven Universal Model, MultiTalent
- MERIT: https://arxiv.org/pdf/2606.01717 · https://github.com/naver-ai/merit
- 병합 기법 참고: TIES-merging, Task Arithmetic, Model Soups, CABS
