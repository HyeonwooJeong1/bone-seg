# 평가지표 근거 정리 (문헌 기반)

"어떤 지표를, 왜" — 표준 프레임워크 + 각 뼈 벤치마크가 실제로 쓰는 것에 근거.

---

## 0. 근거 프레임워크
- **Metrics Reloaded** (Maier-Hein, Reinke et al., *Nature Methods* 2024) — 문제유형(semantic seg / instance seg / detection)에 **맞는 지표를 고르라**는 표준. 참조구현: MONAI `MetricsReloaded`.
  핵심 교훈(우리에 직접 적용):
  - **overlap(DSC) 하나로는 부족** — 특히 **작은/얇은 구조**에서 DSC가 불안정·경계 민감도 낮음 → **boundary 지표(NSD)** 병행.
  - **HD는 이상치에 매우 민감** → **HD95** 사용(원시 HD 지양).
  - **여러 인스턴스를 구별하는 과제(척추·늑골)** 는 semantic DSC만으로 부족 → **instance/detection 지표** 필요.
- **Panoptica** (Kofler et al., arXiv 2312.02608) — 3D **instance-wise** 표준 구현: **PQ = RQ×SQ**(+ ASSD, clDice). VerSe·SPINEPS가 사용. TP = instance IoU ≥ 0.5.

## 1. 뼈 벤치마크가 실제 쓰는 지표
| 벤치마크 | 지표 | 시사점(우리) |
|---|---|---|
| **VerSe**(척추, MedIA'21) | **Dice + Hausdorff**(seg) / **Identification Rate** + **mean localization error(centroid 거리 mm)**(labeling); **부위별(경추/흉추상·하/요추) 보고** | 척추=**seg + 식별율 + centroid 오차**, **부위 층화** 표준 |
| **RibSeg v2**(늑골) | **Label-Dice**(24늑골, 빈GT 제외) + **Label-Accuracy**(늑골 Recall>0.7이면 정검출) — **All/First/Intermediate/Twelfth 층화** | 늑골=**per-rib Dice + 검출율**, **위치별 층화** |
| **TotalSegmentator / Skellytour**(전신 뼈) | **DSC + NSD@3mm** (Skellytour DSC .94/NSD .99) | 전신 뼈의 **경계 표준 = NSD@τ(기본 3mm)** |
| **SPINEPS**(척추 instance) | **RQ/SQ/PQ**(panoptica), TP@IoU≥0.5 | 척추/늑골 instance는 **Panoptic Quality** |

## 2. 우리가 쓸 지표 (근거와 함께)

### 2.1 Semantic (per-class, 모든 뼈)
- **DSC** — 표준 overlap. per-class + 부위 매크로.
- **NSD@τ** (Normalized Surface Dice) — **경계 정확도**. τ=**3mm 기본**(TotalSeg/Skellytour 관례), **얇은 뼈는 더 작은 τ**(예 늑골·비골 1.5–2mm)로 별도. *(근거: TotalSeg/Skellytour + Metrics Reloaded의 boundary 권고)*
- **HD95, ASSD** — 표면 거리(이상치 강건). *(Metrics Reloaded: HD 대신 HD95)*
- **RVE**(상대 부피오차) — 임상 부피 활용.

### 2.2 Instance / 식별 (복잡한 뼈: 개별 척추·늑골)
- **PQ = RQ×SQ** (panoptica식, TP@IoU≥0.5) — 검출(RQ)과 분할품질(SQ) 분리. *(근거: VerSe/SPINEPS/panoptica)*
- **Identification Rate** — GT 인스턴스 중 **같은 라벨로 IoU≥0.5 검출** 비율. *(근거: VerSe)*
- **Localization error** — 매칭된 인스턴스 **centroid Euclidean 거리(mm)** 평균. *(근거: VerSe)*
- **Rib Label-Accuracy** — 늑골 Recall>0.7이면 정검출(All/First/Intermediate/Twelfth 층화). *(근거: RibSeg v2)*
- **clDice**(Centerline Dice) — **얇고 긴** 늑골에 적합. *(근거: panoptica/tubular)*

### 2.3 오류 구조 (복잡한 뼈 심층 — RQ4)
- **인접 혼동행렬**: GT 라벨 voxel이 어떤 pred 라벨로 새는지(off-diagonal 질량). 척추 **off-by-one 열거오류**, 인접 늑골 혼동 포착.
- **좌/우 swap율**: L/R 쌍에서 예측 질량이 반대편으로 간 비율.
- **이행부 지표**: C7–T1, T12–L1, L5–S1 인스턴스의 **id.rate·열거오류** 별도 집계.
- **난이도 층 × 지표**: 얇음/다중인접/좌우/접촉면/이행부/변이 각각에 위 지표.

## 3. 학습 중 관찰값 (근거: nnU-Net 표준 + 우리 진단)
- train/val loss, **pseudo-Dice(EMA)**, per-class val Dice, LR, epoch time (nnU-Net 로그).
- **병합 진단**(MERIT식): merging gain, linear-mode-connectivity loss barrier, θ⁰ displacement 비율, perturbation robustness.
- **conflict 산출물**: 코사인행렬 C, PCA 임베딩, 분할 vs 해부학 일치도(ARI/NMI).

## 4. 구현 방침
- **우리 자체 구현**(numpy+scipy, 로컬 GPU-free 테스트 가능): DSC·NSD·ASSD·HD95(있음) + instance(PQ/RQ/SQ·id-rate·localization·confusion·L/R swap).
- **외부 교차검증**(서버, 선택): **panoptica**·**MONAI MetricsReloaded**로 우리 수치 sanity-check.
- 코드: `ai_bone/eval/metrics.py`(semantic) + `ai_bone/eval/instance_metrics.py`(instance/식별) + `evaluate.py`(집계·층화).

## 출처
- Metrics Reloaded: https://www.nature.com/articles/s41592-023-02151-z · pitfalls https://www.nature.com/articles/s41592-023-02150-0 · impl https://github.com/Project-MONAI/MetricsReloaded
- VerSe (MedIA 2021) https://www.sciencedirect.com/science/article/abs/pii/S1361841521002127
- RibSeg v2 https://arxiv.org/abs/2210.09309 · Panoptica https://arxiv.org/abs/2312.02608 · SPINEPS https://arxiv.org/abs/2402.16368
- TotalSegmentator https://pubs.rsna.org/doi/full/10.1148/ryai.230024 · Skellytour https://pubs.rsna.org/doi/10.1148/ryai.240050
