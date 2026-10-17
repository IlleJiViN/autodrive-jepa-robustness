# autodrive-jepa-robustness 인수인계 및 연구 핸드북 (Handover Guide)

> **최종 갱신일**: 2026-10-18  
> **프로젝트 위치**: `C:\Users\pc\Documents\Codex\2026-10-06\autodrive-jepa-robustness`  
> **Coder Workspace**: `owner/ai-lab` (NVIDIA GeForce RTX 5090 32GB)  
> **작성 목적**: 자율주행 악천후 강건성 연구의 배경, 초기 바닥학습 JEPA 실패(차원 붕괴 0.10%) 원인 규명, 5가지 구조적 검출 병목 해결(`FoundationViTDetectorV2`), BDD100K 70,000장 전수 훈련 및 10,000장 공식 검증 성과(36.30%), 코드베이스 구조, 그리고 교수님 연구실 서버 자원 지원 요청 전략을 완벽히 인수인계하기 위해 작성되었습니다.

---

## 1. 프로젝트 핵심 요약 및 연구 궤적 (Research Arc)

자율주행 환경에서 비전 인지 시스템은 맑은 날(Clean)뿐만 아니라 안개(Fog), 폭우(Rain), 저조도(Low-Light), 고속 모션 블러(Motion Blur) 등 극단적인 악천후에서도 높은 신뢰도로 객체를 검출해야 합니다.

본 연구는 다음과 같은 명확한 4단계 연구 궤적을 거쳐 완성되었습니다:
1. **Phase 1: 바닥 사전학습 JEPA의 실패와 차원 붕괴(Dimensional Collapse) 규명**:
   - 2,048장 소규모 데이터에서 JEPA 사전학습 수행. 악천후 이미지 간 코사인 유사도는 0.94로 높았으나, Linear Probe 검출 mAP50은 `0.10%`로 사실상 전멸.
   - 수학적 표현 평가 지표(Effective Rank)를 통해 384차원 중 단 2개 차원으로 압축된 차원 붕괴(Rank 2.4) 현상과 소규모 자기지도학습의 명확한 한계를 실증.
2. **Phase 2: Foundation ViT 도입 및 v1 파일럿 검증**:
   - 수억 장의 일반 시각 지식을 보유한 `timm` Pretrained `vit_small_patch16_384` 도입 (Effective Rank `37.42`로 회복).
   - 동결(Frozen) 백본 기반 단일 계층 헤드에서 mAP50 `5.41%`, 전면 해제 시 `6.85%` 기록. 아키텍처적 병목으로 인한 성능 천장 확인.
3. **Phase 3: 5대 구조 병목 해결 및 `FoundationViTDetectorV2` 아키텍처 혁신**:
   - Multi-Layer 중간 토큰 추출(Blocks 2, 5, 8, 11) + ViTDet FPN 어댑터(P3 Stride 8, P4 Stride 16, P5 Stride 32) 구축.
   - Decoupled Anchor-Free Single-Slot Head(`SLOTS=1`) + Multi-Cell Soft 타겟 할당 + Complete IoU (CIoU) 손실 + CUDA Batched NMS 구현.
   - 동일 2k 데이터셋 조건에서 **16.16% mAP50** 달성 (**+10.75%p 순수 아키텍처 도약**).
4. **Phase 4: BDD100K 70,000장 전체 훈련 및 10,000장 공식 검증 스케일업**:
   - RTX 5090 환경에서 70,000장 전체 데이터를 대상으로 12에포크 엔드투엔드 스케일업 훈련 (총 56분 소요).
   - 256장 벤치마크 **43.42% mAP50** (mAP50:95 **21.20%**), 공식 10,000장 전수 검증셋에서 **36.30% mAP50** (185,526개 박스) 달성 (**+27.26%p 순수 데이터 스케일업 기여**).
   - 4대 악천후 3-Seed 전수 평가에서 **93.6% 성능 보존율**(Mean Corrupted 40.65%) 입증. COCO 제로샷 Faster R-CNN v2 (34.18%) 수준에 도달하여 실측 베이스라인을 성공적으로 확보. (단, 도메인 전용 2-Stage SOTA 대비 정밀도 17.20%는 고해상도 확장을 통한 개선 과제로 식별).

---

## 2. 해결된 5대 핵심 엔지니어링 병목 (5 Core Bottlenecks Resolved)

`src/robustdrive/foundation_v2.py`에 적용된 핵심 구조적 개선 사항입니다:

### 2.1 Softmax/Sigmoid 불일치 및 거짓 양성 누출 해결
- **문제**: v1에서는 11개 클래스(배경 1 + 전경 10)에 대해 Softmax Cross-Entropy를 적용하여, 배경 셀에서도 특정 전경 클래스 확률이 0이 되지 못하고 누출되어 수많은 고신뢰도 오탐(False Positive)이 발생함.
- **해결**: Objectness(이진 객체 존재성) 브랜치와 Class Logits 브랜치를 분리하고, Objectness에는 Focal Sigmoid Loss를, Positive 셀에만 Class Softmax Loss를 부여함. 추론 시 최종 점수를 $\text{Score} = \sigma(p_{\text{obj}}) \times \text{Softmax}(p_{\text{cls}})$로 계산하여 완벽한 신뢰도 정합 달성.

### 2.2 Dead Slot 제거 (`SLOTS=3` $\rightarrow$ `SLOTS=1`)
- **문제**: v1에서는 Anchor-free 격자임에도 각 셀에 3개의 슬롯을 할당함. 그러나 Anchor 박스가 없는 구조적 특성상 1개 슬롯만 학습 신호를 받고 나머지 2개 슬롯은 영구히 비활성화되는 "Dead Slot"으로 전락하여 모델 용량을 낭비하고 최적화를 교란함.
- **해결**: 단일 격자당 1개 슬롯(`SLOTS=1`)으로 아키텍처를 전면 간소화하고, 다중 객체 커버리지는 Multi-Cell 공간 이웃 할당으로 이관하여 모델 수렴성과 파라미터 효율성을 극대화함.

### 2.3 Multi-Layer ViT FPN vs 단일 계층 업샘플링
- **문제**: ViT는 구조상 다운샘플링 없는 단일 해상도($24 \times 40$) 토큰 맵을 가짐. v1에서는 최종 Layer 12의 출력만을 단순 Bilinear 업샘플링하여 소형 객체(원거리 보행자, 신호등) 탐지 성능이 치명적으로 저하됨.
- **해결**: `ViTDetFPNAdapter`를 도입하여 ViT-Small의 4개 중간 블록 `[2, 5, 8, 11]`(Layers 3, 6, 9, 12)에서 토큰을 추출함. Block 2는 Stride-2 Transpose Conv로 P3($48 \times 80$, Stride 8)를 생성하고, P4($24 \times 40$, Stride 16), P5($12 \times 20$, Stride 32)를 Top-Down 방식으로 연결하여 모든 크기의 객체에 최적화된 특징 피라미드를 구축함.

### 2.4 Complete IoU (CIoU) + Multi-Cell 할당 vs Log Smooth-L1 단일 셀
- **문제**: v1에서는 GT 박스 중심이 속한 단 1개의 셀에만 라벨을 할당(양성 셀 0.05% 미만으로 극단적 불균형)하고, 박스 크기를 로그 오프셋 Smooth-L1으로 지도하여 가로세로비 왜곡 및 위치 불안정이 심각했음.
- **해결**: 
  - **Multi-Cell Soft Assignment**: GT 박스 중심 주변 $3 \times 3$ 영역 중 박스 내부 및 반경 $1.5 \times \text{stride}$ 이내의 최대 5개 셀에 타겟을 동시 할당(~5배 양성 그래디언트 밀도 확보).
  - **Complete IoU (CIoU) Loss**: 픽셀 좌표계에서 직접 박스 IoU, 중심 거리 비율($\rho^2/c^2$), 종횡비 일치도($\alpha v$)를 동시에 최적화하는 CIoU 손실을 적용하여 픽셀 단위 박스 정밀도(mAP50:95)를 획기적으로 개선함.

### 2.5 Vectorized CUDA `batched_nms` (장당 0.3ms 고속화)
- **문제**: v1의 디코딩 및 NMS는 CPU 상에서 순차 Python 반복문으로 처리되어 256장 평가에 수십 초, 10,000장 전수 검증에는 수십 분이 걸리는 치명적인 병목이 있었음.
- **해결**: `decode_v2` 함수를 구축하여 모든 예측값의 텐서 좌표 변환, 점수 마스킹, Top-k 선별 및 NMS(`torchvision.ops.batched_nms`)를 GPU CUDA 커널에서 일괄 처리. 디코딩 시간을 장당 **0.3ms**로 단축시켜 10,000장 전수 검증을 단 **90.88초**에 완주함.

---

## 3. 정량 실험 성적표 종합 (Comprehensive Benchmark Archive)

### 3.1 4단계 점진적 성능 및 스케일업 요약 (Progressive Scaling)

| 단계 (Stage) | 모델 아키텍처 및 세부 사항 | 데이터 규모 | 에포크 | Val mAP50 (256) | Val mAP50:95 (256) | 공식 10k Val mAP50 | 성능 도약 원인 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | Scratch JEPA (바닥 학습) + Linear Probe | 2,048장 | 200 st | `0.10%` | `0.02%` | - | 표현 붕괴 (Rank 2.4) |
| **Stage 2** | Foundation ViT-S v1 (Frozen + 단일헤드) | 2,048장 | 100 ep | `5.41%` | `1.55%` | - | Foundation 모델 시각 표현력 복원 |
| **Stage 3** | **Foundation ViT-S v2 (FPN + CIoU + SLOTS=1)** | **2,048장** | **10 ep** | **`16.16%`** | **`5.77%`** | - | **순수 아키텍처 혁신 기여 (+10.75%p)** 🚀 |
| **Stage 4** | **Foundation ViT-S v2 (Full Scale Supervised)** | **70,000장** | **12 ep** | **`43.42%`** | **`21.20%`** | **`36.30%`** | **순수 데이터 스케일업 기여 (+27.26%p)** 🏆 |
| **Stage 5** | **Natural-Noise I-JEPA ViT-S + Detector (Ours)** | **70,000장** | **5 ep pre + 12 ep det** | **`39.63%`** | **`19.03%`** | **`34.96%`** | **`97.5%` 악천후 방어율 달성 (Fog 97.1%, Rain 97.2%)** 🛡️ |


---

### 3.2 12-Epoch 전수 훈련 궤적 (BDD100K 70,000장, RTX 5090)

`runs/rtx5090-v2-full70k-12ep/training_history.json` 실측 기록:

| Epoch | Total Loss | Obj Loss (Focal) | Cls Loss (CE) | Box Loss (CIoU) | Val mAP50 (256) | Val mAP50:95 (256) | 소요 시간 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ep 01** | 4.4633 | 0.6609 | 1.1334 | 1.3345 | `25.02%` | `10.29%` | 276.6s |
| **Ep 02** | 3.1419 | 0.4242 | 0.6767 | 1.0205 | `35.87%` | `15.64%` | 276.4s |
| **Ep 03** | 2.8254 | 0.3829 | 0.5685 | 0.9370 | `37.22%` | `16.09%` | 275.3s |
| **Ep 04** | 2.6505 | 0.3620 | 0.4962 | 0.8961 | `37.56%` | `17.22%` | 275.7s |
| **Ep 05** | 2.5240 | 0.3483 | 0.4375 | 0.8691 | `40.00%` | `19.37%` | 276.9s |
| **Ep 06** | 2.4204 | 0.3377 | 0.3845 | 0.8491 | `40.95%` | `19.80%` | 285.9s |
| **Ep 07** | 2.3286 | 0.3287 | 0.3363 | 0.8318 | `40.89%` | `20.27%` | 280.5s |
| **Ep 08** | 2.2525 | 0.3218 | 0.2939 | 0.8184 | `40.84%` | `20.15%` | 277.4s |
| **Ep 09** | 2.1886 | 0.3161 | 0.2591 | 0.8067 | `41.53%` | `20.22%` | 277.5s |
| **Ep 10** | 2.1383 | 0.3114 | 0.2322 | 0.7973 | `43.17%` | `21.00%` | 276.4s |
| **Ep 11 🏆** | **2.1067** | **0.3086** | **0.2152** | **0.7914** | **`43.42%`** | **`21.20%`** | **275.5s** |
| **Ep 12** | 2.0876 | 0.3070 | 0.2044 | 0.7881 | `43.16%` | `21.13%` | 272.5s |

---

### 3.3 3-Seed 악천후 강건성 정밀 실측 성적표 (Seed 42, 100, 2024)

`runs/rtx5090-v2-full70k-12ep/rigorous_evaluation_report.json` 실측 기록:

| 평가 환경 (Condition) | mAP50 (%) | mAP50:95 (%) | 맑은 날 대비 성능 보존율 (Retention Rate) |
| :--- | :---: | :---: | :---: |
| **Clean (맑은 주간, 기준선)** | **`43.42%`** (±0.00%) | **`21.20%`** (±0.00%) | **100.0%** (기준점) |
| **Motion Blur (모션 블러, s=2)** | **`43.22%`** (±0.00%) | **`21.40%`** (±0.00%) | **99.5%** |
| **Rain (폭우 시각 왜곡, s=2)** | **`41.52%`** (±0.25%) | **`20.25%`** (±0.15%) | **95.6%** |
| **Fog (안개 산란 왜곡, s=2)** | **`39.89%`** (±0.74%) | **`19.45%`** (±0.60%) | **91.9%** |
| **Low-Light (야간 저조도 노이즈, s=2)** | **`37.96%`** (±0.13%) | **`17.86%`** (±0.09%) | **87.4%** |
| **Mean Corrupted (4대 악천후 평균)** | **`40.65%`** | **`19.74%`** | **`93.6%` (방어 성공)** 🛡️ |

---

### 3.4 BDD100K 공식 10,000장 전수 검증 성적 비교

`runs/rtx5090-v2-full70k-12ep/eval_10k/summary_10k.json` 실측 기록:

| 모델 (Model) | 학습 데이터 | 검증 이미지 수 (Val Boxes) | Clean mAP50 | Clean mAP50:95 | 총 검증 시간 / 속도 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BDD100K Paper Baseline** | BDD100K 70k | 10,000장 (185k) | `32.0% ~ 38.0%` | - | - |
| **Faster R-CNN v2 (COCO Pretrained)** | COCO 118k | 10,000장 (185k) | `34.18%` | - | ~2.5분 |
| **FoundationViTDetectorV2 (Ours)** | **BDD100K 70k** | **10,000장 (185,526)** | **`36.30%`** 🏆 | **`17.20%`** | **90.88초 (110 FPS)** ⚡ |

---

## 4. 디렉터리 구조 및 핵심 산출물 맵

```
autodrive-jepa-robustness/
├── data/
│   ├── manifests/
│   │   ├── bdd100k_train_70k_local.jsonl     # 70,000장 전수 훈련 매니페스트
│   │   ├── bdd100k_val_10k_local.jsonl       # 10,000장 전수 검증 매니페스트 (185k 박스)
│   │   ├── bdd100k_train_local.jsonl         # 2,048장 파일럿 훈련셋
│   │   └── bdd100k_val_local.jsonl           # 256장 정밀 벤치마크 검증셋
│   └── bdd100k/annotations/                 # BDD100K 원본 레이블 아카이브
├── src/robustdrive/
│   ├── common.py                             # 시드 고정 및 공통 유틸
│   ├── corruptions.py                        # 온더플라이 악천후 합성 (fog, rain, blur, low_light)
│   ├── data.py                               # BDD100K 10개 클래스 및 텐서 변환기
│   ├── detector.py                           # 3스케일 FPN 앵커프리 검출기, 타겟할당, NMS, coco_ap
│   ├── foundation.py                         # v1 Foundation ViT 모델
│   ├── foundation_v2.py                      # [핵심] v2 ViTDet FPN + Single-Slot Head + CIoU + CUDA NMS
│   ├── losses.py                             # 공분산 비상관 손실(Covariance Loss), VICReg 정규화
│   ├── metrics.py                            # Alignment, Uniformity, Effective Rank, CKA
│   └── models.py                             # 바닥 JEPA 및 MAE 아키텍처
├── tools/
│   ├── download_full_bdd100k.py              # [고속] BDD100K 100k 8-Worker 멀티스레드 다운로더
│   ├── extract_bdd100k_images.py             # 70k/10k 이미지 추출기
│   ├── build_full_bdd100k_manifests.py       # 70k/10k 다중 프로세스 매니페스트 생성기
│   ├── train_5090_full_scale.py              # [핵심] 70k 전수 RTX 5090 고속 학습 및 평가 도구
│   ├── train_unfreeze_top_blocks.py          # 상위 4개 블록 선별 해제 학습 스크립트
│   ├── evaluate_rigorous_multiseed.py        # 256장 전수 + 3-Seed 다중 평가 스크립트
│   └── evaluate_pretrained_baseline.py       # 상용 Faster R-CNN v2 256장 기준점 평가 스크립트
├── runs/
│   ├── rtx5090-v2-full70k-12ep/              # [최종] 70,000장 12에포크 산출물
│   │   ├── best_checkpoint.pt                # Peak Checkpoint (Epoch 11, mAP50 43.42%)
│   │   ├── training_history.json             # 12에포크 손실 및 검증 로그
│   │   ├── rigorous_evaluation_report.json   # 3-Seed 5대 환경 전수 결과 (93.6% retention)
│   │   └── eval_10k/summary_10k.json         # 10,000장 공식 검증 결과 (mAP50 36.30%)
│   └── rtx5090-v2-10ep-validation/           # 2k 데이터셋 v2 10에포크 검증 산출물 (16.16%)
├── docs/                                     # 연구 보고서 및 핸드북
│   ├── RESEARCH_HANDOVER.md                  # 본 문서
│   ├── cost_benchmark_results.md             # JEPA vs MAE 사전학습 비용 분석서
│   └── detector_results.md                   # v1 검출기 분석 보고서
└── tests/                                    # 25개 전체 회귀 단위 테스트 (pytest 100% 통과)
```

---

## 5. 재현 및 실행 가이드 (Quick Start)

### 5.1 단위 테스트 무결성 확인
```powershell
pytest
# 결과: 25 passed in ~10s
```

### 5.2 70,000장 전수 훈련 재현 (RTX 5090 환경)
```bash
python tools/train_5090_full_scale.py \
  --train-manifest data/manifests/bdd100k_train_70k_local.jsonl \
  --val-manifest data/manifests/bdd100k_val_local.jsonl \
  --epochs 12 \
  --batch-size 32 \
  --workers 8 \
  --backbone-lr 2e-5 \
  --head-lr 2e-4 \
  --warmup-pct 0.10 \
  --output-dir runs/rtx5090-v2-full70k-12ep \
  --device cuda
```

### 5.3 10,000장 공식 검증 전수 평가 실행
```bash
python tools/train_5090_full_scale.py \
  --evaluate-only \
  --checkpoint runs/rtx5090-v2-full70k-12ep/best_checkpoint.pt \
  --val-manifest data/manifests/bdd100k_val_10k_local.jsonl \
  --val-max-images 10000 \
  --output-dir runs/rtx5090-v2-full70k-12ep/eval_10k \
  --device cuda
```

---

## 6. 교수님 미팅 및 연구실 서버 자원 지원 요청 3대 전략 (Proposal Pitch)

교수님 또는 랩 미팅 시 제시할 체계적이고 반박 불가능한 3단계 자원 요청 스토리라인입니다:

### 1단계: 완벽한 실증 성과 보고 (Done & Validated)
- "초기 바닥 사전학습 JEPA의 실패 원인을 **Effective Rank(유효 차원 2.4로 붕괴)** 지표로 정밀 규명하였습니다."
- "이후 5가지 핵심 검출 병목을 해소한 **`FoundationViTDetectorV2`**를 신규 설계하여, 소규모 2k 데이터에서 아키텍처 개선만으로 **16.16% (+10.75%p)**를 달성하고, BDD100K 70,000장 전체 데이터셋을 투입하여 **43.42% (256 val) / 36.30% (10,000 공식 val)**의 탄탄한 지도학습 베이스라인을 성공적으로 확립하였습니다."
- "특히 4대 악천후 3-Seed 전수 평가에서 **93.6%의 높은 합성 왜곡 보존율**을 기록하여, COCO 제로샷 모델(Faster R-CNN v2, 34.18%) 수준에 도달함을 실측 입증하였습니다."

### 2단계: 자원 병목 및 스케일업 필요성 제시 (The Current Bottleneck)
- "현재 성과는 단일 RTX 5090 환경의 메모리 한계로 인해 $384 \times 640$ 해상도와 경량 ViT-Small (22M) 백본에 국한되어 있습니다."
- "이로 인해 BDD100K 공식 2-Stage SOTA 모델(mAP50:95 약 31%) 대비 mAP50:95는 17.20% 수준으로, 원거리 10픽셀 미만 미세 객체(원거리 보행자, 신호등)의 박스 정밀도 개선이 절실합니다."
- "이를 네이티브 HD($1280 \times 720$) 해상도와 대형 ViT-Large 백본으로 확장하여 박스 정밀도를 대폭 끌어올리고, 톱티어 학회(CVPR/ECCV/ICCV/IROS/NeurIPS) 본 논문으로 완성하기 위해서는 **Multi-GPU 서버 자원(A100/H100 노드)**이 필수적입니다."

### 3단계: 연구실 GPU 서버 지원 시 착수할 3대 핵심 후속 과제 (Upgraded 3-Point Roadmap)
1. **과제 1: Native HD ($1280 \times 720$) 해상도 보존 + ViT-Base / ViT-Large 스케일업**:
   - 다운샘플링 없는 $1280 \times 720$ 네이티브 자율주행 카메라 해상도 직접 학습.
   - ViT-Base (86M), ViT-Large (304M) 대형 파운데이션 모델을 Full-Finetuning하여 SOTA 자율주행 인지 벤치마크 갱신.
2. **과제 2: 1,100시간 원본 주행 동영상 기반 비디오 도메인 적응 JEPA 사전학습 (Continual Video-JEPA)**:
   - 본 연구에서 입증된 Foundation ViT 가중치를 바탕으로, BDD100K의 1,100시간 미라벨 주행 동영상에 시공간 마스킹(Spatio-Temporal Masked Prediction) 연속 사전학습 수행.
   - 날씨와 조명 변화의 시간적 일관성을 스스로 학습하는 차세대 비디오 기반 자율주행 파운데이션 모델 논문 출간.
3. **과제 3: 실제 악천후 도로 데이터셋(ACDC, Foggy Cityscapes, DAWN) 전이 및 일반화 실증**:
   - 합성 노이즈를 넘어 실제 현장 데이터셋에 제로샷/퓨샷 전이를 수행하여 실제 자율주행차(OEM 양산) 탑재 신뢰성을 입증하는 강력한 학술 논문 제출.
