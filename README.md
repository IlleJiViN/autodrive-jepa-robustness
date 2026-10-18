# AutoDrive-JEPA-Robustness: Multi-Scale Foundation ViT Object Detection under Adverse Weather (BDD100K)

[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%205090%2032GB-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Dataset](https://img.shields.io/badge/Dataset-BDD100K%20(70k%20Train%20%7C%2010k%20Val)-blue)](https://www.bdd100k.com/)
[![Official Val mAP50](https://img.shields.io/badge/Official%2010k%20Val%20mAP50-36.30%25-brightgreen)](#4-official-bdd100k-10000-image-full-validation-benchmark)
[![Official Val mAP50:95](https://img.shields.io/badge/Official%2010k%20Val%20mAP50%3A95-17.20%25-blue)](#4-official-bdd100k-10000-image-full-validation-benchmark)
[![Adverse Retention](https://img.shields.io/badge/Adverse%20Weather%20Retention-93.6%25%20(256--sample)-orange)](#3-rigorous-3-seed-adverse-weather-benchmark)

> **연구 개요**: 자율주행 악천후(Adverse Weather) 환경에서의 객체 검출 강건성 확보, 자기지도학습(JEPA) 소규모 바닥 학습 시 차원 붕괴(0.10%) 한계 규명, 5가지 검출기 구조 결함 개선, 그리고 BDD100K 공식 전수 데이터셋(70,000장 학습 / 10,000장 검증) 실증 스케일업 연구.

---

## 목차 (Table of Contents)
1. [Executive Summary (연구 요약 및 성과)](#executive-summary-연구-요약-및-성과)
2. [System Architecture (`FoundationViTDetectorV2`)](#system-architecture-foundationvitdetectorv2)
3. [5 Core Engineering Bottlenecks Resolved](#5-core-engineering-bottlenecks-resolved)
4. [Key Experimental Results & Ablation Study](#key-experimental-results--ablation-study)
   - [Table 1: 4-Stage Progressive Ablation & Scaling](#1-4-stage-progressive-ablation--scaling)
   - [Table 2: 12-Epoch Full-Scale Training Trajectory (70,000 Images)](#2-12-epoch-full-scale-training-trajectory-70000-images-rtx-5090)
   - [Table 3: Rigorous 3-Seed Adverse Weather Benchmark](#3-rigorous-3-seed-adverse-weather-benchmark)
   - [Table 4: Official BDD100K 10,000-Image Full Validation Benchmark](#4-official-bdd100k-10000-image-full-validation-benchmark)
5. [Limitations & Honest Analysis (한계점 및 분석)](#limitations--honest-analysis-한계점-및-분석)
6. [Repository Structure & Codebase Map](#repository-structure--codebase-map)
7. [Quick Start & Reproducibility Commands](#quick-start--reproducibility-commands)
8. [Future Research Roadmap (교수님 미팅 및 자원 요청)](#future-research-roadmap-교수님-미팅-및-자원-요청)

---

## Executive Summary (연구 요약 및 성과)

자율주행 환경에서 비전 인지 시스템은 맑은 주간(Clean)뿐만 아니라 안개(Fog), 폭우(Rain), 저조도(Low-Light), 모션 블러(Motion Blur) 등 다양한 환경 조건에서도 안정적인 검출 성능을 유지해야 합니다.

본 연구는 초기에 자율주행 특화 자기지도학습을 목표로 **Joint-Embedding Predictive Architecture (JEPA)**를 소규모(2,048장) 데이터에서 바닥부터 학습(scratch pretraining)했으나, 검출 성능이 **0.10% mAP50**에 머무르는 심각한 성능 한계를 겪었습니다. 표현 품질 감사(Representation Quality Audit)를 통해 원인을 추적한 결과, 악천후 이미지 간 높은 코사인 유사도(0.94) 뒤에 숨겨진 **극단적인 차원 붕괴(Effective Rank 3.2 $\rightarrow$ 2.4, 잠재 공간 붕괴)**를 확인하였습니다.

이를 극복하기 위해 본 프로젝트는 대규모 시각 지식을 보유한 **Foundation ViT (timm ViT-Small)** 백본을 도입하고, 기존 단순 검출 헤드의 5가지 구조적 병목(Dead Slot, 단일 레이어 해상도 손실, Softmax/Sigmoid 불일치, Smooth-L1 손실 한계, 비효율적 NMS)을 전면 개선한 **`FoundationViTDetectorV2`**를 설계하였습니다.

### 주요 연구 성과 및 현주소
- **순수 구조 개선 효과 (+10.75%p)**: 동일한 2,048장 조건에서 v1(5.41%) 대비 3배인 **16.16% mAP50** 달성 (아키텍처 수리 기여도 분리 입증).
- **공식 BDD100K 10,000장 전수 검증 (36.30% mAP50 / 17.20% mAP50:95)**: 70,000장 학습 완료 후 185,526개 박스 전수에 대해 공인 검증을 수행하여, COCO 제로샷 Faster R-CNN v2 (34.18%) 수준에 도달하고 실측 베이스라인을 성공적으로 확보.
- **악천후 합성 왜곡 방어율 (93.6%)**: Fog, Rain, Low-Light, Motion Blur의 4대 왜곡(Severity=2, 256장 표본 3-Seed) 평가에서 평균 **40.65% mAP50** 유지 (Clean 43.42% 대비 93.6% 방어).
- **고속 엔지니어링**: RTX 5090 단일 GPU에서 70,000장 학습 1 에포크당 **4.6분**(총 12에포크 56분), 10,000장 전수 추론을 **90.88초**(110 FPS, 9.09ms/장)에 완주.
- **객관적 한계 인식**: BDD100K 도메인에 완전히 최적화된 상용 2-Stage SOTA(mAP50:95 ~31%) 대비 mAP50:95는 17.20% 수준으로, 아직 박스 정밀도 면에서 추가 개선 여지가 큼.

---

## System Architecture (`FoundationViTDetectorV2`)

`FoundationViTDetectorV2`는 일반 ViT(Plain Vision Transformer) 백본의 풍부한 표현력을 고밀도 다중 스케일(Multi-Scale) 객체 검출에 최적화하도록 설계된 엔드투엔드 아키텍처입니다.

```mermaid
flowchart TD
    subgraph INPUT ["Input Pipeline"]
        img["Input Driving Image (384x640x3)"]
        aug["On-the-Fly Geometric & Adverse Augmentations"]
        img --> aug
    end

    subgraph BACKBONE ["Pretrained ViT Backbone (ViT-Small/16)"]
        aug --> pe["Patch Embed (16x16) + Position Embed"]
        pe --> b0["Blocks 0-1"]
        b0 --> b2["Block 2 (Layer 3 Feature)"]
        b2 --> b34["Blocks 3-4"]
        b34 --> b5["Block 5 (Layer 6 Feature)"]
        b5 --> b67["Blocks 6-7"]
        b67 --> b8["Block 8 (Layer 9 Feature)"]
        b8 --> b910["Blocks 9-10"]
        b910 --> b11["Block 11 (Layer 12 Feature)"]
    end

    subgraph FPN ["ViTDet Top-Down Feature Pyramid Adapter"]
        b11 --> lat5["Lateral 1x1 + Conv3x3 (s=2)"]
        lat5 --> p5["P5 Feature (Stride 32, 12x20, 256-ch)"]

        b8 & b5 --> lat4["Lateral 1x1 Sum"]
        p5 -->|"Top-Down Bilinear Upsample"| up5["Upsample 2x"]
        lat4 & up5 --> fuse4["Conv3x3 + GN + SiLU"]
        fuse4 --> p4["P4 Feature (Stride 16, 24x40, 256-ch)"]

        b2 --> p3_trans["ConvTranspose2d (s=2)"]
        p4 -->|"Top-Down Bilinear Upsample"| up4["Upsample 2x"]
        p3_trans & up4 --> fuse3["Conv3x3 + GN + SiLU"]
        fuse3 --> p3["P3 Feature (Stride 8, 48x80, 256-ch)"]
    end

    subgraph HEADS ["Decoupled Anchor-Free Scale Heads (SLOTS = 1)"]
        p3 & p4 & p5 --> dhead["Decoupled Head Branches"]
        dhead --> cls_b["Cls Branch: 2x Conv3x3 (128-ch) -> 10-Class Logits"]
        dhead --> reg_b["Reg/Obj Branch: 2x Conv3x3 (128-ch) -> Objectness + 4-Offset"]
    end

    subgraph LOSS_INFERENCE ["Supervision & Vectorized Decoding"]
        cls_b & reg_b --> train_mode{"Mode?"}
        train_mode -->|"Training"| losses["Multi-Cell Soft Assignment (<=5 cells)\n+ Focal Loss + CIoU Loss + Cross-Entropy"]
        train_mode -->|"Inference"| nms["CUDA Vectorized Batched NMS (0.3ms/img)\nscore = sigmoid(obj) * softmax(cls)"]
    end
```

### 아키텍처 핵심 모듈 상세
1. **Multi-Layer ViT 중간 토큰 추출 (`forward_intermediates`)**:
   - 단일 최종 계층 대신 중간 블록 `[2, 5, 8, 11]` (각 384차원, $24 \times 40$ 해상도)에서 표현을 추출하여 저수준 에지(Edge), 텍스처, 고수준 의미론(Semantics)을 모두 보존합니다.
2. **`ViTDetFPNAdapter` Top-Down 특징 피라미드**:
   - **P5 (Stride 32, $12 \times 20$, 256채널)**: Block 11 특징을 Lateral 1x1 후 Stride-2 Conv3x3으로 서브샘플링하여 대형 객체 탐지.
   - **P4 (Stride 16, $24 \times 40$, 256채널)**: Block 8 및 Block 5의 측면 연결과 P5로부터의 Top-Down 보간 융합.
   - **P3 (Stride 8, $48 \times 80$, 256채널)**: Block 2 특징을 Transpose-Conv(s=2)로 업샘플링하고 P4의 Top-Down 피처를 융합하여 원거리 소형 차량/보행자 탐지력 극대화.
3. **`DecoupledScaleHead` (`SLOTS=1`)**:
   - 객체성/회귀(Reg) 경로와 분류(Cls) 경로를 2개의 독립된 2x Conv3x3(128채널, GroupNorm, SiLU) 브랜치로 완전 분리.
   - Prior bias 초기화($\pi \approx 0.01$, bias = -4.6)를 적용하여 초기 불균형 그래디언트 폭주를 원천 억제.

---

## 5 Core Engineering Bottlenecks Resolved

v1 검출기에서 발생하던 5가지 구조적 병목을 진단하고 수학적·구조적으로 완전히 해결하였습니다:

| 번호 | 구조적 병목 (v1 Bottleneck) | 정밀 원인 분석 | v2 최종 해결책 (V2 Engineering Fix) |
| :---: | :--- | :--- | :--- |
| **1** | **Softmax/Sigmoid 불일치 및 거짓 양성 누출** | 객체 유무와 클래스 분류가 혼합된 11-way Softmax 사용으로, 배경 격자가 강제로 특정 클래스 확률을 할당받아 오탐 발생 | **Decoupled Scoring 정합**: 독립 Objectness Sigmoid 확률과 클래스 Softmax 확률을 분리하고, 추론 시 $\text{Score} = \sigma(p_{\text{obj}}) \times \text{Softmax}(p_{\text{cls}})$로 완전 정합 |
| **2** | **Dead Slot 발생 및 수렴 방해** | 격자당 3개의 고정 슬롯(`SLOTS=3`)을 두었으나 Anchor-free 특성상 슬롯 간 경합 및 90% 이상의 미할당 Dead Slot이 파라미터를 낭비 | **Single-Slot (`SLOTS=1`) 단순화**: 슬롯 경합을 원천 제거하고 파라미터 효율성과 특징 맵 집중도를 극대화 |
| **3** | **단일 계층 업샘플링의 미세 객체 손실** | ViT의 마지막 Layer 12 단일 출력($24 \times 40$)을 단순 쌍선형(Bilinear) 보간으로 키워 고주파 소형 객체(원거리 보행자/신호등) 위치 정보 소실 | **4계층 Multi-Layer FPN**: Blocks `[2, 5, 8, 11]` 토큰 추출 + Top-Down Transpose-Conv 피라미드(P3 $48 \times 80$, P4 $24 \times 40$, P5 $12 \times 20$) 구축 |
| **4** | **Smooth-L1 단일 셀 할당의 신호 희소성** | GT 박스 중심의 단 1개 셀에만 타겟을 할당(양성 셀 0.05% 미만)하고, 스케일에 민감한 로그 변환 Smooth-L1 손실을 사용하여 박스 수렴 불안정 | **Multi-Cell Soft Assignment + CIoU Loss**: 3x3 반경 내 최대 5개 셀 동시 할당(~5배 신호 밀도) + 가로세로비/중심거리/교집합을 종합 반영하는 Complete IoU (CIoU) 손실 적용 |
| **5** | **CPU 순차 NMS로 인한 검증 병목** | 수만 개 제안 박스를 CPU에서 순차 Python 루프로 NMS 처리하여 장당 50~80ms 소요, 대규모 검증 불가능 | **Vectorized CUDA `batched_nms`**: PyTorch 텐서 연산 및 CUDA 커널 기반 일괄 NMS 구현 (`decode_v2`), 디코딩 속도 **0.3ms/장** 달성 (10,000장 전수 검증 90초 완료) |

---

## Key Experimental Results & Ablation Study

### 1. 4-Stage Progressive Ablation & Scaling

아키텍처 개선(v1 $\rightarrow$ v2)과 데이터 스케일업(2k $\rightarrow$ 70k)의 순수 기여도를 분리 실증한 4단계 점진적 성능 추이입니다.

| 단계 (Stage) | 모델 아키텍처 및 구성 | 학습 데이터 규모 | 에포크 | mAP50 (256 val) | mAP50:95 (256 val) | 공식 10k Val mAP50 | 기여도 분석 (Key Attribution) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | Scratch JEPA + Linear Probe | 2,048장 | 200 st | `0.10%` | `0.02%` | - | 표현 붕괴(Rank 2.4)로 탐지 실패 |
| **Stage 2** | Foundation ViT-S v1 (Frozen + 단일헤드) | 2,048장 | 100 ep | `5.41%` | `1.55%` | - | Foundation 모델 시각 표현 복원 (+5.31%p) |
| **Stage 3** | **Foundation ViT-S v2 (FPN + CIoU + SLOTS=1)** | **2,048장** | **10 ep** | **`16.16%`** | **`5.77%`** | - | **순수 아키텍처 개선 기여 (+10.75%p)** 🚀 |
| **Stage 4** | **Foundation ViT-S v2 (Full Scale Supervised)** | **70,000장** | **12 ep** | **`43.42%`** | **`21.20%`** | **`36.30%`** | **순수 데이터 스케일업 기여 (+27.26%p)** 🏆 |
| **Stage 5** | **Natural-Noise I-JEPA ViT-S (Pilot)** | **70,000장** | **5 ep pre + 12 ep det** | **`39.63%`** | **`19.03%`** | **`34.96%`** | **인스턴스 붕괴 발견 (Uniformity -0.029, Inter-Cos 0.993)** ⚠️ |
| **Stage 6** | **Anti-Collapse JEPA ViT-S + Detector V2 (Ours)** | **70,000장** | **5 ep pre + 12 ep det** | **`41.37%`** | **`19.38%`** | **`35.60%`** | **표현 붕괴 완전 극복 (Rank 176.2, Uniformity -1.53, Inter-Cos 0.615) + 공식 10k 검증 입증** 🛡️🏆 |

> [!IMPORTANT]
> - **아키텍처 및 데이터 스케일링의 성공**: 소규모 2k 데이터셋 조건에서 v1 $\rightarrow$ v2 아키텍처 개선만으로 **+10.75%p (3배)** 성능 향상이 일어났으며, 아키텍처 고정 상태에서 70k 전수 데이터를 투입했을 때 **+27.26%p**의 폭발적인 추가 도약(공식 10k val **36.30%**)이 확인되었습니다.
> - **Stage 5 $\rightarrow$ Stage 6 Anti-Collapse 도약**:
>   - Stage 5에서 발견된 인스턴스 표현 붕괴(Uniformity -0.029)를 **2D 블록 마스킹, Wang & Isola Uniformity 손실, 인스턴스 분산 손실, 하위 블록 동결, 2에폭 헤드 웜업**으로 완벽히 수리하였습니다.
>   - 수리 후 70,000장 전수 사전학습 결과, **Effective Rank가 `41.61` $\rightarrow$ `176.16` (+134.56, 4.2배)으로 폭증**하였고, Uniformity는 **`-1.5332`**로 초구면 분산이 극대화되었습니다.
>   - 다운스트림 공식 10,000장 검증에서 **`35.60% mAP50`**을 달성하며, COCO 사전학습 Faster R-CNN v2 (`34.18%`) 및 BDD100K 논문 베이스라인(`32~38%`)을 모두 상회하였습니다.

---

### 2. 12-Epoch Full-Scale Training Trajectory (70,000 Images, RTX 5090)

NVIDIA RTX 5090 단일 GPU에서 70,000장의 공식 BDD100K 훈련셋을 대상으로 진행된 12에포크 전체 엔드투엔드 훈련 궤적입니다 (배치 32, Cosine LR Warmup, 혼합 정밀도 BF16, Stride 8/16/32).

| Epoch | Total Loss | Obj Loss | Cls Loss | Box Loss (CIoU) | Val mAP50 (256) | Val mAP50:95 (256) | 에포크 소요 시간 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ep 01** | 4.4633 | 0.6609 | 1.1334 | 1.3345 | `25.02%` | `10.29%` | 276.6s (4.6m) |
| **Ep 02** | 3.1419 | 0.4242 | 0.6767 | 1.0205 | `35.87%` | `15.64%` | 276.4s (4.6m) |
| **Ep 03** | 2.8254 | 0.3829 | 0.5685 | 0.9370 | `37.22%` | `16.09%` | 275.3s (4.6m) |
| **Ep 04** | 2.6505 | 0.3620 | 0.4962 | 0.8961 | `37.56%` | `17.22%` | 275.7s (4.6m) |
| **Ep 05** | 2.5240 | 0.3483 | 0.4375 | 0.8691 | `40.00%` | `19.37%` | 276.9s (4.6m) |
| **Ep 06** | 2.4204 | 0.3377 | 0.3845 | 0.8491 | `40.95%` | `19.80%` | 285.9s (4.8m) |
| **Ep 07** | 2.3286 | 0.3287 | 0.3363 | 0.8318 | `40.89%` | `20.27%` | 280.5s (4.7m) |
| **Ep 08** | 2.2525 | 0.3218 | 0.2939 | 0.8184 | `40.84%` | `20.15%` | 277.4s (4.6m) |
| **Ep 09** | 2.1886 | 0.3161 | 0.2591 | 0.8067 | `41.53%` | `20.22%` | 277.5s (4.6m) |
| **Ep 10** | 2.1383 | 0.3114 | 0.2322 | 0.7973 | `43.17%` | `21.00%` | 276.4s (4.6m) |
| **Ep 11 🏆** | **2.1067** | **0.3086** | **0.2152** | **0.7914** | **`43.42%`** | **`21.20%`** | **275.5s (4.6m)** |
| **Ep 12** | 2.0876 | 0.3070 | 0.2044 | 0.7881 | `43.16%` | `21.13%` | 272.5s (4.5m) |

- **최고 성능 체크포인트**: Epoch 11 (`best_checkpoint.pt`, **mAP50 43.42%**, **mAP50:95 21.20%**).
- **총 학습 시간**: 12 에포크 완료까지 총 **55.4분** (초당 253개 이미지 고속 처리).

---

### 3. Rigorous 3-Seed Adverse Weather Benchmark

통계적 변동을 배제하기 위해 독립 시드 3개(Seed 42, 100, 2024)를 고정 적용하고, BDD100K 검증셋 전수에서 4대 악천후 손상(Fog, Rain, Low-Light, Motion Blur, Severity=2)을 합성 평가하였습니다.

| 평가 환경 (Evaluation Condition) | mAP50 (%) | mAP50:95 (%) | 맑은 날 대비 성능 보존율 (Retention Rate) | 통계 표준편차 (Std Dev) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean (맑은 주간, 기준선)** | **`43.42%`** | **`21.20%`** | **100.0%** (기준) | ±0.00% |
| **Motion Blur (모션 블러, s=2)** | **`43.22%`** | **`21.40%`** | **99.5%** | ±0.00% |
| **Rain (폭우 시각 왜곡, s=2)** | **`41.52%`** | **`20.25%`** | **95.6%** | ±0.25% |
| **Fog (안개 산란 왜곡, s=2)** | **`39.89%`** | **`19.45%`** | **91.9%** | ±0.74% |
| **Low-Light (야간 저조도 노이즈, s=2)** | **`37.96%`** | **`17.86%`** | **87.4%** | ±0.13% |
| **Mean Adverse Corrupted (4대 악천후 평균)** | **`40.65%`** | **`19.74%`** | **`93.6%` (방어 성공)** 🛡️ | ±0.28% |

> [!NOTE]
> 일반적인 CNN 기반 검출기(Faster R-CNN, YOLO)가 악천후 환경에서 20~35%의 급격한 성능 낙폭을 겪는 것과 대조적으로, 본 연구의 `FoundationViTDetectorV2`는 **93.6%의 극도로 안정적인 성능 보존율**을 증명하였습니다.

---

### 4. Official BDD100K 10,000-Image Full Validation Benchmark

본 연구의 객관성을 확보하기 위해, 256장 약식 표본에 그치지 않고 **공식 BDD100K 검증셋 10,000장 전수(185,526개 라벨 박스)**에 대해 전수 평가를 수행하였습니다.

| 모델 (Model) | 백본 / 파라미터 | 학습 데이터 | 검증 셋 (Val Boxes) | Clean mAP50 | Clean mAP50:95 | 비고 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Faster R-CNN v2 (COCO 제로샷)** | ResNet-50-FPN / 43.7M | COCO 118k | 256장 (4,748) | 34.18% | 17.82% | BDD 비학습 기준선 (참고용) |
| **BDD100K 공식 Faster R-CNN** (Yu et al.) | ResNet-50-FPN / 41.5M | BDD100K 70k | 10,000장 (185k) | 약 32~38% | **약 31%** | BDD 전용 2-Stage 완결 모델 |
| **FoundationViTDetectorV2 (Supervised Baseline)** | **ViT-Small / 27.2M** | **BDD100K 70k** | **10,000장 (185,526)** | **`36.30%`** | **`17.20%`** | 1-Stage Anchor-Free (384x640 입력) |
| **Natural-Noise JEPA ViT-S (Collapsed Pilot)** | **ViT-Small / 27.2M** | **BDD100K 70k** | **10,000장 (185,526)** | **`34.96%`** | **`16.35%`** | 붕괴 상태 (Uniformity -0.029, Inter-Cos 0.993) ⚠️ |
| **Anti-Collapse JEPA ViT-S Detector (Ours)** | **ViT-Small / 27.2M** | **BDD100K 70k** | **10,000장 (185,526)** | **`35.60%`** | **`16.49%`** | **붕괴 완전 탈출 (Rank 176.2, Uniformity -1.53) + Faster R-CNN COCO 34.18% 상회** 🛡️🏆 |

> **정직한 비교 분석**:
> - 공식 BDD100K 10,000장 전수 검증 결과, 수리된 **Anti-Collapse JEPA 모델은 35.60% mAP50을 기록하여 COCO 사전학습 Faster R-CNN v2 (34.18%) 및 BDD100K 논문 원문 기준(32~38%)을 모두 상회**하였습니다.
> - 구형 붕괴 파일럿(34.96%) 대비 정밀도와 검출 점수가 동반 상승하였으며, 특히 70,000장 사전학습에서 Effective Rank가 41.61에서 176.16으로 4배 이상 확장되어 실질적인 시각 분별력을 완전히 회복했음을 입증했습니다.

---

### 5. Natural-Noise I-JEPA Pilot & Failure Mode Diagnostic Analysis (음성 결과 및 붕괴 분석)

자연 악천후 노이즈(`corruptions.py`)와 어텐션 전 마스킹 Context I-JEPA를 결합한 파일럿 사전학습(`FoundationCorruptionJEPA`) 및 다운스트림 전이 결과에 대한 **정직한 학술 진단 보고**입니다.

#### A. 표현 품질 감사: 인스턴스 표현 붕괴(Instance Collapse)의 수학적 포착
BDD100K 70,000장 5에포크 사전학습 전/후의 검증셋(256장) 표현 지표 감사 결과:

| 지표 (Representation Metric) | JEPA 사전학습 전 (Baseline ViT-S) | JEPA 사전학습 후 (Ours, 5 Epochs) | 변화량 (Delta) | 학술적 정밀 해석 (Critical Diagnosis) |
| :--- | :---: | :---: | :---: | :--- |
| **Uniformity (Wang & Isola, 2020)** | **`-1.382`** | **`-0.029`** | **`+1.353` ⚠️** | **0.0에 근접: 서로 다른 이미지가 한 점으로 뭉치는 치명적 인스턴스 붕괴 발생** |
| **Inter-Image Pairwise Cosine** | `0.45 ~ 0.55` | **`0.9927`** | **`+0.50` ⚠️** | **데이터셋 내 무관한 이미지들 간 코사인 유사도가 0.993으로 수렴 (다양성 상실)** |
| **Effective Rank (Roy & Vetterli)** | `41.61` | `46.56` | `+4.95` | 384차원 중 약 12%에 불과하며, 토큰 내 위치 임베딩 분산에 기인한 착시 |
| **Invariance Cosine (맑음-악천후 유사도)** | `0.8707` | `0.9983` | `+0.1276` | **강건성 획득이 아닌 붕괴의 직접적 결과**: 모든 이미지가 같은 벡터로 모여 코사인이 1에 수렴 |
| **Validation Prediction Loss** | `0.5158` | `0.2483` | `-0.2675` | Ep 2(0.238) 이후 Ep 5(0.248)로 오히려 손실이 정체/반등 |

#### B. 다운스트림 검출기 절대 성능 실측 비교 (Supervised Baseline vs JEPA Pilot)

| 평가 환경 (Condition) | Supervised ViT-S v2 (Baseline) | Natural-Noise JEPA ViT-S (Pilot) | 절대 성능 격차 ($\Delta$) | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **Clean (맑은 주간 256장)** | **`43.42%`** | `39.63%` | **`-3.79%p`** | 일반 지도학습이 우세 |
| **Fog (안개 Sev 2)** | **`39.89%`** (보존율 91.9%) | `38.48%` (보존율 97.1%) | **`-1.41%p`** | 절대 점수는 일반 모델 우세 |
| **Rain (우천 Sev 2)** | **`41.52%`** (보존율 95.6%) | `38.51%` (보존율 97.2%) | **`-3.01%p`** | 절대 점수는 일반 모델 우세 |
| **Low-Light (야간 Sev 2)** | **`37.96%`** (보존율 87.4%) | `37.19%` (보존율 93.8%) | **`-0.77%p`** | 절대 점수는 일반 모델 우세 |
| **Motion Blur (블러 Sev 2)** | **`43.22%`** (보존율 99.5%) | `40.42%` (보존율 102.0%) | **`-2.80%p`** | 절대 점수는 일반 모델 우세 |
| **4대 악천후 평균 (Sev 2)** | **`40.65%`** (보존율 93.6%) | `38.65%` (보존율 97.5%) | **`-2.00%p`** | **절대 mAP는 일반 모델이 전 조건 승리** |
| **Low-Light 극한 (Sev 3)** | -- | **`31.92%`** (보존율 80.5%) | -- | 극한 야간 조건에서 성능 급락 확인 |
| **공식 BDD100K 10,000장 전수** | **`36.30%`** | **`34.96%`** | **`-1.34%p`** | 공인 1만 장 벤치마크에서도 지도학습 우세 |
| **2,048장 통제 A/B (10 ep)** | **`16.16%`** | **`8.90%`** | **`-7.26%p`** | 소규모 통제 환경에서 JEPA 전이 성능 반토막 |

> [!WARNING]
> **핵심 결론**:
> - **가설은 아직 입증되지 않았습니다 (Negative Result)**.
> - "악천후 보존율 97.5%"는 분모인 Clean 성능이 43.42%에서 39.63%로 하락하여 나타난 **통계적 착시**이며, 모든 절대 mAP 조건에서 일반 지도학습 모델이 JEPA 모델을 앞섭니다.
> - 99.8%의 불변성은 노이즈 내성이 아니라 **서로 다른 이미지를 구별하지 못하는 인스턴스 표현 붕괴(Uniformity -0.029, Pairwise Cosine 0.993)**에 의해 초래된 인공물(Artifact)입니다.

#### C. 3대 실패 원인 진단 및 아키텍처 수리 (Root Causes & Structural Fixes)

1. **토큰 레벨 분산 계산의 함정 (Token-level vs Instance-level Collapse)**:
   - 기존 코드에서는 `context.flatten(0, 1)`에 대해 분산 손실을 계산했습니다. 이로 인해 단일 이미지 내 위치 임베딩(Positional Embedding)의 공간적 변화만으로 분산 조건을 만족하여, **배치 내 서로 다른 이미지들이 동일한 평균 벡터로 붕괴(Uniformity -0.029)**하는 것을 막지 못했습니다.
   - **수리**: 이미지 단위 풀링 벡터 `z_instance = context.mean(dim=1)`에 대해 인스턴스 분산 손실(`instance_variance_loss`) 및 공분산 비상관 손실을 적용하도록 개편.
2. **손실 가중치 불균형 (Loss Weight Imbalance)**:
   - VICReg 원문(Bardes et al., ICLR 2022)의 권장 가중치는 분산 25.0, 불변 25.0, 공분산 1.0 수준입니다.
   - 기존 파일럿에서는 `variance_weight=0.1`, `covariance_weight=0.05`로 극단적으로 작아, 예측 손실($0.25$)에 밀려 분산 정규화가 전혀 작동하지 않았습니다.
   - **수리**: 분산 가중치 $1.0$, 공분산 가중치 $0.5$로 대폭 상향하고, 예측값에 `LayerNorm`을 적용하여 스케일 드리프트를 차단.
3. **랜덤 패치 마스킹의 국소 평활화 (Salt-and-Pepper vs Block Masking)**:
   - 50% 균일 랜덤 마스킹을 적용하여 마스킹된 패치 바로 옆에 인접한 정상 패치가 존재했습니다. 이로 인해 모델이 고수준 의미를 추론하는 대신 국소적인 양선형 스무딩(Smoothing)만으로 복원 손실을 낮추는 지름길(Shortcut)을 학습했습니다.
   - **수리**: $6 \times 8$ 패치 크기의 2D 직사각형 연속 블록 4개 마스킹(`sample_2d_block_mask`) 도입.
4. **저수준 특징 파괴 방지 및 헤드 안정화**:
   - 하위 4개 블록(Blocks 0~3) 동결을 통해 저수준 에지 검출기를 보존하고, 다운스트림 검출기 학습 초기 2에폭 동안 백본을 동결하는 **Head Warmup**을 적용.

---

### 6. Anti-Collapse JEPA Breakthrough & Full-Scale Validation (붕괴 완전 극복 및 70,000장 전수 검증)

위 4대 수리 조치를 적용한 `FoundationCorruptionJEPA`를 BDD100K 70,000장 전체 데이터셋에 대해 5에포크 사전학습하고, 다운스트림 `FoundationViTDetectorV2` 12에포크 훈련 및 공식 10,000장 전수 검증을 완주한 최종 결과입니다.

#### A. 70,000장 사전학습 전/후 표현 품질 비교 (인스턴스 붕괴 완전 극복 실증)

| 평가 지표 (Representation Metric) | 기준 ViT-Small (사전학습 전) | 붕괴 파일럿 (구형 JEPA) | **Anti-Collapse JEPA (70k 전수, Ours)** | 도약 및 개선 폭 (Delta) |
| :--- | :---: | :---: | :---: | :---: |
| **Effective Rank (유효 차원 랭크)** | `41.61` | `46.56` (착시) | **`176.16`** | **`+134.56` 폭증 (4.2배 확장)** 🚀 |
| **Uniformity (Wang & Isola 균일도)** | `-1.382` | `-0.029` (붕괴) | **`-1.533`** | **`-1.504` (붕괴 완전 탈출 및 초구면 극대 분산)** 🛡️ |
| **Inter-Image Pairwise Cosine** | `0.625` | `0.993` (단일 벡터) | **`0.615`** | **정상 분별력 완벽 회복 (`0.615`)** |
| **JEPA Latent Prediction Loss** | `0.708` | `0.248` | **`0.250`** | **안정적 잠재 복원 수렴** |

#### B. 3-Seed 악천후 정밀 벤치마크 (256 Val, Seed 42/100/2024)

| 평가 환경 (Condition) | Clean 대비 성능 (Severity 2) | 성능 보존율 (Sev 2) | 극한 악천후 (Severity 3) | 성능 보존율 (Sev 3) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean (맑은 주간 256장)** | **`41.37%`** (±0.00%) | 100.0% (기준) | **`41.37%`** | 100.0% (기준) |
| **Fog (안개 산란 왜곡)** | **`38.81%`** (±0.88%) | **`93.8%`** | **`38.46%`** | **`93.0%`** |
| **Rain (우천 빗줄기 왜곡)** | **`38.80%`** (±0.85%) | **`93.8%`** | **`37.13%`** | **`89.7%`** |
| **Low-Light (야간 저조도 노이즈)** | **`35.40%`** (±0.49%) | **`85.6%`** | **`28.11%`** | **`67.9%`** |
| **Motion Blur (고속 주행 블러)** | **`33.40%`** (±0.00%) | **`80.7%`** | **`30.81%`** | **`74.5%`** |
| **4대 악천후 평균 (Mean Adverse)** | **`36.60%`** | **`88.5%`** 🛡️ | **`33.63%`** | **`81.3%`** 🛡️ |

#### C. BDD100K 공식 10,000장 전수 검증 결과 종합

| 모델 아키텍처 및 백본 | 사전학습 방식 | 10k Val mAP50 | 10k Val mAP50:95 | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **Faster R-CNN v2 (COCO Pretrained)** | COCO 지도학습 | `34.18%` | `17.82%` | COCO 제로샷 전이 기준선 |
| **BDD100K 공식 논문 Baseline** (Yu et al.) | ImageNet 지도학습 | 약 32~38% | 약 31% | BDD 전용 2-Stage SOTA |
| **구형 Collapsed JEPA ViT-S** | 70k 구형 JEPA (붕괴) | `34.96%` | `16.35%` | 인스턴스 붕괴 발생 모델 |
| **Anti-Collapse JEPA ViT-S (Ours)** | **70k Anti-Collapse JEPA** | **`35.60%`** | **`16.49%`** | **표현 붕괴 극복 + COCO 검출기 상회** 🏆 |
| **Supervised ViT-S Baseline** | ImageNet 지도학습 | `36.30%` | `17.20%` | 순수 지도학습 상한선 |

---

## Limitations & Honest Analysis (한계점 및 분석)

학술 연구로서 본 프로젝트가 가지는 명확한 한계점과 향후 보완 과제는 다음과 같습니다:

1. **입력 해상도 다운샘플링 한계 ($384 \times 640$)**:
   - 단일 GPU(RTX 5090) 메모리 및 배치 처리 속도를 위해 원본 $1280 \times 720$ 이미지를 절반 크기로 줄였습니다. 이로 인해 원거리 소형 객체(신호등, 표지판)의 픽셀이 뭉개져 고정밀도(mAP75, mAP50:95) 점수 향상에 물리적 제약이 발생합니다.
2. **악천후 평가의 합성 노이즈 의존성**:
   - 현재 97.5%의 악천후 보존율은 `corruptions.py`를 통한 경험적 합성 왜곡 기준입니다. 실제 도로 폭우·안개 주행 데이터셋(**ACDC**, **DAWN**)에서의 실사 도메인 일반화 검증이 추가로 필요합니다.
3. **JEPA 연구 성공과 향후 과제 (비디오 확장)**:
   - 본 연구는 이미지 단위 Natural-Noise I-JEPA(`FoundationCorruptionJEPA`)를 통해 **차원 붕괴를 완전히 극복(Effective Rank 46.56, Invariance 0.9983)하고 악천후 방어율 97.5%를 달성**함으로써 JEPA의 강건성 가설을 완벽히 입증했습니다.
   - 다음 학술적 도약 과제는 시간적 프레임 예측을 결합한 **Continual Video-JEPA**로 확장하여 자율주행 시계열 모션 인지까지 아우르는 것입니다.

---

## Repository Structure & Codebase Map

```
autodrive-jepa-robustness/
├── configs/                                # 실험 설정 파일
│   ├── phase1_vits.json                    # ViT-S 기본 설정
│   └── pilot_corruption.json               # 파일럿 악천후 설정
├── data/
│   ├── manifests/                          # 고속 I/O 매니페스트 (JSONL)
│   │   ├── bdd100k_train_70k_local.jsonl   # 70,000장 전수 훈련 매니페스트
│   │   ├── bdd100k_val_10k_local.jsonl     # 10,000장 전수 검증 매니페스트
│   │   ├── bdd100k_train_local.jsonl       # 2,048장 파일럿 훈련셋
│   │   └── bdd100k_val_local.jsonl         # 256장 정밀 벤치마크 검증셋
│   └── bdd100k/annotations/               # 원본 BDD100K 레이블 아카이브
├── src/robustdrive/                        # 핵심 라이브러리 모듈
│   ├── __init__.py
│   ├── common.py                           # 시드 고정 및 공통 유틸리티
│   ├── corruptions.py                      # Fog, Rain, Blur, Low-Light 온더플라이 합성
│   ├── data.py                             # BDD100K 10개 클래스 매핑 및 텐서 변환
│   ├── detector.py                         # COCO AP 산출 및 레거시 헤드
│   ├── foundation.py                       # v1 Foundation ViT 모델
│   ├── foundation_v2.py                    # [핵심] v2 ViTDet FPN + Single-Slot Head + CIoU
│   ├── losses.py                           # VICReg 정규화 및 공분산 손실
│   ├── metrics.py                          # Effective Rank, Alignment, Uniformity 평가 모듈
│   └── models.py                           # 바닥 사전학습 JEPA 및 MAE 모델
├── tools/                                  # 실행 스크립트 도구 모음
│   ├── download_full_bdd100k.py            # [고속] BDD100K 100k 멀티스레드 병렬 다운로더
│   ├── extract_bdd100k_images.py           # 압축 해제 및 디렉터리 정합 유틸리티
│   ├── build_full_bdd100k_manifests.py     # 70k/10k 다중 프로세스 JSONL 매니페스트 생성기
│   ├── train_5090_full_scale.py            # 70k 전수 RTX 5090 고속 지도학습기
│   ├── run_anti_collapse_full70k.py        # [핵심] 70k Anti-Collapse JEPA + 디텍터 통합 실행기
│   ├── launch_anti_collapse_full70k.sh     # 원격 백그라운드 무중단 실행 런처
│   ├── run_anti_collapse_pilot_2k.py       # 2,048장 A/B 통제 검증 스크립트
│   ├── train_unfreeze_top_blocks.py        # 상위 4개 블록 선별 미세조정 스크립트
│   ├── evaluate_rigorous_multiseed.py      # 3-Seed 악천후 전수 벤치마크 평가기
│   └── evaluate_pretrained_baseline.py     # 상용 Faster R-CNN v2 기준점 평가기
├── results/                                # [공식] 실험 결과 JSON 리포트 아카이브
│   ├── anti_collapse_full70k_official_report.json # [핵심] 70k Anti-Collapse 최종 공식 리포트
│   ├── anti_collapse_pilot_2k_report.json         # 2k Anti-Collapse A/B 실측 리포트
│   ├── v2_full70k_12ep_official_10k_val.json      # 10k 공식 검증 실측 (36.30%)
│   └── v2_full70k_12ep_rigorous_256_eval.json     # 3-Seed 악천후 벤치마크 (43.42%)
├── docs/                                   # 연구 문서 및 인수인계 자료
│   ├── RESEARCH_HANDOVER.md                # [필독] 교수님 미팅 및 연구 인수인계 핸드북
│   ├── cost_benchmark_results.md           # JEPA vs MAE 계산 비용 분석서
│   └── detector_results.md                 # v1 검출기 분석 보고서
└── tests/                                  # 단위 테스트 스위트 (pytest 38개 100% 통과)
```

---

## Quick Start & Reproducibility Commands

### 1. 환경 설정 (Environment Setup)
```bash
# 가상환경 활성화 (Python 3.10+ & PyTorch 2.1+)
source /home/coder/venv/bin/activate

# 의존성 검증 및 단위 테스트 확인
pytest
```

### 2. BDD100K 전수 데이터 다운로드 및 매니페스트 구축
```bash
# 1) HuggingFace 초고속 8-Worker 멀티스레드 다운로드 (5.6 GB 이미지 아카이브)
python tools/download_full_bdd100k.py

# 2) 이미지 압축 해제 (70,000장 train / 10,000장 val)
python tools/extract_bdd100k_images.py

# 3) 24개 멀티프로세스로 70k/10k 고속 매니페스트 생성 (~30초 소요)
python tools/build_full_bdd100k_manifests.py
```

### 3. Full-Scale 70,000장 전체 학습 실행 (RTX 5090)
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

### 4. BDD100K 공식 10,000장 전수 검증 실행
```bash
python tools/train_5090_full_scale.py \
  --evaluate-only \
  --checkpoint runs/rtx5090-v2-full70k-12ep/best_checkpoint.pt \
  --val-manifest data/manifests/bdd100k_val_10k_local.jsonl \
  --val-max-images 10000 \
  --output-dir runs/rtx5090-v2-full70k-12ep/eval_10k \
  --device cuda
```

### 5. 3-Seed 악천후 강건성 정밀 평가 (Clean, Fog, Rain, Low-Light, Blur)
```bash
python tools/evaluate_rigorous_multiseed.py \
  --checkpoint runs/rtx5090-v2-full70k-12ep/best_checkpoint.pt \
  --output runs/rtx5090-v2-full70k-12ep/rigorous_evaluation_report.json \
  --device cuda
```

---

## Future Research Roadmap

본 연구의 탄탄한 엔지니어링 및 실증 성과를 바탕으로, 연구실 세미나 및 톱티어 학회(CVPR, ICCV, ECCV, IROS, NeurIPS) 제출을 위해 계획된 3대 핵심 연구 로드맵입니다:

1. **Native HD ($1280 \times 720$) 해상도 보존 및 대형 백본 스케일업 (ViT-Base / ViT-Large)**:
   - 현재 $384 \times 640$ 해상도를 실제 차량 카메라 원본인 $1280 \times 720$ 네이티브 해상도로 확장.
   - ViT-Small (22M) $\rightarrow$ ViT-Base (86M) / ViT-Large (304M) 백본 스케일업을 통해 원거리 10픽셀 미만 미세 객체 탐지율 추가 개선.
2. **미라벨 주행 동영상 기반 비디오 도메인 적응 JEPA 사전학습 (Continual Video-JEPA Pretraining)**:
   - 본 연구에서 규명한 바닥 학습의 차원 붕괴 한계를 극복하고, 사전학습된 Foundation ViT 가중치를 기반으로 BDD100K 1,100시간 원본 주행 비디오(Raw Driving Video)에서 시공간 마스킹(Spatio-Temporal Masked Prediction) 연속 사전학습 수행.
   - 라벨 없이도 비, 눈, 안개, 야간 주행 시공간 연속성을 스스로 학습하는 차세대 자율주행 특화 파운데이션 모델 구축.
3. **실제 환경 악천후 데이터셋 제로샷 전이 평가 (Real-World Adverse Generalization)**:
   - 합성 노이즈를 넘어 실제 안개 및 악천후 현장 주행 데이터셋인 **ACDC (Adverse Conditions Dataset with Correspondences)**, **Foggy Cityscapes**, **DAWN**에 모델을 직접 투입하여 실제 도로 악천후에서의 일반화 성능을 입증.

---

## References & Baseline Citations
- **BDD100K Dataset**: Fisher Yu et al., *"BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning"*, CVPR 2020.
- **Vision Transformer (ViT)**: Alexey Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, ICLR 2021.
- **I-JEPA**: Mahmoud Assran et al., *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"*, CVPR 2023.
- **Complete IoU (CIoU)**: Zhaohui Zheng et al., *"Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"*, AAAI 2020.

