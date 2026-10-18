# Antigravity (AGY) Agent Workspace Guidelines: AutoDrive-JEPA-Robustness

## 1. Project & Environment Context
This repository contains the research codebase for **Multi-Scale Foundation ViT Object Detection under Adverse Weather (BDD100K)** using **Corruption-JEPA (Joint-Embedding Predictive Architecture)**.

### Hardware & VM Background
- **Primary Training VM**: NVIDIA GeForce RTX 5090 (32GB VRAM), graciously provided by our collaborator/friend.
- **VM Context & Resource Guide**: Refer to [`VM_CONTEXT.md`](./VM_CONTEXT.md) for full system status, disk allocation, and artifact locations.
- **Safety Directive**: All heavy model training has been completed and checkpoints are stored under `runs/`. Do not launch unprompted heavy training runs that would saturate the GPU without explicit user instruction.

---

## 2. Quick References for AGY
- **Core Models**:
  - `src/robustdrive/foundation_v2.py`: `FoundationViTDetectorV2` (ViT-Small, Multi-Scale FPN P3-P5, Decoupled Scale Head, CIoU Loss, GPU Batched NMS).
  - `src/robustdrive/foundation_jepa.py`: `FoundationCorruptionJEPA` (Natural-noise masked prediction, VICReg variance + covariance + Wang & Isola uniformity anti-collapse regularizers).
- **Benchmark Reports**:
  - `results/anti_collapse_full70k_official_report.json`: BDD100K 70,000-image train, 10,000-image official val (`35.60%` mAP50, `16.49%` mAP50:95, 93.8% weather retention).
  - `results/sweetspot_pilot_2k_report.json`: Balanced Regularization Sweet-Spot (`16.31%` clean mAP50, Effective Rank `106.70`, Invariance `0.9287`).
- **Tests**: Run `pytest tests -q` to verify offline functionality (38 unit tests covering all components).

---

## 3. Communication Directive
When answering questions for the VM owner/friend, provide warm, clear, and technically grounded explanations, referencing [`VM_CONTEXT.md`](./VM_CONTEXT.md).
