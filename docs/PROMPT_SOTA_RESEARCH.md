# AutoDrive-JEPA: Master AI Research Prompt
## Role: World-Class Autonomous Driving Vision Architect & PyTorch Engineer

> **Usage Guide**: Copy and paste the prompt below into Claude 3.5 Sonnet, GPT-4o, or a new Antigravity session to get an in-depth architectural diagnosis, concrete drop-in PyTorch modules (TAL, DFL, Window Attention), and next-generation design strategies for bridging the gap to SOTA.

---

```markdown
# Role: World-Class Autonomous Driving Vision Architect & PyTorch Engineer

You are a principal computer vision researcher and high-performance PyTorch architect specializing in autonomous driving perception, robust foundation transformers, and real-time object detection under adverse weather.

I am conducting research on **Multi-Scale Foundation ViT Object Detection under Adverse Weather (BDD100K)** using **Joint-Embedding Predictive Architecture (JEPA)**. I need you to critically analyze my current architecture against modern State-of-the-Art (SOTA) object detection literature, identify the exact architectural bottlenecks holding back my absolute mAP, and propose concrete, production-grade PyTorch module designs to bridge the gap while preserving my model's unique adverse weather robustness advantage.

---

### [SECTION 1] My Current System Architecture & Empirical Baseline

My codebase implements an end-to-end multi-scale vision transformer detector (`FoundationViTDetectorV2`) and a self-supervised natural-noise pretraining pipeline (`FoundationCorruptionJEPA`) evaluated on BDD100K (70,000 train images, 10,000 official val images, 10 driving classes):

1. **Backbone & Token Extraction**:
   - Backbone: `timm.create_model('vit_small_patch16_384', pretrained=True, img_size=(384, 640))`
   - Parameters: 21.8M (embed_dim = 384, depth = 12, heads = 6)
   - Input shape: `(B, 3, 384, 640)`, Patch grid: `(24, 40)` = 960 tokens
   - Multi-layer extraction: Intermediate tokens from Blocks `[2, 5, 8, 11]` (each reshaped to `(B, 384, 24, 40)`)
2. **Feature Pyramid Adapter (`ViTDetFPNAdapter`)**:
   - `P5` (stride 32, 12x20): `lateral_p5(Conv1x1)` + `Conv3x3(stride=2)`
   - `P4` (stride 16, 24x40): `lateral_p4(Conv1x1)` + top-down bilinear upsampled P5 + `Conv3x3`
   - `P3` (stride 8, 48x80): `ConvTranspose2d(stride=2)` from Block 2 + top-down bilinear upsampled P4 + `Conv3x3`
3. **Detection Head & Target Assignment**:
   - Head: Anchor-free `DecoupledScaleHead` per level with `SLOTS = 1`. Separate 2x Conv3x3 classification branch and 2x Conv3x3 regression/objectness branch.
   - Target Assignment: Static heuristic center assignment (`assign_targets_v2`). Up to a 3x3 grid around GT center inside radius `1.5 * stride` + fixed soft scale boundaries (P3: side <= 48, P4: 24 <= side <= 96, P5: side >= 48).
   - Losses: Complete IoU (`CIoU`) loss for boxes + Focal Sigmoid loss for Objectness + Cross-Entropy for classes.
   - Inference Decoding: Vectorized CUDA `torchvision.ops.batched_nms` (top 1000 candidates, IoU threshold 0.50).
4. **Self-Supervised JEPA Pretraining (`FoundationCorruptionJEPA`)**:
   - Natural noise pretraining using synthetic on-the-fly weather corruptions (`corruptions.py`: fog, rain, low_light, motion_blur).
   - Teacher (EMA target encoder) takes Clean image; Student (Context encoder) takes corrupted view with 50% block masking.
   - Anti-collapse regularizers: VICReg variance hinge (`floor=1.0`) + covariance decorrelation + Wang & Isola hyperspherical uniformity (`uniformity_weight=0.05, var=0.15, cov=0.05`).
5. **Empirical Benchmarks (RTX 5090)**:
   - Official BDD100K 10,000-image Val: **Clean mAP@50 = 35.60%**, **mAP@50:95 = 16.49%** (Speed: **54.2 FPS**, 18.4ms/img).
   - 3-Seed Adverse Weather Benchmark (256 val): Clean 41.37% -> Fog (Sev 2) **38.81% (93.8% retention)**, Rain (Sev 2) **38.80% (93.8% retention)**.
   - Representation Health: Effective Rank expanded from 41.6 to **106.7**, Invariance Cosine preserved at **0.9287**, Uniformity at **-0.7403**.

---

### [SECTION 2] The 5 SOTA Reference Papers to Anchor Against

Please benchmark my architecture against the innovations in these 5 papers:

1. **ViTDet: Exploring Plain Vision Transformer Backbones for Object Detection** (Li et al., Meta AI / FAIR, ECCV 2022, arXiv:2203.16527)
   - *Key Mechanisms*: Non-hierarchical Plain ViT for detection, 75% Windowed Self-Attention + 25% Global Attention to solve the $O(N^2)$ resolution scaling wall, Simple FPN without heavy lateral connections.
2. **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information** (Wang et al., CVPR 2024, arXiv:2402.13616)
   - *Key Mechanisms*: Programmable Gradient Information (PGI) with auxiliary reversible branch, GELAN architecture, Task-Aligned Assigner (TAL), Distribution Focal Loss (DFL).
3. **RT-DETR: DETRs Beat YOLOs on Real-time Object Detection** (Lv et al., Baidu, CVPR 2024, arXiv:2304.08069)
   - *Key Mechanisms*: Efficient Hybrid Encoder (AIFI intra-scale interaction + CCFM cross-scale fusion), uncertainty-minimal query selection, 100+ FPS real-time Transformer execution.
4. **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** (Zhang et al., ICLR 2023, arXiv:2203.03605)
   - *Key Mechanisms*: Contrastive DeNoising (CDN) training, Mixed Query Selection, Look Forward Twice (iterative box refinement across 6 decoder layers).
5. **IA-YOLO: Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions** (Liu et al., AAAI 2022, arXiv:2112.08088)
   - *Key Mechanisms*: Differentiable Image Processing (DIP) module with CNN-PP parameter predictor trained end-to-end for defogging/enhancement before detection.

---

### [SECTION 3] Your Actionable Tasks

Please provide a deep, mathematically rigorous, and concrete technical report answering:

1. **Gap Analysis (Why am I at 35.6% while SOTA is at 68-74%?):**
   - Deconstruct the exact architectural bottlenecks between my `FoundationViTDetectorV2` and YOLOv9 / DINO.
   - Quantify how much of the gap comes from: (a) Input resolution ($384 \times 640$ vs $1280 \times 720$), (b) Static $3 \times 3$ target assignment vs dynamic TAL / Hungarian matching, (c) Plain ViT $16 \times 16$ patch stem vs multi-stride hierarchical stems, and (d) Lack of box distribution modeling (DFL).
2. **ViT Resolution Scaling Strategy (Solving the $O(N^2)$ Trap):**
   - If I want to scale my input resolution from $384 \times 640$ to $768 \times 1280$ (native HD) without crashing GPU memory or dropping from 54 FPS to 5 FPS, how should I adapt Meta's ViTDet Window Attention into my `timm` ViT-Small backbone?
3. **Head & Assignment Upgrade (TAL & DFL for ViT):**
   - Provide a complete, drop-in PyTorch module implementing **Task-Aligned Assigner (TAL)** and **Distribution Focal Loss (DFL)** tailored for my `FoundationViTDetectorV2` output heads.
4. **Adverse Weather Defense (JEPA vs IA-YOLO):**
   - My model achieves 93.8% weather retention via Corruption-JEPA representation invariance. How can I combine my feature-space invariance with modern multiscale adapters without suffering from the small-object survivorship bias when moving to high resolution?
```
