# JEPA and masked-pixel pretraining cost pilot

## Comparison

Compare three encoders with the same ViT-S backbone (patch 16, width 384, 12 layers, 6 heads), 384×640 input, 60% random patch masking, batch size 8, BF16, AdamW schedule, seed 17, and 2,560 optimizer steps over the same 2,048 training images:

- clean JEPA: clean visible patches predict clean target representations;
- corruption-aware JEPA: corrupted visible patches predict clean target representations;
- corruption-aware masked autoencoder: corrupted visible patches predict normalized clean target pixels.

The masked autoencoder uses a 192-dimensional, four-layer pixel decoder, matching the JEPA predictor's decoder geometry. It does not use a target encoder or EMA. Both corruption-aware methods draw from the same four synthetic corruptions and severities 1–3.

This is a small in-repository denoising masked-autoencoder baseline, not a reproduction of an official MAE training recipe or published benchmark.

## Cost measures

- wall time for the training invocation and in-loop training schedule;
- RTX 5090 power draw sampled every 500 ms during each invocation, integrated over timestamps to estimate GPU energy;
- idle GPU power sampled for 15 seconds immediately before the runs, for an approximate idle-adjusted estimate;
- peak CUDA allocated and reserved memory;
- final frozen encoder is evaluated with the same downstream detection-head protocol.

The energy value is GPU-board energy estimated from periodic power samples, not whole-system energy or an electricity bill. It is a single run per method and is sensitive to GPU clock, temperature, and background load.

## Boundaries

This is a small pilot on 2,048 train and 256 validation images from the legacy-labeled BDD100K mirror subset, with one seed and synthetic corruption. It can support a local, hardware-specific compute observation. It cannot establish general JEPA efficiency, production cost, real-weather robustness, or driving safety. The detector probe is custom and prior pilot mAP was very low; report its measured result without treating it as deployment performance.
