# JEPA and masked-pixel pretraining cost pilot

## Conclusion

This matched pilot does **not** support the claim that JEPA is cheaper than a masked-pixel objective. On `owner/jwj-01`, the custom corruption-aware masked autoencoder used 19.2% less training-loop time and about 45.6% less idle-adjusted GPU-board energy than corruption-aware JEPA for the same 2,560-step budget. Its detector mAP was in the same very-low range as clean JEPA and higher than corruption-aware JEPA on the synthetic-corruption mean.

The result describes these implementations and this one-seed subset. It is not a general ranking of JEPA and MAE, and the MAE implementation here is a small custom denoising masked autoencoder, not an official MAE reproduction.

## Matched pretraining setup

- Same ViT-S encoder geometry: patch 16, width 384, 12 layers, 6 heads; 384×640 input and 60% random patch masking.
- Same 2,048-image training manifest, seed 17, ten epochs, batch 8, AdamW schedule, BF16, RTX 5090 32 GB, PyTorch 2.14.1+cu130 and CUDA 13.0. Each method completed 2,560 optimizer steps from a fresh initialization.
- Clean JEPA uses clean visible patches and clean target representations. Corruption-aware JEPA uses the selected synthetic corruption for its visible patches and predicts clean target representations. The masked autoencoder uses those same corrupted views and predicts normalized clean pixels at masked positions.
- JEPA and the masked autoencoder use the same 192-wide, four-layer prediction/decoder geometry. MAE has no EMA target encoder.
- GPU power was sampled every 500 ms. Total board energy is the trapezoidal integral of sampled `nvidia-smi` board power; idle-adjusted energy subtracts the mean idle sample of 34.28 W. This excludes CPU, storage, network, and whole-system energy.

## Measured pretraining cost

| Pretraining method | Job wall time | Training-loop time | GPU board energy | Idle-adjusted GPU energy | Peak CUDA allocated |
|---|---:|---:|---:|---:|---:|
| Clean JEPA | 140.34 s | 135.11 s | 10.64 Wh | 9.33 Wh | 1.35 GiB |
| Corruption-aware JEPA | 148.04 s | 145.37 s | 11.35 Wh | 9.96 Wh | 1.35 GiB |
| Corruption-aware masked autoencoder | 120.06 s | 117.40 s | 6.54 Wh | 5.42 Wh | 1.26 GiB |

Against corruption-aware JEPA, this masked autoencoder run used 18.9% less job wall time, 19.2% less training-loop time, 42.4% less total sampled GPU energy, and 45.6% less idle-adjusted GPU energy. Against clean JEPA, it used 14.4% less job wall time and 41.9% less idle-adjusted GPU energy.

Corruption-aware JEPA itself cost 5.5% more job wall time and 6.8% more idle-adjusted GPU energy than clean JEPA in this run. Its detection score below did not improve over clean JEPA.

## Matched frozen-encoder detection probe

Each encoder was frozen. An identical custom P3/P4/P5 detector head was trained for 25 epochs on 2,048 clean training images, then evaluated on the same 256 validation images and 20 synthetic corruptions. The validation split supports nine classes. Values are percentages (`AP fraction × 100`).

| Encoder | Clean mAP50 | Mean corrupt mAP50 | Mean corrupt mAP50:95 |
|---|---:|---:|---:|
| Randomly initialized ViT-S | 0.2428% | 0.1388% | 0.0334% |
| Clean JEPA ViT-S | 0.2499% | 0.1472% | 0.0347% |
| Corruption-aware JEPA ViT-S | 0.1259% | 0.0835% | 0.0175% |
| Corruption-aware masked autoencoder ViT-S | 0.1872% | 0.1436% | 0.0484% |

Clean JEPA and the masked autoencoder are close on mean corrupt mAP50: clean JEPA is 0.0036 percentage points higher. The masked autoencoder has higher mean corrupt mAP50:95 in this probe. Corruption-aware JEPA is lower than both on the reported detection measures. All absolute mAP scores are extremely low, so these rankings are unstable and are not evidence of useful driving perception.

## What this supports

- For these custom implementations on this RTX 5090, corruption-aware JEPA was slower and consumed more GPU-board energy than the custom masked-pixel baseline at the same step count.
- The ViT-S backbone is held constant across the learned encoders, so the table compares pretraining objectives on that backbone; it is not a comparison of “JEPA versus ViT.”
- The random ViT row is an untrained encoder and is only a reference for the weak detector probe. Its zero pretraining cost is not a meaningful learned-model cost baseline.
- No inference-latency or inference-energy comparison was run. Because all retained encoders use the same ViT-S architecture, this experiment does not establish a JEPA inference-cost advantage.
- Results use one seed, a 2,048/256 subset, legacy BDD100K mirror labels, a custom detector, and synthetic corruption only. They do not establish real-weather generalization, production cost, or driving safety.

## Reproducibility files

- [Cost and power summary](../runs/cost_benchmark_summary.json)
- [Detection probe summary](../runs/detector_cost_probe_summary.json)
- [Local checkpoint backup receipt](../runs/cost_benchmark_local_backup.json)
- [Benchmark protocol](cost_benchmark_protocol.md)
- [MAE baseline implementation](../tools/run_mae_baseline.py)
- [Cost measurement runner](../tools/run_cost_benchmark.py)
- [Matched detection runner](../tools/run_detector_cost_probe.py)
