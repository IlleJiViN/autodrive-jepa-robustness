# JEPA object-detection robustness pilot: results

Run: `runs/detector-probe-seed17-v2/`, completed on `owner/jwj-01` with seed 17.
The machine-readable summary is `runs/detector_probe_summary.json`; all per-condition metrics and the three trained detector checkpoints were copied locally and SHA-256 verified. The copy receipt is `runs/detector_local_backup.json`.

## Protocol actually run

- Data: labeled legacy BDD100K mirror subset, 2,048 train images / 37,868 boxes and 256 validation images / 4,978 boxes. All train and validation image hashes were checked; there were no unlabeled pilot images. The validation subset contains 9 of the 10 manifest classes.
- Encoders: random ViT (0 JEPA steps), clean-JEPA final encoder (2,760 cumulative steps), and corruption-JEPA final encoder (2,760 cumulative steps). Encoders stayed frozen while separate identical three-scale heads were trained.
- Detector: custom anchor-free P3/P4/P5 probe with three slots per cell; this is not an Ultralytics YOLO detector. Each head saw the same clean training data for 25 epochs, batch size 16, AdamW at 0.0003, and fixed final-epoch selection.
- Evaluation: 256 validation images in clean and 20 synthetic conditions (fog, rain, motion blur, low light × severities 1–5). Reported AP is the custom 101-point interpolated, class-aware-NMS implementation, averaged over the 9 classes present in validation. The 300 highest-scoring boxes per image entered NMS; at most 100 detections were scored.
- The first run was stopped after its random baseline exposed a training/decoder mismatch for center offsets. Its files remain on the VM for audit but are excluded from this report. The corrected v2 run applies sigmoid to center offsets in both loss and decoding.

## Results

Values below are mAP percentages (fraction × 100), so for example 0.105% is an AP fraction of 0.00105.

| Frozen encoder | Clean mAP50 | Clean mAP50:95 | Mean corrupt mAP50 | Mean corrupt mAP50:95 | Final head loss |
|---|---:|---:|---:|---:|---:|
| Random ViT | 0.243% | 0.0545% | 0.1388% | 0.0334% | 1.3829 |
| Clean JEPA | 0.1235% | 0.0257% | 0.0958% | 0.0233% | 2.3936 |
| Corruption JEPA | 0.1902% | 0.0504% | 0.1048% | 0.0280% | 2.1758 |

Mean mAP50 by corruption family:

| Family | Random ViT | Clean JEPA | Corruption JEPA |
|---|---:|---:|---:|
| Fog | 0.0910% | 0.0474% | 0.0495% |
| Rain | 0.2446% | 0.1243% | 0.1723% |
| Motion blur | 0.1766% | 0.1171% | 0.1279% |
| Low light | 0.0430% | 0.0944% | 0.0695% |

Corruption JEPA exceeded clean JEPA's mean corrupted mAP by 0.0090 percentage points for mAP50 (+9.4% relative) and 0.0047 points for mAP50:95 (+20.2% relative). It remained below the random encoder by 0.0340 points for mAP50 (-24.5% relative) and 0.0054 points for mAP50:95 (-16.2% relative). The family results are mixed: corruption JEPA is above clean JEPA for fog, rain, and motion blur, but below it for low light; random ViT is higher than both JEPA encoders in every family mean.

## Interpretation and limits

This run does **not** establish a detection-robustness improvement. Every mAP value is extremely low, the 25-epoch custom probe head is weak, the result uses one seed and a small same-source validation subset, and the synthetic corruptions are not real adverse-weather footage. The modest relative gain over clean JEPA is small in absolute AP and loses to random ViT; treat it as a pipeline result that motivates a stronger, standardized detector probe, longer/equal-budget tuning, multiple seeds, and real-weather holdouts.

The detection results are separate from the earlier image-level class-presence probe (macro AP); one must not substitute that easier task's score for box-detection mAP.
