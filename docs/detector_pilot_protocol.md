# JEPA encoder robustness: labeled detection pilot

## Question

Does the corruption-trained JEPA representation retain more object-detection performance under synthetic image corruption than a clean-trained JEPA representation or a randomly initialized ViT encoder, when each representation receives the same frozen-encoder detector training budget?

## Data

- Source: the matched legacy BDD100K box annotations already associated with the pilot image mirror; this is not the current Detection2020 benchmark.
- Train: 2,048 images with 37,868 boxes. Validation: 256 images with 4,978 boxes.
- Ten supported classes use the manifest's normalized legacy labels. Every encoder uses the same clean training images, image size, seed, head initialization, optimizer schedule, and fixed validation split.
- The validation data is a small pilot subset. It is not a full BDD100K result and does not establish real-weather robustness or driving safety.

## Encoders and detector

- Compare the randomly initialized ViT, the clean-JEPA final encoder, and the corruption-JEPA final encoder. The two JEPA encoders are the final scheduled exports at 2,760 cumulative optimizer steps; their selected first-continuation-epoch exports are not used.
- Freeze each encoder. Cache its 24×40×384 patch tokens for the clean training images. Train one identical three-level P3/P4/P5 anchor-free detector head per encoder, with three slots per cell.
- Train 25 fixed epochs, batch size 16, AdamW learning rate 0.0003 and weight decay 0.01. Do not select a head using validation metrics.
- Box assignment uses area scale and center cell. Train center offsets through sigmoid and log-width/log-height as linear values, matching the decoder. Keep the 300 highest-scoring candidates per image for class-aware NMS, then retain at most 100 predictions per image.

## Evaluation

- Score clean validation images and the same images with fog, rain, motion blur, and low-light corruption at severities 1–5.
- Report per-class AP50, mean AP over IoU thresholds 0.50:0.95, clean mAP, and the mean over the 20 corrupted conditions. AP uses 101-point interpolated precision and one-to-one greedy IoU matching.
- The primary comparison is corruption-JEPA minus clean-JEPA and random-encoder mean corrupt mAP. Keep clean accuracy visible alongside corruption scores.
- Report the single-seed, small-subset, synthetic-corruption limits with every result. Do not describe the custom probe head as an Ultralytics YOLO model.

## Outputs and execution state

The runner is `tools/run_detector_probe.py`; its configuration is `configs/detector_pilot.json`. It records the protocol, per-epoch losses, frozen encoder hashes, detector checkpoints, each condition's AP output, a combined summary, and a status file under `runs/detector-probe-seed17-v2/` and `runs/`.

Training checkpoints are saved at epoch boundaries and validated by encoder, manifest, and config identity before resuming. Completed feature caches and evaluation conditions are reused only when their identity and checksum match. Partial cache files are retained for inspection and are never silently treated as complete.

An initial run exposed a mismatch between training and decoding for center offsets; its metrics are retained on the VM but excluded. The corrected v2 run is complete. See [docs/detector_results.md](detector_results.md) for measured values and limitations.
