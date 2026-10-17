# Representation Robustness Audit: Beyond Cosine Similarity

Generated on: 2026-10-17 16:27:29

This report evaluates representations across Alignment, Uniformity, Effective Rank, Unbiased Linear CKA, and k-NN Jaccard neighborhood overlap.

## 1. Executive Summary Table

| Model | Clean ERank | Clean Uniformity (↓) | Mean Corrupt ERank | Mean Alignment (↓) | Mean Linear CKA (↑) | Mean k-NN Jaccard (↑) | Mean Cosine |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corruption JEPA** | 2.4 | -0.941 | 2.3 | 0.2746 | **0.8136** | **0.5895** | 0.8627 |
| **Clean JEPA** | 3.2 | -0.731 | 2.9 | 0.7757 | **0.6930** | **0.5709** | 0.6122 |
| **Random ViT** | 5.9 | -0.566 | 4.8 | 0.4904 | **0.8054** | **0.6004** | 0.7548 |

## 2. In-Depth Interpretation

- **Effective Rank**: Indicates whether the embedding space is spanning diverse dimensions. Values < 10 indicate severe dimensional collapse.
- **Alignment (Wang & Isola, 2020)**: Distance on the hypersphere between clean and corrupted pairs (Lower is better).
- **Uniformity (Wang & Isola, 2020)**: Measures how evenly points are distributed on the hypersphere (Lower is better; very high values indicate collapse).
- **Linear CKA (Kornblith et al., 2019)**: Geometric structural similarity between clean and corrupted feature manifolds (Higher is better, invariant to orthogonal rotations).
- **k-NN Jaccard (k=5)**: Topology preservation rate of the local sample neighborhood despite adverse corruptions.

## 3. Condition-by-Condition Breakdown

### Model: Corruption JEPA

| Family | Severity | Cosine (↑) | Alignment (↓) | Uniformity (↓) | ERank | Linear CKA (↑) | k-NN Jaccard (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| fog | 1 | 0.9775 | 0.0451 | -1.100 | 2.5 | 0.9586 | 0.4980 |
| fog | 3 | 0.8748 | 0.2504 | -0.903 | 2.6 | 0.8038 | 0.2649 |
| fog | 5 | 0.7484 | 0.5031 | -0.259 | 2.2 | 0.8799 | 0.2459 |
| rain | 1 | 1.0000 | 0.0000 | -0.942 | 2.4 | 1.0000 | 0.9792 |
| rain | 3 | 0.9999 | 0.0002 | -0.947 | 2.4 | 0.9998 | 0.9427 |
| rain | 5 | 0.9991 | 0.0018 | -0.962 | 2.4 | 0.9977 | 0.8549 |
| motion_blur | 1 | 0.9999 | 0.0001 | -0.947 | 2.4 | 0.9999 | 0.9583 |
| motion_blur | 3 | 0.9999 | 0.0002 | -0.947 | 2.4 | 0.9998 | 0.9427 |
| motion_blur | 5 | 0.9999 | 0.0003 | -0.946 | 2.4 | 0.9998 | 0.9271 |
| low_light | 1 | 0.9289 | 0.1421 | -0.715 | 2.6 | 0.7429 | 0.2671 |
| low_light | 3 | 0.5574 | 0.8852 | -0.825 | 2.0 | 0.2212 | 0.1028 |
| low_light | 5 | 0.2669 | 1.4663 | -0.702 | 1.7 | 0.1595 | 0.0907 |


### Model: Clean JEPA

| Family | Severity | Cosine (↑) | Alignment (↓) | Uniformity (↓) | ERank | Linear CKA (↑) | k-NN Jaccard (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| fog | 1 | 0.8736 | 0.2529 | -0.722 | 2.6 | 0.9056 | 0.4097 |
| fog | 3 | 0.1061 | 1.7878 | -0.178 | 2.3 | 0.3413 | 0.2319 |
| fog | 5 | 0.3127 | 1.3746 | -0.006 | 2.7 | 0.6311 | 0.2574 |
| rain | 1 | 1.0000 | 0.0000 | -0.730 | 3.2 | 1.0000 | 0.9844 |
| rain | 3 | 0.9992 | 0.0017 | -0.721 | 3.1 | 0.9997 | 0.9531 |
| rain | 5 | 0.9919 | 0.0162 | -0.704 | 3.0 | 0.9969 | 0.8437 |
| motion_blur | 1 | 0.9999 | 0.0001 | -0.734 | 3.2 | 0.9999 | 0.9479 |
| motion_blur | 3 | 0.9999 | 0.0002 | -0.734 | 3.2 | 0.9998 | 0.9323 |
| motion_blur | 5 | 0.9999 | 0.0002 | -0.734 | 3.2 | 0.9998 | 0.9219 |
| low_light | 1 | 0.3662 | 1.2676 | -0.772 | 4.1 | 0.3846 | 0.1712 |
| low_light | 3 | -0.0115 | 2.0230 | -0.220 | 2.0 | 0.0259 | 0.1076 |
| low_light | 5 | -0.2920 | 2.5840 | -0.039 | 1.8 | 0.0318 | 0.0896 |


### Model: Random ViT

| Family | Severity | Cosine (↑) | Alignment (↓) | Uniformity (↓) | ERank | Linear CKA (↑) | k-NN Jaccard (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| fog | 1 | 0.9121 | 0.1757 | -0.636 | 5.9 | 0.8961 | 0.4666 |
| fog | 3 | 0.4876 | 1.0249 | -0.125 | 3.5 | 0.7306 | 0.3457 |
| fog | 5 | 0.3988 | 1.2025 | -0.012 | 3.4 | 0.8586 | 0.3351 |
| rain | 1 | 1.0000 | 0.0000 | -0.566 | 5.9 | 1.0000 | 0.9948 |
| rain | 3 | 0.9995 | 0.0009 | -0.566 | 5.9 | 0.9997 | 0.9598 |
| rain | 5 | 0.9949 | 0.0102 | -0.564 | 5.8 | 0.9966 | 0.8757 |
| motion_blur | 1 | 0.9988 | 0.0023 | -0.585 | 6.0 | 0.9988 | 0.9025 |
| motion_blur | 3 | 0.9985 | 0.0029 | -0.587 | 6.0 | 0.9986 | 0.8973 |
| motion_blur | 5 | 0.9984 | 0.0032 | -0.588 | 6.0 | 0.9984 | 0.8973 |
| low_light | 1 | 0.7734 | 0.4531 | -0.403 | 4.2 | 0.5950 | 0.2622 |
| low_light | 3 | 0.3377 | 1.3246 | -0.113 | 2.6 | 0.3131 | 0.1380 |
| low_light | 5 | 0.1578 | 1.6844 | -0.015 | 2.6 | 0.2793 | 0.1298 |

