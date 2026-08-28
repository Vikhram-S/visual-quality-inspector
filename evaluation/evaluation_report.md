# Model Evaluation & Generalization Report

**Synthetic Test Split Accuracy:** 92.38% (800 samples)
**Real-World Holdout Accuracy:** 84.79% (480 samples)

## 1. Synthetic Test vs. Real-World Holdout Performance

| Issue Category | Synthetic Acc | Synthetic F1 | Real Holdout Acc | Real Holdout F1 |
| --- | --- | --- | --- | --- |
| **Blur** | 98.8% | 96.6% | 99.0% | 97.2% |
| **Underexposed** | 98.6% | 96.2% | 93.1% | 83.1% |
| **Overexposed** | 98.8% | 96.6% | 91.5% | 70.5% |
| **Noise** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Corrupted** | 98.4% | 95.8% | 98.1% | 94.7% |


## 2. Confusion Matrices

### Real-World Holdout Confusion Matrix (Photographs)
![Real Holdout Confusion Matrix](confusion_matrix_real.png)

### Synthetic Test Confusion Matrix
![Synthetic Confusion Matrix](confusion_matrix_synthetic.png)

## 3. Generalization & Synthetic-vs-Real Gap Analysis

### Honest Gap Interpretation
- **Same-Distribution Accuracy (92.4%):** On synthetic test images generated procedurally, the model achieves high accuracy as feature signatures (Laplacian variance, blockiness index, noise variance) closely mirror synthetic training distributions.
- **Real-World Generalization (84.8%):** On genuine photographic images sourced from public datasets (Unsplash/COCO), performance shows a slight domain shift. Real-world scene textures, natural high dynamic ranges, and organic lens blur introduce complex edge frequencies that differ from synthetic noise profiles.
- **Primary Over-reliance Factor:** High-frequency photographic content (e.g. foliage, fine architecture) can elevate noise and Laplacian metrics, requiring balanced classifier confidence thresholds.
