# Image Quality and Defect Detection System

A full-stack AI-powered application for automated image quality assessment, defect classification, and spatial defect localization. Built to run on CPU-only, low-resource hardware, with no dependency on external AI or vision APIs.

---

## Executive Summary

This system performs automated visual quality assessment and defect classification (blur, underexposure, overexposure, noise, and JPEG blockiness corruption) using a hybrid classical computer vision and machine learning pipeline, without requiring GPU infrastructure or large pretrained neural network weights.

- **Hybrid classical CV and learned ensemble.** OpenCV is used to extract engineered quality features: Laplacian variance, Tenengrad gradient magnitude, FFT-based blur ratio, shadow and highlight clipping percentage, noise residual variance, Immerkaer noise estimation, and an 8x8 DCT blockiness index.
- **ML-first inference.** A Random Forest ensemble (per-issue binary classifiers) and a Gradient Boosting classifier (overall quality label) operate on the extracted feature vectors. Model footprint is under 3 MB, with inference latency under 15 ms per image on CPU.
- **Explainability and spatial localization.** The system produces human-readable decision rationales and an 8x8 spatial heatmap localizing the image regions contributing to a detected defect.
- **Leakage-free, multi-distribution evaluation.** Model performance is reported separately on an unseen synthetic test split and an independent real-world photographic holdout set, with the holdout images committed to the repository to keep evaluation reproducible offline.

---

## Model Decision Architecture

The inference engine (`backend/ml_engine.py`) follows a hybrid ML-first architecture with a documented safety-net fallback:

1. **Primary decision signal (machine learning).** Independent Random Forest binary classifiers estimate a defect probability, P(ml), for each issue class. An issue is primarily flagged when P(ml) >= 0.50.
2. **Safety-net fallback (documented physical bounds).** Raw feature thresholds act only as a safety net for extreme out-of-distribution cases — for example, near-zero luminance (< 15.0), fully saturated highlights (> 245.0), or unreadable/corrupted bitstreams. This prevents zero-shot false negatives on extreme inputs without allowing hand-tuned rules to override the learned model on ordinary images.

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                              REACT + VITE FRONTEND                                |
|  - Image upload with client-side MIME and size validation                        |
|  - Interactive quality score display and detected-issue breakdown                |
|  - Spatial defect heatmap overlay (8x8 grid patch localization)                  |
|  - Paginated analysis history view                                               |
+----------------------------------------+------------------------------------------+
                                          | REST API (HTTP / JSON / FormData)
                                          v
+-----------------------------------------------------------------------------------+
|                             FASTAPI BACKEND SERVICE                               |
|  - POST /api/analyze         Validation, feature extraction, inference, heatmap  |
|  - GET  /api/analyses        Paginated analysis history                          |
|  - GET  /api/analyses/{id}   Single record and heatmap retrieval                 |
|  - GET  /api/images/{id}     Stored image retrieval                              |
|  - GET  /api/health          Service and model status check                      |
+--------------------+-----------------------------------+--------------------------+
                      |                                   |
                      v                                   v
+------------------------------------+   +------------------------------------------+
|      ML FEATURE AND INFERENCE      |   |              SQLITE DATABASE             |
|  - Feature extraction (OpenCV)     |   |  - SQLAlchemy ORM                        |
|  - Random Forest / Gradient        |   |  - Persists analysis records, image      |
|    Boosting classifiers            |   |    statistics, detected issues, and      |
|  - Defect heatmap generator        |   |    stored file references                |
+------------------------------------+   +------------------------------------------+
```

---

## Tech Stack

- **Backend:** FastAPI, Pytest, HTTPX, SQLAlchemy, Pydantic v2, Uvicorn
- **ML Engine:** OpenCV (opencv-python-headless), scikit-learn, NumPy, Matplotlib, Joblib
- **Frontend:** React 19, Vite, Vitest, React Testing Library
- **CI / Deployment:** GitHub Actions, Docker, Docker Compose, Nginx

---

## Setup Instructions

### Option A: Local Development

**Backend**
```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
The backend runs at `http://127.0.0.1:8000`; interactive API documentation is available at `http://127.0.0.1:8000/docs`.

**Frontend**
```bash
cd frontend
npm install
npm run dev
```
The frontend runs at `http://localhost:5173`.

### Option B: Docker Compose

```bash
docker compose up --build -d
```
- Frontend: `http://localhost`
- Backend API: `http://localhost:8000`

Note: the Docker configuration has been manually reviewed for correctness (base images, exposed ports, environment variables, inter-service networking), but has not been smoke-tested against a local Docker daemon due to environment constraints on the development machine used for this submission.

---

## Reproducing the Reported Evaluation Results

The evaluation metrics in `metrics.json` and `evaluation_report.md` can be reproduced from a clean clone without internet access:

```bash
python ml/generate_dataset.py   # generate procedural splits; load committed real-world source photos
python ml/train.py              # train the Random Forest / Gradient Boosting ensemble
python ml/evaluate.py           # run the dual-distribution evaluation
```

The 30 base photographs used for the real-world holdout set are committed under `ml/dataset/real_holdout_source/` (sourced from Unsplash, used under its free-to-use license). Committing these images ensures dataset generation and evaluation are deterministic and reproducible offline, with no silent fallback to synthetic substitutes.

---

## Automated Testing

**Backend**
```bash
python -m pytest backend/tests
```

**Frontend**
```bash
cd frontend
npm test
```

---

## Model Evaluation and Results

Model performance is reported on two independent datasets to eliminate data leakage and assess genuine generalization:

1. **Unseen synthetic test split (800 samples).** Generated from a disjoint random seed range (5000+) across 20 distinct procedural base pattern families, with no overlap with the training seed range.
2. **Real-world holdout set (480 samples).** Genuine photographs sourced from Unsplash, subjected to controlled, labeled degradations.

### Generalization Summary

| Defect Category | Synthetic Test Acc. | Synthetic F1 | Real Holdout Acc. | Real Holdout F1 |
| --- | --- | --- | --- | --- |
| Blur | 98.8% | 96.6% | 99.2% | 97.8% |
| Underexposure | 98.6% | 96.2% | 93.1% | 83.1% |
| Overexposure | 98.8% | 96.6% | 90.8% | 67.6% |
| Noise | 100.0% | 100.0% | 100.0% | 100.0% |
| Corruption / Blockiness | 98.4% | 95.8% | 98.3% | 95.3% |

- Overall synthetic test accuracy: **92.38%** (800 samples)
- Overall real-world holdout accuracy: **85.00%** (480 samples)
- Real-world dataset provenance: **30/30 genuine photographs**, 0 synthetic fallbacks

---

## Limitations

1. **Synthetic-to-real domain gap.** Accuracy on the synthetic test split (92.38%) exceeds accuracy on the real-world holdout (85.00%). Procedural generation produces exact, noise-free ground-truth boundaries, while real photographs introduce complex scene texture, non-uniform shadow structure, and organic lens aberration that shift the feature distribution away from the training data.
2. **Overexposure is the weakest category on real photographs.** Real scenes with bright skies or specular highlights can register high mean luminance without severe highlight clipping, causing feature overlap with clean images under moderate overexposure; under severe overexposure, flat saturated regions are occasionally misclassified as JPEG block corruption by the blockiness estimator.
3. **Dense high-frequency textures can inflate raw blur-related statistics.** Backgrounds such as foliage or brickwork raise raw Laplacian variance independent of actual sharpness, which is why the ML classifier probability (rather than the raw statistic alone) is used as the primary blur-detection signal.

---

## License and Attribution

Real-world holdout images sourced from Unsplash, used under the Unsplash license (free to use, attribution not required).
