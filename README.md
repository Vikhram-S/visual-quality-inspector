# Image Quality & Defect Detection System

A full-stack, AI-powered application for automated visual defect detection, explainable image quality scoring, and spatial defect localization. Designed specifically to run efficiently on low-resource hardware (CPU-only, 8GB RAM) with **zero external API dependencies**.

---

## 📌 Executive Summary & Architecture Rationale

This project addresses visual defect classification (Blur, Underexposure, Overexposure, Noise, JPEG Blockiness Corruption) without requiring heavy CUDA/GPU infrastructure or multi-gigabyte neural network weights.

- **Hybrid Classical CV + Learned Ensemble:** Uses OpenCV to extract engineered quality features (Laplacian variance, Tenengrad magnitude, FFT blur ratios, shadow/highlight clipping %, noise residual variance, Immerkaer noise estimation, and 8x8 DCT blockiness index).
- **ML-First Inference:** Trains a Random Forest multi-issue ensemble + Gradient Boosting overall classifier on top of extracted feature vectors. Model footprint is **< 3 MB** and executes inference in **< 15ms per image** on CPU.
- **Explainable AI & Spatial Heatmaps:** Provides human-readable decision rationales alongside an 8x8 spatial defect localization heatmap overlaying low-quality image regions.
- **Leakage-Free Multi-Distribution Evaluation:** Evaluates model performance across both unseen synthetic test splits and an independent real-world photographic holdout dataset cached locally in the repository.

---

## 🧠 Model Decision Architecture

The inference engine (`backend/ml_engine.py`) follows a **Hybrid ML-First + Safety Net** paradigm:

1. **Primary Decision Signal (Machine Learning):**
   - Individual Random Forest binary classifiers predict defect probability ($P_{ml}$) for each defect class.
   - An issue is primarily detected if **$P_{ml} \ge 0.50$**.
2. **Safety Net Fallback (Documented Physical Bounds):**
   - Raw feature metrics act strictly as a safety net for extreme out-of-distribution edge cases (e.g. completely pitch-black images with luminance $< 15.0$, pure white blown-out pixels with luminance $> 245.0$, or unreadable bitstream corruptions).
   - Prevents zero-shot false negatives while preventing hand-tuned rules from overriding learned model patterns.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 REACT + VITE FRONTEND                             |
|  - Dark Glassmorphic Interface & Interactive Radial Score Gauge                   |
|  - Drag & Drop Image Upload with Client-Side MIME / Size Validation               |
|  - Spatial Defect Heatmap Overlay Toggle (8x8 Grid Patch Localization)            |
|  - Paginated History Explorer with SQLite Result Synchronization                 |
+----------------------------------------+------------------------------------------+
                                         | REST API (HTTP / JSON / FormData)
                                         v
+-----------------------------------------------------------------------------------+
|                               FASTAPI BACKEND SERVICE                             |
|  - POST /api/analyze         (Validation, Feature Extraction, Inference, Heatmap) |
|  - GET  /api/analyses        (Paginated analysis history)                         |
|  - GET  /api/analyses/{id}   (Single record & dynamic heatmap retrieval)          |
|  - GET  /api/images/{id}     (Disk storage image stream)                          |
|  - GET  /api/health          (Engine status check)                                |
+--------------------+-----------------------------------+--------------------------+
                     |                                   |
                     v                                   v
+------------------------------------+   +------------------------------------------+
|       ML FEATURE & INFERENCE       |   |             SQLITE DATABASE              |
|  - Feature Extractor (OpenCV)      |   |  - SQLAlchemy ORM                        |
|  - Random Forest Classifiers       |   |  - Persists analysis records, stats,    |
|  - Defect Heatmap Overlay Generator|   |    issues JSON, and saved file paths      |
+------------------------------------+   +------------------------------------------+
```

---

## 🛠️ Tech Stack & Dependencies

- **Backend:** FastAPI, PyTest, HTTPX, SQLAlchemy, Pydantic v2, Uvicorn
- **ML Engine:** OpenCV (`opencv-python-headless`), scikit-learn, NumPy, Matplotlib, Joblib
- **Frontend:** React 19, Vite 8, Vitest, React Testing Library, Lucide Icons, Vanilla CSS
- **CI / Containerization:** GitHub Actions, Docker, Docker Compose, Nginx

---

## 🚀 Quickstart & Setup Instructions

### Option A: Local Development

#### 1. Backend Setup
```bash
# Install backend Python dependencies
pip install -r requirements.txt

# Start FastAPI backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Backend runs at `http://127.0.0.1:8000`. Interactive Swagger API Docs available at `http://127.0.0.1:8000/docs`.

#### 2. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Frontend runs at `http://localhost:5173`.

---

### Option B: Docker Compose Deployment

```bash
docker compose up --build -d
```
- **Frontend Web App:** `http://localhost`
- **Backend API:** `http://localhost:8000`

---

## 🔁 How to Reproduce Our Exact Evaluation Numbers

To reproduce the exact evaluation metrics (`metrics.json` and `evaluation_report.md`) from a clean offline clone without requiring internet access:

```bash
# Step 1: Generate procedural splits & load committed real-world source photos
python ml/generate_dataset.py

# Step 2: Train Random Forest classifiers & overall ensemble
python ml/train.py

# Step 3: Run comprehensive dual-distribution evaluation
python ml/evaluate.py
```

*Note on Data Provenance:* The 30 clean base photographs for the real-world holdout dataset are committed directly to `ml/dataset/real_holdout_source/` (sourced from Unsplash, free to use per the Unsplash license). This ensures dataset generation executes deterministically offline without silent network fallbacks.

---

## 🧪 Automated Testing Suite

### Backend Pytest Suite
```bash
python -m pytest backend/tests
```

### Frontend Vitest Suite
```bash
cd frontend
npm test
```

---

## 📈 Model Evaluation & Empirical Results

The system is evaluated across two distinct datasets to eliminate data leakage and assess genuine domain generalization:

1. **Unseen Synthetic Test Split (800 samples):** Generated using separate random seed ranges (`5000+`) across 20 distinct procedural base pattern families.
2. **Real-World Holdout Set (480 samples):** Real photographs sourced from Unsplash subjected to controlled degradations.

### Generalization Metrics Summary

| Defect Category | Synthetic Test Acc | Synthetic F1 | Real Holdout Acc | Real Holdout F1 |
| --- | --- | --- | --- | --- |
| **Blur** | 98.8% | 96.6% | 99.0% | 97.2% |
| **Underexposure** | 98.6% | 96.2% | 93.1% | 83.1% |
| **Overexposure** | 98.8% | 96.6% | 91.5% | 70.5% |
| **Noise** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Corruption / Blockiness** | 98.4% | 95.8% | 98.1% | 94.7% |

- **Overall Synthetic Test Accuracy:** **92.38%** (800 samples)
- **Overall Real-World Holdout Accuracy:** **85.00%** (480 samples)
- **Real-World Dataset Provenance:** **30/30 genuine photographs** (0 synthetic fallbacks)

---

## ⚠️ Known Limitations & Honesty Note

1. **Synthetic-to-Real Domain Gap:**
   Model accuracy on synthetic test splits (**92.38%**) exceeds accuracy on real photographic holdouts (**85.00%**). Procedural data generation provides exact ground-truth boundaries, whereas real photographs feature complex scene textures, organic shadows, and non-uniform lens aberrations that create feature distribution shifts.
2. **Per-Class Error Analysis (Overexposure):**
   Real photographic scenes containing bright background skies or specular reflections introduce high mean luminance without highlight clipping. When subjected to light overexposure, feature overlap occurs with clean photographic scenes.
3. **High-Frequency Background Textures:**
   Dense photographic elements (such as foliage or architectural brickwork) introduce high spatial variance in raw Laplacian calculations, requiring the ML classifier probability ($P_{ml} \ge 0.50$) to filter false positive blur calls.
