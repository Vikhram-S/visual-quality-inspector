# Image Quality & Defect Detection System

A complete, full-stack, AI-powered application for automated image quality scoring and visual defect detection. Designed specifically to run efficiently on low-resource hardware (CPU-only, 8GB RAM) with **zero external API dependencies**.

---

## 📌 Executive Summary & Hardware Optimization Rationale

This project addresses visual defect classification (Blur, Underexposure, Overexposure, Noise, JPEG Corruption) without requiring heavy CUDA/GPU infrastructure or multi-gigabyte neural networks.

- **Hybrid Classical CV + Learned Ensemble:** Uses OpenCV/scikit-image to extract engineered quality features (Laplacian variance, Tenengrad magnitude, FFT blur ratios, shadow/highlight clipping %, noise residual variance, and 8x8 DCT blockiness index).
- **Lightweight Inference:** Trains a Random Forest multi-issue classifier + Gradient Boosting overall quality head on top of extracted feature vectors. Model footprint is **< 3 MB** and runs inference in **< 15ms per image** on CPU.
- **Explainable AI:** Does not act as a black box. Reports exact mathematical feature metrics alongside human-readable decision explanations (e.g. *"Blur detected: Laplacian variance = 12.3 (below threshold 100.0)"*).
- **Synthetic Degradation Strategy:** Programmatically generates training data from clean base scenes using controlled physical degradation models (Gaussian blur, LUT exposure curves, additive noise, JPEG byte compression), eliminating the need for large manual datasets.

---

## 🏗️ Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                                 REACT + VITE FRONTEND                             |
|  - Modern Dark Glassmorphic UI                                                    |
|  - Drag & Drop Upload with Client Validation                                      |
|  - Real-time Score Radial Gauge, Issue Severity Badges, & Feature Diagnostics     |
|  - History Log View with Stored Image Previews                                    |
+----------------------------------------+------------------------------------------+
                                         | REST API (HTTP / JSON / FormData)
                                         v
+-----------------------------------------------------------------------------------+
|                               FASTAPI BACKEND SERVICE                             |
|  - POST /api/analyze         (File validation, ML Pipeline execution)             |
|  - GET  /api/analyses        (Paginated analysis history)                         |
|  - GET  /api/analyses/{id}   (Single record retrieval)                          |
|  - GET  /api/images/{id}     (Stored image stream)                                |
|  - GET  /api/health          (Status check)                                       |
+--------------------+-----------------------------------+--------------------------+
                     |                                   |
                     v                                   v
+------------------------------------+   +------------------------------------------+
|       ML FEATURE & INFERENCE       |   |             SQLITE DATABASE              |
|  - Feature Extractor (OpenCV)      |   |  - SQLAlchemy ORM                        |
|  - Random Forest Classifiers       |   |  - Stores record metadata, quality score, |
|  - Rule-based Explainability Engine|   |    issues JSON, stats, & file paths     |
+------------------------------------+   +------------------------------------------+
```

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.11/3.14), Uvicorn, SQLAlchemy, Pydantic
- **ML Engine:** OpenCV (`opencv-python-headless`), scikit-learn, scikit-image, NumPy, Joblib
- **Database:** SQLite (file-based, zero-config, swappable to PostgreSQL via `DATABASE_URL`)
- **Frontend:** React, Vite, Lucide Icons, Custom Vanilla CSS Design System
- **Containerization:** Docker, Docker Compose, Multi-stage Nginx static build

---

## 🚀 Quickstart & Setup Instructions

### Option A: Local Development (Recommended for rapid dev)

#### 1. Prerequisites
- Python 3.10+ installed
- Node.js 18+ and npm installed

#### 2. Backend Setup
```bash
# Clone repo & navigate to workspace
cd e:\IIIT-H_SE_INTERN

# Install backend Python dependencies
pip install -r backend/requirements.txt

# (Optional) Generate synthetic dataset & train model
python ml/generate_dataset.py
python ml/train.py
python ml/evaluate.py

# Start FastAPI backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Backend will run at `http://127.0.0.1:8000`. API Documentation is available at `http://127.0.0.1:8000/docs`.

#### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Frontend will run at `http://localhost:5173`.

---

### Option B: Docker Compose Deployment

To build and run both the backend and frontend in containerized environments:

```bash
docker compose up --build -d
```

- **Frontend Application:** Access at `http://localhost`
- **Backend API:** Access at `http://localhost:8000`

---

## 📊 ML Pipeline & Model (Re)Training

To retrain the ML model from scratch:

```bash
# Step 1: Generate synthetic clean + degraded dataset (~1920 images)
python ml/generate_dataset.py

# Step 2: Extract features & train Random Forest classifiers
python ml/train.py

# Step 3: Run evaluation on unseen held-out test set
python ml/evaluate.py
```

Evaluation outputs will be generated in `/evaluation`:
- `evaluation/metrics.json`: Accuracy, precision, recall, F1 per defect type
- `evaluation/confusion_matrix.png`: Multi-class confusion matrix plot
- `evaluation/evaluation_report.md`: Markdown summary of metrics and limitations

---

## 📡 API Documentation & Sample Requests

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/api/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-08-27T15:08:09.620924"
}
```

### 2. Analyze Image
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@sample_images/sample_blur.jpg"
```

**Response:**
```json
{
  "id": "b33ccd09-7fc1-49d6-bf54-d7eb8b1c1269",
  "filename": "sample_blur.jpg",
  "quality_score": 65.0,
  "quality_label": "DEFECTIVE",
  "issues": [
    {"type": "blur", "severity": "high", "confidence": 0.99}
  ],
  "image_stats": {
    "laplacian_var": 8.35,
    "tenengrad_val": 1209.08,
    "fft_blur_ratio": 0.96,
    "mean_luminance": 209.52,
    "shadow_clip_pct": 0.0,
    "highlight_clip_pct": 17.06,
    "noise_variance": 1.55,
    "blockiness_index": 1.18,
    "entropy": 5.93
  },
  "explanation": "Blur detected: Laplacian variance = 8.4 (below optimal sharpness threshold 100.0).",
  "created_at": "2026-08-27T15:08:16.370407"
}
```

### 3. List Past Analyses (Paginated)
```bash
curl -X GET "http://127.0.0.1:8000/api/analyses?page=1&limit=10"
```

---

## 📈 Evaluation & Performance Results

Evaluated on an unseen held-out synthetic test set (**320 test samples**):

| Defect Category | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| **Blur** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Underexposure** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Overexposure** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Noise** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Corruption / Blockiness** | 100.0% | 100.0% | 100.0% | 100.0% |

**Overall Multi-class Test Accuracy:** **100.0%**

---

## ⚠️ Known Limitations & Failure Cases

1. **High-Frequency Noise vs. Mild Blur interaction:** Extremely fine Gaussian noise can artificially elevate Laplacian variance, slightly overestimating sharpness on blurry images with heavy noise.
2. **Intentional High Dynamic Range (HDR) Highlights:** Bright specular highlights (e.g. sunsets or light bulbs) might trigger light overexposure warnings if highlight clipping exceeds 25%.
3. **Synthetic Domain Gap:** Controlled synthetic degradations provide zero-cost labeling but do not cover physical lens distortion or chromatic aberrations.
