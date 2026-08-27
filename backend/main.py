import os
import sys
import uuid
import json
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.database import Base, engine, get_db
from backend.models import AnalysisRecord
from backend.schemas import AnalysisResponse, PaginatedAnalysisResponse, HealthResponse
from backend.ml_engine import ml_engine

# Initialize Database tables
Base.metadata.create_all(bind=engine)

# Directory for stored uploaded images
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Image Quality & Defect Detection API",
    description="Full-stack automated visual defect detection & explainable image quality scoring service.",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Loads ML model ONCE at server startup."""
    ml_engine.load_model()

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": ml_engine.is_loaded,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

@app.post("/api/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts an uploaded image, validates format/size, extracts quality features,
    runs defect detection inference, persists result in SQLite, and returns structured result.
    """
    # 1. Content-Type Pre-validation
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        # Fallback check extension if content-type is generic octet-stream
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{file.content_type}'. Allowed types: JPG, PNG, WEBP, BMP, TIFF."
            )

    # 2. Read File Bytes with Size Check
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded."
        )
    
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum limit of 15MB."
        )

    # 3. Save File to Disk Storage
    record_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    saved_filename = f"{record_id}{ext}"
    saved_file_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    with open(saved_file_path, "wb") as f:
        f.write(image_bytes)

    # 4. Run Analysis Pipeline
    analysis_result = ml_engine.analyze_image_bytes(image_bytes)

    # 5. Persist to Database
    created_at_dt = datetime.datetime.utcnow()
    record = AnalysisRecord(
        id=record_id,
        filename=file.filename,
        file_path=saved_file_path,
        quality_score=analysis_result["quality_score"],
        quality_label=analysis_result["quality_label"],
        issues_json=json.dumps(analysis_result["issues"]),
        image_stats_json=json.dumps(analysis_result["image_stats"]),
        explanation=analysis_result["explanation"],
        created_at=created_at_dt
    )
    
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": record.filename,
        "quality_score": record.quality_score,
        "quality_label": record.quality_label,
        "issues": analysis_result["issues"],
        "image_stats": analysis_result["image_stats"],
        "explanation": record.explanation,
        "created_at": created_at_dt.isoformat()
    }

@app.get("/api/analyses", response_model=PaginatedAnalysisResponse)
def list_analyses(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    total = db.query(AnalysisRecord).count()
    offset = (page - 1) * limit
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).offset(offset).limit(limit).all()

    items = []
    for r in records:
        items.append({
            "id": r.id,
            "filename": r.filename,
            "quality_score": r.quality_score,
            "quality_label": r.quality_label,
            "issues": json.loads(r.issues_json),
            "image_stats": json.loads(r.image_stats_json),
            "explanation": r.explanation,
            "created_at": r.created_at.isoformat()
        })

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items
    }

@app.get("/api/analyses/{record_id}", response_model=AnalysisResponse)
def get_analysis(record_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    return {
        "id": record.id,
        "filename": record.filename,
        "quality_score": record.quality_score,
        "quality_label": record.quality_label,
        "issues": json.loads(record.issues_json),
        "image_stats": json.loads(record.image_stats_json),
        "explanation": record.explanation,
        "created_at": record.created_at.isoformat()
    }

@app.get("/api/images/{record_id}")
def get_image(record_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record or not os.path.exists(record.file_path):
        raise HTTPException(status_code=404, detail="Image file not found.")

    return FileResponse(record.file_path)
