import os
import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def create_dummy_jpeg():
    """Generates valid 100x100 RGB JPEG image bytes."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.circle(img, (50, 50), 30, (255, 0, 0), -1)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_analyze_valid_image():
    image_bytes = create_dummy_jpeg()
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.jpg"
    assert isinstance(data["quality_score"], float)
    assert data["quality_label"] in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
    assert isinstance(data["issues"], list)
    assert "laplacian_var" in data["image_stats"]
    assert "explanation" in data
    assert "heatmap_base64" in data

def test_analyze_corrupted_unreadable_file():
    corrupted_bytes = b"NOT_AN_IMAGE_HEADER_BYTES_RANDOM_CORRUPTION_12345"
    files = {"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    # Must return graceful DEFECTIVE response, not a 500 server error
    assert response.status_code == 201
    data = response.json()
    assert data["quality_label"] == "DEFECTIVE"
    assert any(issue["type"] == "corrupted" for issue in data["issues"])

def test_analyze_oversized_file():
    large_bytes = b"0" * (16 * 1024 * 1024)
    files = {"file": ("large.jpg", large_bytes, "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"].lower()

def test_analyze_wrong_file_type():
    txt_bytes = b"Hello world text file content"
    files = {"file": ("document.txt", txt_bytes, "text/plain")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code in [400, 415]

def test_list_analyses_pagination():
    response = client.get("/api/analyses?page=1&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["page"] == 1
    assert data["limit"] == 5
    assert isinstance(data["items"], list)

def test_get_analysis_nonexistent_id():
    response = client.get("/api/analyses/nonexistent-id-0000-0000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
