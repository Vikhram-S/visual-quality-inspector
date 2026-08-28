from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class IssueItem(BaseModel):
    type: str = Field(..., example="noise")
    severity: str = Field(..., example="low")  # "low", "medium", "high"
    confidence: float = Field(..., example=0.71)

class AnalysisResponse(BaseModel):
    id: str
    filename: str
    quality_score: float = Field(..., example=82.0)
    quality_label: str = Field(..., example="ACCEPTABLE")
    issues: List[IssueItem]
    image_stats: Dict[str, float]
    explanation: str
    created_at: str
    heatmap_base64: Optional[str] = None
    heatmap_grid: Optional[Any] = None

class PaginatedAnalysisResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[AnalysisResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
