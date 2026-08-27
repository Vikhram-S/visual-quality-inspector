import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime
from backend.database import Base

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(50), nullable=False)
    issues_json = Column(Text, nullable=False)        # JSON string of detected issues
    image_stats_json = Column(Text, nullable=False)   # JSON string of extracted stats
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
