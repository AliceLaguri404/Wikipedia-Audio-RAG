"""
Health check endpoint.
"""
from fastapi import APIRouter
from datetime import datetime

from src.models.schemas import HealthResponse
from config.settings import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now(),
        services={
            "asr": "ready",
            "translation": "ready",
            "scraper": "ready",
            "vectordb": "ready",
            "llm": "ready",
            "deduplication": "ready"
        }
    )