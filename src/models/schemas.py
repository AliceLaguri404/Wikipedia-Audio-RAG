"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ========== Common Models ==========

class StatusEnum(str, Enum):
    """Status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class BaseResponse(BaseModel):
    """Base response model."""
    status: StatusEnum
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ========== Task 3: ASR Models ==========

class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    language_code: str = Field(
        default="en-IN",
        description="Language code for transcription"
    )
    with_timestamps: bool = Field(
        default=True,
        description="Include timestamps in transcription"
    )
    with_diarization: bool = Field(
        default=False,
        description="Enable speaker diarization"
    )
    num_speakers: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of speakers (if diarization enabled)"
    )


class TranscriptionResponse(BaseResponse):
    """Response model for audio transcription."""
    transcribed_text: str
    language_detected: Optional[str] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None


# ========== Task 4: Translation Models ==========

class TranslationRequest(BaseModel):
    """Request model for text translation."""
    text: str = Field(..., min_length=1, description="Text to translate")
    source_language: str = Field(
        default="auto",
        description="Source language code or 'auto' for detection"
    )
    target_language: str = Field(
        default="en-IN",
        description="Target language code"
    )


class TranslationResponse(BaseResponse):
    """Response model for text translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str


# ========== Task 1: Document Scraping Models ==========

class ScrapingRequest(BaseModel):
    """Request model for Wikipedia scraping."""
    query: str = Field(..., min_length=1, description="Search query for Wikipedia")
    force_refresh: bool = Field(
        default=False,
        description="Force re-scraping even if cached"
    )


class ScrapingResponse(BaseResponse):
    """Response model for Wikipedia scraping."""
    query: str
    article_title: str
    article_url: str
    text_length: int
    cached: bool
    file_path: str


# ========== Task 2: Vector DB Models ==========

class VectorDBCreateRequest(BaseModel):
    """Request model for vector DB creation."""
    document_path: str = Field(..., description="Path to document file")
    collection_name: str = Field(..., description="Collection name")
    chunk_size: Optional[int] = Field(
        default=512,
        ge=128,
        le=2048,
        description="Chunk size in tokens"
    )
    chunk_overlap: Optional[int] = Field(
        default=50,
        ge=0,
        le=256,
        description="Chunk overlap in tokens"
    )
    force_recreate: bool = Field(
        default=False,
        description="Force recreate collection if exists"
    )


class VectorDBCreateResponse(BaseResponse):
    """Response model for vector DB creation."""
    collection_name: str
    total_chunks: int
    embedding_dimension: int
    storage_path: str


class VectorDBStatsResponse(BaseResponse):
    """Response model for vector DB statistics."""
    collections: List[str]
    total_documents: int
    storage_size_mb: float


# ========== Retrieval Models ==========

class RetrievalRequest(BaseModel):
    """Request model for document retrieval."""
    query: str = Field(..., min_length=1, description="Search query")
    collection_name: str = Field(..., description="Collection to search")
    top_k: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of results to retrieve"
    )


class RetrievedChunk(BaseModel):
    """Model for a retrieved chunk."""
    text: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str


class RetrievalResponse(BaseResponse):
    """Response model for document retrieval."""
    query: str
    results: List[RetrievedChunk]
    total_results: int


# ========== Task 5: RAG Pipeline Models ==========

class RAGQueryRequest(BaseModel):
    """Request model for end-to-end RAG query."""
    # Audio is uploaded as file, so this is for text-based testing
    text_query: Optional[str] = Field(
        default=None,
        description="Text query (if not using audio)"
    )
    language_code: str = Field(
        default="en-IN",
        description="Language code for audio transcription"
    )
    with_diarization: bool = Field(
        default=False,
        description="Enable speaker diarization"
    )
    num_speakers: int = Field(default=2, ge=1, le=10)


class RAGQueryResponse(BaseResponse):
    """Response model for RAG query."""
    original_audio_language: Optional[str] = None
    transcribed_text: Optional[str] = None
    translated_text: str
    extracted_keyword: str
    wikipedia_article: str
    retrieved_chunks: List[RetrievedChunk]
    llm_answer: str
    processing_time_seconds: float
    cache_hit: bool


# ========== Health Models ==========

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]


# ========== Error Models ==========

class ErrorDetail(BaseModel):
    """Error detail model."""
    field: Optional[str] = None
    message: str
    error_type: str


class ErrorResponse(BaseModel):
    """Error response model."""
    status: StatusEnum = StatusEnum.ERROR
    message: str
    errors: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.now)