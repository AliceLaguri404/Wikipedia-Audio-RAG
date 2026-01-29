"""
Configuration settings for Voice-Enabled RAG System.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    APP_NAME: str = "Voice-Enabled RAG System"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    ASR_MODEL_ID: str = "ai4bharat/indic-conformer-600m-multilingual" 
    ASR_DEVICE: str = "mps"
    HF_API_TOKEN: str =  ""
    # Sarvam AI API
    SARVAM_API_KEY: str =  ""
    SARVAM_ASR_MODEL: str = "saaras:v3"
    SARVAM_DEFAULT_LANGUAGE: str = "en-IN"
    
    # LLM Configuration
    LLM_PROVIDER: str = "gemini"  # "gemini" or "openai"
    
    # Gemini API (Recommended - FREE!)
    GEMINI_API_KEY: str =  ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    # "gemini-2.5-flash"
    # "gemini-2.5-flash-lite"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_TOKENS: int = 2048
    
    # OpenAI API (Alternative - Paid)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2048
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int =200
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Vector Database Configuration
    VECTOR_DB_BACKEND: str = "qdrant"  
    QDRANT_PATH: str = "./storage/qdrant_db"  # For local Qdrant
    QDRANT_URL: str = "http://localhost:6333"  # For remote Qdrant (optional)
    COLLECTION_NAME: str = "voice_rag_knowledge_base"
    TOP_K_RESULTS: int = 2
    
    # Storage Paths
    STORAGE_DIR: Path = Path("./storage")
    DOCUMENTS_DIR: Path = Path("./storage/documents")
    TEMP_DIR: Path = Path("./storage/temp")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Caching
    ENABLE_DOCUMENT_CACHE: bool = True
    ENABLE_COLLECTION_CACHE: bool = True
    CACHE_TTL: int = 86400  # seconds 31536000  # 1 year in seconds
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    # ALLOWED_AUDIO_FORMATS: list = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    ALLOWED_AUDIO_FORMATS: list = ['.wav', '.mp3', '.aac', '.m4a', '.webm', '.ogg', ".flac",".opus"]
    
    # Wikipedia Scraping
    WIKIPEDIA_API_URL: str = "https://en.wikipedia.org/w/api.php"
    SCRAPER_USER_AGENT: str = "VoiceRAGBot/1.0 (Educational Purpose)"
    SCRAPER_TIMEOUT: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    settings = Settings()
    
    # Create necessary directories
    settings.STORAGE_DIR.mkdir(exist_ok=True)
    settings.DOCUMENTS_DIR.mkdir(exist_ok=True)
    settings.TEMP_DIR.mkdir(exist_ok=True)
    
    return settings


# Export for easy imports
settings = get_settings()