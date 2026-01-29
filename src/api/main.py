"""
FastAPI main application for Voice-Enabled RAG System.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from config.settings import settings
from src.api.routes import health, transcribe, translate, documents, vectordb, chat
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Voice-Enabled RAG System")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    
    # Initialize services (lazy loading will happen on first request)
    logger.info("✅ Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Voice-enabled conversational system with RAG capabilities",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Health"]
)

app.include_router(
    transcribe.router,
    prefix=settings.API_V1_PREFIX,
    tags=["ASR - Task 3"]
)

app.include_router(
    translate.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Translation - Task 4"]
)

app.include_router(
    documents.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Documents - Task 1"]
)

app.include_router(
    vectordb.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Vector DB - Task 2"]
)

app.include_router(
    chat.router,
    prefix=settings.API_V1_PREFIX,
    tags=["RAG Pipeline - Task 5"]
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Voice-Enabled RAG System API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )