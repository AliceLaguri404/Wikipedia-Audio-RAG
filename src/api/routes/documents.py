"""
Document Scraping Endpoints - Task 1
"""
from fastapi import APIRouter, HTTPException
from typing import List

from src.models.schemas import ScrapingRequest, ScrapingResponse, StatusEnum
from src.services.scraper_service import get_scraper_service
from src.core.deduplication import get_deduplication_service
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)
router = APIRouter()


@router.post("/documents/scrape", response_model=ScrapingResponse)
async def scrape_wikipedia(request: ScrapingRequest):
    """
    Scrape Wikipedia article for given query.
    
    Args:
        request: ScrapingRequest with query and options
        
    Returns:
        ScrapingResponse with article details
    """
    try:
        logger.info(f"📚 Scraping request for: {request.query}")
        
        # Get scraper service
        scraper_service = get_scraper_service()
        
        # Scrape article
        result = await scraper_service.scrape_wikipedia(
            query=request.query,
            force_refresh=request.force_refresh,
            max_retries=3
        )
        
        return ScrapingResponse(
            status=StatusEnum.SUCCESS,
            message="Article scraped successfully" if not result["cached"] else "Retrieved from cache",
            query=result["query"],
            article_title=result["article_title"],
            article_url=result["article_url"],
            text_length=result["text_length"],
            cached=result["cached"],
            file_path=result["file_path"]
        )
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scraping error: {str(e)}"
        )


@router.get("/documents")
async def list_documents():
    """
    List all cached documents.
    
    Returns:
        List of cached documents
    """
    try:
        dedup_service = get_deduplication_service()
        
        documents = []
        for doc_hash, doc_info in dedup_service.cache.get("documents", {}).items():
            documents.append({
                "keyword": doc_info["keyword"],
                "article_title": doc_info["article_title"],
                "file_path": doc_info["file_path"],
                "cached_at": doc_info["cached_at"]
            })
        
        return {
            "status": "success",
            "total_documents": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@router.delete("/documents/cache")
async def clear_document_cache():
    """
    Clear document cache.
    
    Returns:
        Success message
    """
    try:
        dedup_service = get_deduplication_service()
        
        count = len(dedup_service.cache.get("documents", {}))
        dedup_service.cache["documents"] = {}
        dedup_service._save_cache()
        
        return {
            "status": "success",
            "message": f"Cleared {count} cached documents"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )