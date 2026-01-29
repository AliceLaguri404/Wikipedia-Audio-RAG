"""
Vector Database Endpoints - Task 2 & Advanced RAG
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.services.Ingest_retrieve_service import get_vectordb_service
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)
router = APIRouter()

# --- Request Models ---

class RAGQueryRequest(BaseModel):
    query: str  

class RAGQueryResponse(BaseModel):
    status: str
    original_query: str
    sub_queries: List[str]
    documents_ingested: int
    collection_name: str

class RetrievalRequest(BaseModel):
    query: str
    collection_name: str = "voice_rag_knowledge_base"
    top_k: int = 2

class RetrievalResponse(BaseModel):
    query: str
    results: List[dict]

# --- Endpoints ---

@router.post("/vectordb/ingest-query", response_model=RAGQueryResponse)
async def ingest_complex_query(request: RAGQueryRequest):
    """
    🚀 RAG Ingestion Pipeline (The "Smart" Endpoint):
    1. Receives complex query (e.g., "AQI vs IQ")
    2. Decomposes into sub-topics via LLM
    3. Scrapes Wiki for all topics
    4. Chunks & Ingests into Qdrant (handling duplicates automatically)
    """
    try:
        service = get_vectordb_service(backend=settings.VECTOR_DB_BACKEND)
        
        result = await service.process_query_to_vectordb(
            user_query=request.query,
            collection_name=settings.COLLECTION_NAME
        )
        
        return RAGQueryResponse(
            status="success",
            original_query=result["original_query"],
            sub_queries=result["sub_queries"],
            documents_ingested=result["documents_ingested"],
            collection_name=result["collection_name"]
        )
    except Exception as e:
        logger.error(f"RAG ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectordb/retrieve", response_model=RetrievalResponse)
async def retrieve_context(request: RetrievalRequest):
    """
    Test Endpoint: Retrieve context chunks for a query.
    Used to verify that data was ingested correctly.
    """
    try:
        service = get_vectordb_service(backend=settings.VECTOR_DB_BACKEND)
        topics = [request.query] if isinstance(request.query, str) else request.query
        logger.info(f"🔎 Retrieval requested for {len(topics)} topics: {topics}")
        result = await service.retrieve_for_topics(
            topics=topics,
            collection_name=request.collection_name or settings.COLLECTION_NAME
        )
        
        logger.info(f"✅ Successfully retrieved {len(result['results'])} total chunks.")
        
        return RetrievalResponse(
            query=str(topics),
            results=result["results"]
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectordb/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Maintenance: Delete a collection to start fresh.
    """
    try:
        service = get_vectordb_service(backend=settings.VECTOR_DB_BACKEND)
        service.client.delete_collection(collection_name)
        return {"status": "success", "message": f"Collection '{collection_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))