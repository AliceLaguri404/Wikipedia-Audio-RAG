"""
RAG Pipeline Endpoint - Conversational Support
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List
import json
import time
import os

from src.models.schemas import RAGQueryResponse, StatusEnum, RetrievedChunk
from src.services.asr_service import get_asr_service
from src.services.translation_service import get_translation_service
from src.services.Ingest_retrieve_service import get_vectordb_service
from src.services.llm_service import get_llm_service
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)
router = APIRouter()

@router.post("/chat/query", response_model=RAGQueryResponse)
async def rag_query(
    audio_file: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(None),
    chat_history: str = Form("[]"), 
    language_code: str = Form("en-IN"),
    with_diarization: bool = Form(False),
    num_speakers: int = Form(2)
):
    """
    Conversational RAG Pipeline:
    1. Transcribe/Translate
    2. Contextualize Query (Rewrite based on History) 🔄
    3. Decompose & Scrape
    4. Retrieve
    5. Generate Answer
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not audio_file and not text_query:
            raise HTTPException(status_code=400, detail="No input provided")
        
        logger.info("🚀 Starting RAG pipeline")
        
        # --- 1. Transcribe ---
        if audio_file:
            logger.info("🎤 Step 1: Audio transcription")
            temp_audio_path = settings.TEMP_DIR / f"audio_{int(time.time())}_{audio_file.filename}"
            with open(temp_audio_path, "wb") as f:
                f.write(await audio_file.read())
            
            asr_service = get_asr_service()
            asr_result = await asr_service.transcribe_audio(
                str(temp_audio_path), 
                language_code=language_code,
                with_timestamps=True,
                with_diarization=with_diarization,
                num_speakers=num_speakers
            )
            transcribed_text = asr_result["transcribed_text"]
            audio_language = asr_result["language_detected"]
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        else:
            transcribed_text = text_query
            audio_language = "en"
            logger.info(f"📝 Text Input: {transcribed_text}")

        # --- 2. Translate ---
        logger.info("🌐 Step 2: Translation")
        translation_service = get_translation_service()
        translation_result = await translation_service.translate_to_english(
            text=transcribed_text,
            target_language="en-IN"
        )
        english_text = translation_result["translated_text"]

    # --- 3. Query Decomposition ---
        llm_service = get_llm_service(provider=settings.LLM_PROVIDER)
        # decompose_query now returns {"titles": [...], "improvised_queries": [...]}
        decomposition_result = await llm_service.decompose_query(english_text) 
        print(f"decomposition_result: {decomposition_result}")
        target_titles = decomposition_result.get("titles", [english_text])
        search_queries = decomposition_result.get("improvised_queries", [english_text])
        
        logger.info(f"✨ Target Wiki Titles: '{target_titles}'")
        logger.info(f"🔍 Optimized Search Queries: '{search_queries}'")

        # refined_query = english_text
        
        # --- 4. Incremental Ingestion ---
        vectordb_service = get_vectordb_service(backend=settings.VECTOR_DB_BACKEND)
        
        # Use target_titles for scraping as they are exact Wikipedia matches
        logger.info("📥 Step 4: Incremental Ingestion")
        ingestion_results = []
        for title in target_titles:
            logger.info(f"📚 Processing topic: {title}")
            try:
                # Pass a single-item list or just the string depending on your specific method signature
                result = await vectordb_service.process_query_to_vectordb(
                    user_query=title, 
                    collection_name=settings.COLLECTION_NAME
                )
                ingestion_results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to process topic '{title}': {e}")
                continue

        # Combine results for logging or response metadata
        total_new_chunks = sum(res.get("new_chunks_added", 0) for res in ingestion_results)
        logger.info(f"✅ Ingestion complete. Added {total_new_chunks} new chunks across {len(target_titles)} topics.")

        # --- 5. Pure Retrieval ---
        logger.info("🔎 Step 5: Retrieval")
        # Use improvised_queries for retrieval as they are keyword-dense for vector search
        retrieval_result = await vectordb_service.retrieve_for_topics(
            topics=search_queries, 
            collection_name=settings.COLLECTION_NAME
        )
        
        # Convert results to RetrievedChunk objects
        retrieved_chunks = [
            RetrievedChunk(
                text=chunk["document"],
                metadata=chunk["metadata"],
                score=chunk.get("score", 0.0),
                chunk_id=str(chunk["id"])
            ) for chunk in retrieval_result["results"]
        ]
        print(f"retrieved_chunks: {retrieved_chunks}")
        
        # --- 6. Generate Answer ---
        logger.info("🤖 Step 6: Generation")

        context_list = []
        for i, chunk in enumerate(retrieved_chunks):
            source_url = chunk.metadata.get("article_url", "#")
            article_title = chunk.metadata.get("article_title", "Source")
            
            formatted_chunk = f"""
            [Source ID: {i+1}]
            Title: {article_title}
            URL: {source_url}
            Content: {chunk.text}
            """
            context_list.append(formatted_chunk)
            
        context_str = "\n".join(context_list)
        
        full_prompt = f"""
        You are a highly detailed and knowledgeable AI assistant. 
        Your goal is to provide a comprehensive answer to the User Question using ONLY the information provided in the Context.

        CONTEXT:
        {context_str}

        USER QUESTION:
        {english_text}

        CRITICAL INSTRUCTIONS:
        1. **Direct Answer**: Start directly with the answer. Do not use phrases like "Based on the context" or "The provided text states." Speak naturally and confidently.
        2. **Selective Relevance**: If the context contains multiple topics, focus ONLY on the parts that answer "{english_text}". Ignore irrelevant information.
        3. **Citation Style**: Use [Source ID: X] immediately after the sentence or paragraph that uses information from that source. 
        4. **Depth**: Provide a detailed explanation of at least 3-4 paragraphs. If the context allows, break down complex concepts into logical sections.
        5. **No Hallucination**: If the context does not contain the answer, simply state that the information is not available in the current knowledge base.

        RESPONSE:
        """

        answer = await llm_service.generate_answer(
            query=full_prompt, 
            context="", 
        )
        
        processing_time = time.time() - start_time
        return RAGQueryResponse(
            status=StatusEnum.SUCCESS,
            message="Success",
            original_audio_language=audio_language,
            transcribed_text=transcribed_text,
            translated_text=english_text,
            extracted_keyword=english_text, 
            # Returning the titles we attempted to scrape
            wikipedia_article=str(target_titles),
            retrieved_chunks=retrieved_chunks,
            llm_answer=answer,
            processing_time_seconds=round(processing_time, 2),
            cache_hit=False
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))