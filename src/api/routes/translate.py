"""
Translation Endpoint - Task 4
"""
from fastapi import APIRouter, HTTPException

from src.models.schemas import TranslationRequest, TranslationResponse, StatusEnum
from src.services.translation_service import get_translation_service
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text to English using Sarvam AI.
    
    Args:
        request: TranslationRequest with text and language codes
        
    Returns:
        TranslationResponse with translated text
    """
    try:
        logger.info(f"🌐 Translation request (length: {len(request.text)} chars)")
        
        # Get translation service
        translation_service = get_translation_service()
        
        # Translate
        result = await translation_service.translate_to_english(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
        return TranslationResponse(
            status=StatusEnum.SUCCESS,
            message="Translation completed successfully",
            original_text=result["original_text"],
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            target_language=result["target_language"]
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )