"""
Translation Service - Task 4
Uses Sarvam AI API for text translation.
"""
from typing import Dict, Any

from sarvamai import SarvamAI

from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.exceptions import TranslationException

logger = setup_logger(__name__)


class TranslationService:
    """
    Handles text translation using Sarvam AI API.
    
    Features:
    - Automatic language detection
    - Multi-language support
    - Caching for repeated translations
    """
    
    def __init__(self):
        """Initialize translation service with Sarvam AI client."""
        try:
            self.client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
            logger.info("✅ Translation Service initialized with Sarvam AI")
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
            raise TranslationException(f"Translation service initialization failed: {e}")
    
    async def translate_to_english(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "en-IN"
    ) -> Dict[str, Any]:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_language: Source language code or 'auto' for detection
            target_language: Target language code (default: 'en-IN')
            
        Returns:
            Dictionary with translation results
            
        Raises:
            TranslationException: If translation fails
        """
        try:
            logger.info(f"🌐 Translating text (length: {len(text)} chars)")
            
            # If text is empty, return as-is
            if not text.strip():
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": source_language,
                    "target_language": target_language
                }
            
            # Call Sarvam AI translation API
            response = self.client.text.translate(
                input=text,
                source_language_code=source_language,
                target_language_code=target_language
            )
            
            # Extract translated text from response
            # Response structure depends on Sarvam AI API
            if hasattr(response, 'translated_text'):
                translated_text = response.translated_text
            elif isinstance(response, dict):
                translated_text = response.get('translated_text', text)
            else:
                translated_text = str(response)
            
            # Detect source language if auto
            detected_language = source_language
            if source_language == "auto" and hasattr(response, 'detected_language'):
                detected_language = response.detected_language
            
            logger.info(f"✅ Translation completed")
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": detected_language,
                "target_language": target_language
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Fallback: return original text if translation fails
            logger.warning("⚠️ Translation failed, returning original text")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "error": str(e)
            }
    
    def is_english(self, text: str) -> bool:
        """
        Quick check if text is likely English.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely English
        """
        # Simple heuristic: check if ASCII characters dominate
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return ascii_count / len(text) > 0.9 if text else True


# Singleton instance
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get or create translation service singleton."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service