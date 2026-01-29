"""
ASR Service - AI4Bharat IndicConformer (Lightweight / Remote Code)
"""
import torch
import torchaudio
import time
from typing import Dict, Any
from transformers import AutoModel

from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.exceptions import ASRException

logger = setup_logger(__name__)

class ASRService:
    """
    Handles audio transcription using AI4Bharat IndicConformer via HF AutoModel.
    """
    
    def __init__(self):
        """Initialize Model using Remote Code."""
        try:
            logger.info(f"🔄 Loading IndicConformer (Lightweight): {settings.ASR_MODEL_ID}...")
            
            # Load model with trust_remote_code=True
            # This downloads the 'model.py' from Hugging Face and executes it
            self.model = AutoModel.from_pretrained(
                settings.ASR_MODEL_ID, 
                trust_remote_code=True,
                token=settings.HF_API_TOKEN if settings.HF_API_TOKEN else None
            )
            
            logger.info("IndicConformer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Model: {e}")
            logger.error("Ensure you have accepted the license at: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual")
            logger.error("Ensure you have run: 'huggingface-cli login'")
            raise ASRException(f"Model loading failed: {e}")
    
    async def transcribe_audio(
        self,
        audio_file_path: str,
        language_code: str = "hi", 
        with_timestamps: bool = False,
        with_diarization: bool = False,
        num_speakers: int = 2
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.
        """
        try:
            logger.info(f"🎤 Starting transcription for: {audio_file_path}")
            start_time = time.time()
            
            import librosa
            
            # Load and resample
            wav_numpy, _ = librosa.load(audio_file_path, sr=16000, mono=True)
            wav = torch.from_numpy(wav_numpy).unsqueeze(0)
            
            # Prepare Language ID
            lang_id = language_code.split('-')[0] # 'hi-IN' -> 'hi'
            
            # CHECK: Block English explicitly to avoid confusing errors
            if lang_id == 'en':
                raise ASRException("IndicConformer does NOT support English ('en'). Please use an Indian language code (e.g., 'hi', 'ta', 'bn').")

            # Inference
            try:
                result = self.model(wav, lang_id, "ctc")
            except KeyError:
                raise ASRException(f"Language code '{lang_id}' is not supported by this model.")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Transcription completed in {processing_time:.2f}s")
            
            return {
                "transcribed_text": result,
                "language_detected": lang_id,
                "duration": processing_time,
                "confidence": 1.0, 
                "timestamps": None, 
                "speakers": None
            }
            
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise ASRException(f"Transcription error: {str(e)}")
    def validate_audio_file(self, file_path: str) -> bool:
        import os
        return os.path.exists(file_path)

# Singleton
_asr_service = None
def get_asr_service():
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service