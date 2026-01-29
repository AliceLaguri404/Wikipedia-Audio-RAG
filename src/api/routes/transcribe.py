"""
ASR Transcription Endpoint
Handles File Uploads and Mic Blobs (WebM/WAV)
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
import os
import time
import mimetypes
from pathlib import Path

# Try/Except to handle if schemas are not yet defined
try:
    from src.models.schemas import TranscriptionResponse, StatusEnum
except ImportError:
    # Fallback for compilation if schema is missing
    from pydantic import BaseModel
    class TranscriptionResponse(BaseModel):
        status: str
        transcribed_text: str
    class StatusEnum:
        SUCCESS = "success"

from src.services.asr_service import get_asr_service
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)
router = APIRouter()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language_code: str = Form("hi-IN"), 
    with_timestamps: bool = Form(True),
    with_diarization: bool = Form(False),
    num_speakers: int = Form(2)
):
    """
    Transcribe audio from File Upload or Microphone Blob.
    """
    temp_path = None
    try:
        logger.info(f"🎤 Transcription request. Filename: {audio_file.filename}, Content-Type: {audio_file.content_type}")
        
        # 1. Determine correct file extension
        # Many mic recordings (blobs) arrive without extension (e.g. filename="blob")
        filename = audio_file.filename
        file_ext = Path(filename).suffix.lower()
        
        if not file_ext:
            # Guess extension from MIME type (e.g., 'audio/webm' -> '.webm')
            file_ext = mimetypes.guess_extension(audio_file.content_type)
            if not file_ext:
                # Default fallback for web audio
                file_ext = ".webm" if "webm" in audio_file.content_type else ".wav"
        
        # 2. Create distinct temp filename
        safe_filename = f"audio_{int(time.time())}{file_ext}"
        temp_path = settings.TEMP_DIR / safe_filename
        
        # Ensure temp dir exists
        settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # 3. Save the file
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        logger.info(f"📥 Saved audio to: {temp_path}")
        
        # 4. Get Service & Transcribe
        asr_service = get_asr_service()
        
        # Validate (Now the file has a proper extension, validation should pass)
        if not asr_service.validate_audio_file(str(temp_path)):
            raise HTTPException(status_code=400, detail="Invalid audio format. Supported: .wav, .mp3, .webm, .ogg")

        result = await asr_service.transcribe_audio(
            audio_file_path=str(temp_path),
            language_code=language_code,
            with_timestamps=with_timestamps,
            with_diarization=with_diarization,
            num_speakers=num_speakers
        )
        
        return TranscriptionResponse(
            status=StatusEnum.SUCCESS,
            message="Audio transcribed successfully",
            transcribed_text=result["transcribed_text"],
            language_detected=result.get("language_detected"),
            duration=result.get("duration"),
            confidence=result.get("confidence")
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 5. Cleanup Temp File
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"🧹 Cleaned up temp file: {temp_path}")
            except Exception:
                pass