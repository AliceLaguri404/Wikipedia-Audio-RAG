#!/usr/bin/env python3
"""
Easy startup script for Voice-Enabled RAG System.
Handles initialization and runs the FastAPI server.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Check if environment is properly configured."""
    from config.settings import settings
    
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check API keys
    if not settings.SARVAM_API_KEY:
        issues.append("❌ SARVAM_API_KEY not set")
    else:
        print("✅ Sarvam API key configured")
    
    if settings.LLM_PROVIDER == "gemini":
        if not settings.GEMINI_API_KEY:
            issues.append("❌ GEMINI_API_KEY not set (required for LLM_PROVIDER=gemini)")
        else:
            print("✅ Gemini API key configured")
    elif settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            issues.append("❌ OPENAI_API_KEY not set (required for LLM_PROVIDER=openai)")
        else:
            print("✅ OpenAI API key configured")
    
    # Check directories
    for dir_path in [settings.STORAGE_DIR, settings.DOCUMENTS_DIR, settings.TEMP_DIR]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Print configuration
    print(f"\n📊 Configuration:")
    print(f"  - LLM Provider: {settings.LLM_PROVIDER}")
    print(f"  - Vector DB Backend: {settings.VECTOR_DB_BACKEND}")
    print(f"  - ASR Model ID: {settings.ASR_MODEL_ID}")
    
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Please update your .env file with the required API keys")
        return False
    
    print("\n✅ All checks passed!")
    return True


def main():
    """Main startup function."""
    print("=" * 80)
    print("🚀 Voice-Enabled RAG System")
    print("=" * 80)
    print()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Import and run FastAPI
    print("\n🌐 Starting FastAPI server...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🏥 Health check available at: http://localhost:8000/api/v1/health")
    print("\n" + "=" * 80)
    print()
    
    import uvicorn
    from src.api.main import app
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()

