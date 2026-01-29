"""
Custom exceptions for the application.
"""


class VoiceRAGException(Exception):
    """Base exception for Voice RAG application."""
    pass


class ASRException(VoiceRAGException):
    """Exception raised during ASR processing."""
    pass


class TranslationException(VoiceRAGException):
    """Exception raised during translation."""
    pass


class ScrapingException(VoiceRAGException):
    """Exception raised during web scraping."""
    pass


class VectorDBException(VoiceRAGException):
    """Exception raised during vector database operations."""
    pass


class LLMException(VoiceRAGException):
    """Exception raised during LLM processing."""
    pass


class ValidationException(VoiceRAGException):
    """Exception raised during input validation."""
    pass


class CacheException(VoiceRAGException):
    """Exception raised during cache operations."""
    pass