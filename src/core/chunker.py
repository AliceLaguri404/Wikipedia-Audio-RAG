"""
LangChain-based Text Chunking Module
Uses RecursiveCharacterTextSplitter with paragraph & sentence awareness.
"""

from typing import List, Dict, Any
from datetime import datetime
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextChunker:
    """
    LangChain-native text chunker.
    
    Strategy:
    - Paragraph-first splitting
    - Sentence-aware fallback
    - Character-based chunking (model-agnostic)
    - Deterministic overlap
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 150
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",   # paragraphs
                "\n",     # line breaks
                ". ",     # sentence end
                "? ",
                "! ",
                " ",      # fallback
                ""
            ]
        )

        logger.info("✅ LangChain RecursiveCharacterTextSplitter initialized")

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:

        if not text or not text.strip():
            return []

        logger.info(f"✂️ Chunking text ({len(text)} characters)")

        splits = self.splitter.split_text(text)

        logger.info(f"🧩 Created {len(splits)} chunks")

        results = []
        for idx, chunk in enumerate(splits):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": idx,
                "total_chunks": len(splits),
                "chunk_size_chars": len(chunk),
                "created_at": datetime.utcnow().isoformat()
            })

            results.append({
                "text": chunk,
                "metadata": chunk_metadata,
                "chunk_id": self._generate_chunk_id(chunk, idx)
            })

        return results

    @staticmethod
    def _generate_chunk_id(text: str, index: int) -> str:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return f"chunk_{index}_{digest}"
