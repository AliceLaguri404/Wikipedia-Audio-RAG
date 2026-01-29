"""
Deduplication service for documents and chunks.

Implements multi-level caching:
1. Document-level: Check if Wikipedia article already scraped
2. Collection-level: Check if vector DB collection exists
3. Chunk-level: Check for duplicate chunks within collection
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, Set, Dict, Any
from datetime import datetime, timedelta

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DeduplicationService:
    """
    Manages deduplication across documents, collections, and chunks.
    
    Three-level deduplication strategy:
    1. Document Cache: Avoid re-scraping same Wikipedia articles
    2. Collection Cache: Reuse existing vector DB collections
    3. Chunk Cache: Skip duplicate chunks during embedding
    """
    
    def __init__(self):
        self.cache_file = settings.STORAGE_DIR / "cache_index.json"
        self.cache: Dict[str, Any] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"📦 Loaded cache with {len(cache.get('documents', {}))} documents")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return {
            "documents": {},      # keyword -> document info
            "collections": {},    # collection_name -> metadata
            "chunks": {}          # collection_name -> set of chunk hashes
        }
    
    def _save_cache(self):
        """Save cache index to disk."""
        try:
            # Convert sets to lists for JSON serialization
            serializable_cache = self.cache.copy()
            for coll_name, chunk_hashes in serializable_cache.get("chunks", {}).items():
                if isinstance(chunk_hashes, set):
                    serializable_cache["chunks"][coll_name] = list(chunk_hashes)
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    # ========== Document-Level Deduplication ==========
    
    def get_keyword_hash(self, keyword: str) -> str:
        """Generate hash for keyword."""
        return hashlib.md5(keyword.lower().strip().encode()).hexdigest()
    
    def is_document_cached(self, article_title: str) -> bool:
        """
        Check if the specific Wikipedia article has already been scraped.
        
        Args:
            article_title: The EXACT Wikipedia title (e.g., "Python (programming language)")
            
        Returns:
            True if article exists and is fresh, False otherwise
        """
        if not settings.ENABLE_DOCUMENT_CACHE:
            return False
        
        # We now check against the Title Hash, not the keyword hash
        title_hash = self.get_keyword_hash(article_title)
        
        if title_hash in self.cache["documents"]:
            doc_info = self.cache["documents"][title_hash]
            
            # Check if cache is still valid (TTL)
            cached_time = datetime.fromisoformat(doc_info["cached_at"])
            if datetime.now() - cached_time < timedelta(seconds=settings.CACHE_TTL):
                logger.info(f"✅ Wiki Article '{article_title}' found in cache. Skipping scrape.")
                return True
            else:
                logger.info(f"⏰ Wiki Article '{article_title}' cache EXPIRED.")
                del self.cache["documents"][title_hash]
                self._save_cache()
        
        logger.info(f"❌ Wiki Article '{article_title}' NOT in cache. Proceeding to scrape.")
        return False
    
    def get_cached_document(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        Get cached document information.
        
        Args:
            keyword: Search keyword/topic
            
        Returns:
            Document info dict or None
        """
        if not self.is_document_cached(keyword):
            return None
        
        key_hash = self.get_keyword_hash(keyword)
        return self.cache["documents"][key_hash]
    
    def cache_document(self, keyword: str, article_title: str, 
                      file_path: str, article_url: str):
        """
        Cache document information.
        
        Args:
            keyword: Search keyword/topic
            article_title: Wikipedia article title
            file_path: Path to saved document
            article_url: Wikipedia article URL
        """
        key_hash = self.get_keyword_hash(keyword)
        
        self.cache["documents"][key_hash] = {
            "keyword": keyword,
            "article_title": article_title,
            "file_path": file_path,
            "article_url": article_url,
            "cached_at": datetime.now().isoformat()
        }
        
        self._save_cache()
        logger.info(f"💾 Cached document for '{keyword}'")
    
    # ========== Collection-Level Deduplication ==========
    
    def is_collection_cached(self, collection_name: str) -> bool:
        """
        Check if vector DB collection exists.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            True if collection exists and is fresh
        """
        if not settings.ENABLE_COLLECTION_CACHE:
            return False
        
        if collection_name in self.cache["collections"]:
            coll_info = self.cache["collections"][collection_name]
            
            # Check TTL
            cached_time = datetime.fromisoformat(coll_info["created_at"])
            if datetime.now() - cached_time < timedelta(seconds=settings.CACHE_TTL):
                logger.info(f"✅ Collection cache HIT for '{collection_name}'")
                return True
            else:
                logger.info(f"⏰ Collection cache EXPIRED for '{collection_name}'")
                del self.cache["collections"][collection_name]
                if collection_name in self.cache["chunks"]:
                    del self.cache["chunks"][collection_name]
                self._save_cache()
        
        logger.info(f"❌ Collection cache MISS for '{collection_name}'")
        return False
    
    def cache_collection(self, collection_name: str, num_chunks: int, 
                        embedding_dim: int):
        """
        Cache collection information.
        
        Args:
            collection_name: Name of collection
            num_chunks: Number of chunks in collection
            embedding_dim: Embedding dimension
        """
        self.cache["collections"][collection_name] = {
            "name": collection_name,
            "num_chunks": num_chunks,
            "embedding_dim": embedding_dim,
            "created_at": datetime.now().isoformat()
        }
        
        # Initialize chunk hash set for this collection
        if collection_name not in self.cache["chunks"]:
            self.cache["chunks"][collection_name] = set()
        
        self._save_cache()
        logger.info(f"💾 Cached collection '{collection_name}' with {num_chunks} chunks")
    
    # ========== Chunk-Level Deduplication ==========
    
    def get_chunk_hash(self, chunk_text: str) -> str:
        """Generate hash for chunk text."""
        return hashlib.md5(chunk_text.encode()).hexdigest()
    
    def is_chunk_duplicate(self, collection_name: str, chunk_text: str) -> bool:
        """
        Check if chunk already exists in collection.
        
        Args:
            collection_name: Name of collection
            chunk_text: Text content of chunk
            
        Returns:
            True if chunk is duplicate
        """
        if collection_name not in self.cache["chunks"]:
            return False
        
        chunk_hash = self.get_chunk_hash(chunk_text)
        
        # Convert list to set if needed (from JSON loading)
        if isinstance(self.cache["chunks"][collection_name], list):
            self.cache["chunks"][collection_name] = set(self.cache["chunks"][collection_name])
        
        return chunk_hash in self.cache["chunks"][collection_name]
    
    def add_chunk_hash(self, collection_name: str, chunk_text: str):
        """
        Add chunk hash to collection's chunk set.
        
        Args:
            collection_name: Name of collection
            chunk_text: Text content of chunk
        """
        if collection_name not in self.cache["chunks"]:
            self.cache["chunks"][collection_name] = set()
        
        # Convert to set if needed
        if isinstance(self.cache["chunks"][collection_name], list):
            self.cache["chunks"][collection_name] = set(self.cache["chunks"][collection_name])
        
        chunk_hash = self.get_chunk_hash(chunk_text)
        self.cache["chunks"][collection_name].add(chunk_hash)
    
    def filter_duplicate_chunks(self, collection_name: str, 
                               chunks: list) -> list:
        """
        Filter out duplicate chunks.
        
        Args:
            collection_name: Name of collection
            chunks: List of chunk dictionaries
            
        Returns:
            Filtered list without duplicates
        """
        unique_chunks = []
        
        for chunk in chunks:
            if not self.is_chunk_duplicate(collection_name, chunk['text']):
                unique_chunks.append(chunk)
                self.add_chunk_hash(collection_name, chunk['text'])
        
        duplicates_removed = len(chunks) - len(unique_chunks)
        if duplicates_removed > 0:
            logger.info(f"🧹 Removed {duplicates_removed} duplicate chunks")
        
        return unique_chunks
    
    # ========== Utility Methods ==========
    
    def clear_cache(self):
        """Clear all cache data."""
        self.cache = {
            "documents": {},
            "collections": {},
            "chunks": {}
        }
        self._save_cache()
        logger.info("🗑️ Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_documents": len(self.cache["documents"]),
            "total_collections": len(self.cache["collections"]),
            "total_chunk_hashes": sum(
                len(hashes) for hashes in self.cache["chunks"].values()
            )
        }


# Singleton instance
_dedup_service = None


def get_deduplication_service() -> DeduplicationService:
    """Get or create deduplication service singleton."""
    global _dedup_service
    if _dedup_service is None:
        _dedup_service = DeduplicationService()
    return _dedup_service