"""
Embedding Generation Module
Handles text-to-vector conversion using sentence-transformers.
"""
from typing import List
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.
    
    Model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Speed: ~3000 sentences/sec (CPU)
    - Quality: Good for general retrieval
    - Cost: Free (local inference)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"📥 Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise Exception(f"Embedding model initialization failed: {e}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"🧮 Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Convert numpy to list
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise Exception(f"Failed to generate embeddings: {e}")