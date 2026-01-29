"""
Vector Database Service - Qdrant (Docker) Implementation
Features:
- Advanced RAG: Query Decomposition & Parallel Ingestion
- Qdrant Docker Integration
- Robust Chunking & Deduplication
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import asyncio
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.settings import settings
from src.core.chunker import TextChunker
from src.core.embedder import EmbeddingGenerator
from src.core.deduplication import get_deduplication_service
from src.utils.logger import setup_logger
from src.utils.exceptions import VectorDBException

# Import other services
from src.services.scraper_service import get_scraper_service
from src.services.llm_service import get_llm_service
# from qdrant_client.models import SparseVectorParams, SparseIndexParams
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from flashrank import Ranker, RerankRequest

logger = setup_logger(__name__)

class VectorDBService:
    """
    Qdrant Service with Advanced RAG Capabilities.
    """
    
    def __init__(self):
        self.chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        self.embedder = EmbeddingGenerator(model_name=settings.EMBEDDING_MODEL)
        self.dedup_service = get_deduplication_service()
        self.scraper_service = get_scraper_service()
        self.llm_service = get_llm_service(provider=settings.LLM_PROVIDER)

        # 1. Keyword Search (BM25)
        self.sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info(" Sparse Embedder (BM25) initialized")

        # 2. Semantic Re-ranking (Cross-Encoder)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        logger.info(" Hybrid RAG: Sparse Embedder & Reranker initialized")
        
        # Initialize Qdrant Connection
        try:
            logger.info(f"🔌 Connecting to Qdrant at {settings.QDRANT_URL}...")
            self.client = QdrantClient(url=settings.QDRANT_URL)
            # Test connection
            self.client.get_collections()
            logger.info("✅ Connected to Qdrant Docker Server")
        except Exception as e:
            raise VectorDBException(f"Qdrant Connection Failed. Is Docker running? Error: {e}")
        
    # --- Helpers ---

    async def _generate_embeddings_with_retry(self, texts, max_retries):
        for attempt in range(max_retries):
            try:
                return self.embedder.generate_embeddings(texts)
            except Exception as e:
                if attempt == max_retries - 1: raise e
                time.sleep(1)

    async def _store_qdrant(self, collection_name, chunks, embeddings, force_recreate):
        # 1. Recreate collection if names are missing
        exists = any(c.name == collection_name for c in self.client.get_collections().collections)
        
        if force_recreate and exists:
            self.client.delete_collection(collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.embedder.dimension, 
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )

        points = []
        for i, chunk in enumerate(chunks):
            # Generate Sparse Vector for this chunk
            sparse_vec = list(self.sparse_embedder.embed([chunk['text']]))[0]
            
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk['text'][:100] + str(i))),
                    vector={
                        "dense": embeddings[i], 
                        "sparse": sparse_vec.as_object() 
                    },
                    payload={
                        "text": chunk['text'],
                        "metadata": chunk['metadata']
                    }
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)

    def _load_document(self, filepath: str) -> tuple:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        if lines[0].startswith("Wikipedia Article:"):
            article_title = lines[0].replace("Wikipedia Article:", "").strip()
            content = '\n'.join(lines[3:])
        else:
            article_title = Path(filepath).stem

        metadata = {
            'source_file': Path(filepath).name,
            'article_title': article_title
        }
        return content, metadata
    
    async def retrieve_for_topics(
            self, 
            topics: List[str], 
            collection_name: str, 
            # We ignore the fixed 'k_per_topic' and calculate it dynamically
        ) -> Dict[str, Any]:
            """
            Dynamic Hybrid Retrieval: Guarantees 2 chunks per unique topic.
            """
            all_results = []
            seen_ids = set()
            
            # Determine total target chunks
            settings.TOP_K_RESULTS = 2
            total_target_count = len(topics) * settings.TOP_K_RESULTS

            for topic in topics:
                # Stage 1: Broad Retrieval (Get extra candidates for the Reranker)
                # We fetch 10 per topic to give the reranker a healthy pool to sort
                dense_query_vector = await self._generate_embeddings_with_retry([topic], 3)
                sparse_query_vector = list(self.sparse_embedder.embed([topic]))[0]

                search_result = self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(query=dense_query_vector[0], using="dense", limit=10),
                        models.Prefetch(query=sparse_query_vector.as_object(), using="sparse", limit=10),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=20 
                )
                
                for res in search_result.points:
                    if res.id not in seen_ids:
                        seen_ids.add(res.id)
                        all_results.append({
                            'id': res.id,
                            'text': res.payload['text'], 
                            'metadata': res.payload['metadata']
                        })

            # Stage 2: Semantic Reranking
            combined_query = " ".join(topics) 
            rerank_request = RerankRequest(query=combined_query, passages=all_results)
            reranked_results = self.reranker.rerank(rerank_request)

            # Stage 3: Dynamic Slicing
            # This ensures that if you have 3 topics, you get 6 chunks. If 2 topics, you get 4.
            final_results = []
            for res in reranked_results[:total_target_count]:
                final_results.append({
                    'id': res['id'],
                    'document': res['text'],
                    'metadata': res['metadata'],
                    'score': res['score']
                })

            return {
                "results": final_results,
                "total_results": len(final_results),
                "topics_count": len(topics)
            }
        

    async def process_query_to_vectordb(
        self,
        user_query: str,
        collection_name: str = settings.COLLECTION_NAME
    ) -> Dict[str, Any]:
        """
        🚀 MASTER FUNCTION with CHUNK-LEVEL Deduplication
        """
        logger.info(f"🧠 Processing Complex Query: '{user_query}'")
        start_time = time.time()
        
        # 1. Decompose Query
        sub_queries = await self.llm_service.decompose_query(user_query)
        logger.info(f"🔀 Decomposed into: {sub_queries['titles']}")
        
        # 2. Scrape ALL topics (no document-level filtering)
        # Chunk-level deduplication will handle duplicates
        logger.info(f"🌐 Scraping {len(sub_queries['titles'])} topics: {sub_queries['titles']}")
        tasks = [self.scraper_service.scrape_wikipedia(topic) for topic in sub_queries["titles"]]
        scrape_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scraped_docs = []
        for res in scrape_results:
            if isinstance(res, dict) and "file_path" in res:
                scraped_docs.append(res["file_path"])
            elif isinstance(res, Exception):
                logger.error(f"❌ Scrape task failed: {res}")
        
        # 3. Ingest ALL documents (chunk-level dedup inside create_vectordb)
        total_new_chunks = 0
        total_duplicates = 0
        
        for doc_path in scraped_docs:
            ingest_result = await self.create_vectordb(
                document_path=doc_path,
                collection_name=collection_name,
                force_recreate=False 
            )
            total_new_chunks += ingest_result.get("total_chunks", 0)
            total_duplicates += ingest_result.get("duplicates_removed", 0)
        
        total_time = time.time() - start_time
        logger.info(f"✅ RAG Complete: {total_new_chunks} new chunks, {total_duplicates} duplicates skipped in {total_time:.2f}s")
        
        return {
            "status": "success",
            "original_query": user_query,
            "sub_queries": sub_queries["titles"],
            "documents_processed": len(scraped_docs),
            "total_new_chunks": total_new_chunks,
            "total_duplicates_removed": total_duplicates,
            "collection_name": collection_name
        }


    async def create_vectordb(
        self,
        document_path: str,
        collection_name: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        force_recreate: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Ingest a single file into Qdrant with CHUNK-LEVEL deduplication.
        """
        try:
            # Load & Chunk (NO document-level cache check)
            text, metadata = self._load_document(document_path)
            
            original_size = self.chunker.chunk_size
            if chunk_size: self.chunker.chunk_size = chunk_size
            if chunk_overlap: self.chunker.chunk_overlap = chunk_overlap
            
            chunks = self.chunker.chunk_text(text, metadata=metadata)
            
            self.chunker.chunk_size = original_size
            
            if not chunks:
                return {"total_chunks": 0, "status": "empty"}

            # ✅ CHUNK-LEVEL DEDUPLICATION: Filter duplicate chunks
            logger.info(f"📊 Generated {len(chunks)} chunks, checking for duplicates...")
            unique_chunks = self.dedup_service.filter_duplicate_chunks(collection_name, chunks)
            logger.info(f"✅ {len(unique_chunks)} unique chunks (removed {len(chunks) - len(unique_chunks)} duplicates)")
            
            if not unique_chunks:
                logger.info(f"⏭️ All chunks from {Path(document_path).name} already exist in collection")
                return {"total_chunks": 0, "status": "all_duplicates"}

            # Embed only unique chunks
            texts = [chunk['text'] for chunk in unique_chunks]
            embeddings = await self._generate_embeddings_with_retry(texts, max_retries)
            
            # Store in Qdrant
            await self._store_qdrant(collection_name, unique_chunks, embeddings, force_recreate)
            
            # Save cache state
            self.dedup_service._save_cache()
            
            return {
                "collection_name": collection_name,
                "total_chunks": len(unique_chunks),
                "total_generated": len(chunks),
                "duplicates_removed": len(chunks) - len(unique_chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Ingestion failed for {document_path}: {e}")
            raise e


# Singleton
_service_instance = None
def get_vectordb_service(backend: str = "qdrant") -> VectorDBService:
    global _service_instance
    if _service_instance is None:
        _service_instance = VectorDBService()
    return _service_instance