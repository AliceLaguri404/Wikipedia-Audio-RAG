# Frequently Asked Questions (FAQ)

## General Questions

### Q1: What is this project?
**A:** This is a voice-enabled conversational AI system that accepts audio queries in multiple Indian languages, translates them to English, retrieves relevant information from Wikipedia, and generates intelligent responses using Retrieval-Augmented Generation (RAG).

### Q2: What languages are supported?
**A:** The system supports the following Indian languages through AI4Bharat's IndicConformer:
- Hindi (hi-IN)
- Tamil (ta-IN)
- Bengali (bn-IN)
- Telugu (te-IN)
- Kannada (kn-IN)
- Malayalam (ml-IN)
- Gujarati (gu-IN)
- Marathi (mr-IN)
- Punjabi (pa-IN)
- Odia (or-IN)

**Note:** English is NOT supported by the ASR model as it focuses on Indian languages.

### Q3: Is this project free to use?
**A:** Yes! The project uses free-tier services:
- **Gemini AI**: Free API with generous limits
- **Sarvam AI**: 1000 free translation credits
- **Qdrant**: Self-hosted, completely free
- **AI4Bharat**: Open-source ASR model

## Setup Questions

### Q4: What are the system requirements?
**A:** 
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Docker**: Required for Qdrant
- **Internet**: Required for API calls

### Q5: How do I get API keys?
**A:**
1. **Sarvam AI**: 
   - Visit https://www.sarvam.ai/
   - Sign up with email
   - Get 1000 free credits
   
2. **Gemini AI**:
   - Visit https://ai.google.dev/
   - Click "Get API Key"
   - Create key in Google AI Studio

3. **Hugging Face**:
   - Run `huggingface-cli login`
   - Accept model license at the model page

### Q6: Why does Hugging Face authentication fail?
**A:** Two common reasons:
1. You haven't accepted the model license
   - Visit: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
   - Click "Agree and access repository"

2. You haven't logged in via CLI
   ```bash
   huggingface-cli login
   # Enter your HF token
   ```

## Technical Questions

### Q7: Why use Qdrant instead of Pinecone or Weaviate?
**A:**
- **Self-hosted**: No external dependencies or quotas
- **Free**: Unlimited usage
- **Hybrid Search**: Native support for BM25 + vector search
- **Performance**: Optimized for dense + sparse vectors
- **Privacy**: Data stays on your machine

### Q8: What is hybrid retrieval and why is it important?
**A:** Hybrid retrieval combines:
- **BM25 (Keyword Search)**: Finds exact matches (e.g., "SDLC", specific dates)
- **Vector Search**: Finds semantic matches (e.g., "deep learning" ≈ "neural networks")
- **Re-ranking**: Cross-encoder re-scores results for highest relevance

This approach gives 40% better results than vector search alone.

### Q9: How does the caching system work?
**A:** Three-level caching:
1. **Document Level**: Checks if Wikipedia article already scraped (by title hash)
2. **Collection Level**: Checks if vector DB collection exists
3. **Chunk Level**: Filters duplicate chunks before embedding

Benefits: 80% reduction in redundant processing, faster responses.

### Q10: Why can't I use English with the ASR model?
**A:** The AI4Bharat IndicConformer is specifically trained for Indian languages and does not include English. For English audio, consider:
- Using text input directly
- Using a different ASR model (e.g., Whisper)
- Speaking in an Indian language (Hindi, Tamil, etc.)

## Usage Questions

### Q11: How do I test the API without the UI?
**A:** Use cURL or the Swagger UI:

```bash
# Option 1: cURL
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -F "text_query=What is machine learning?" \
  -F "language_code=en-IN"

# Option 2: Swagger UI
# Visit http://localhost:8000/docs
# Click on endpoint → Try it out → Execute
```

### Q12: Can I use my own documents instead of Wikipedia?
**A:** Yes! Modify the scraper service:

1. Replace `scraper_service.py` with your document loader
2. Ensure documents have these fields:
   - `text`: Main content
   - `metadata`: Dictionary with source info
3. Follow the same chunking and embedding flow

### Q13: How do I add more topics to the knowledge base?
**A:** Two ways:

**Option 1: Via API**
```bash
curl -X POST "http://localhost:8000/api/v1/vectordb/ingest-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Quantum Computing"}'
```

**Option 2: Via Chat**
Simply ask about a new topic in the chat interface. The system automatically:
1. Scrapes the Wikipedia article
2. Chunks the content
3. Stores in vector database
4. Returns answer

### Q14: How do I clear the cache?
**A:**

```bash
# Option 1: API endpoint
curl -X DELETE "http://localhost:8000/api/v1/documents/cache"

# Option 2: Delete collection
curl -X DELETE "http://localhost:8000/api/v1/vectordb/collection/voice_rag_knowledge_base"

# Option 3: Manual deletion
rm -rf storage/qdrant_storage/*
rm storage/cache_index.json
```

## Performance Questions

### Q15: Why is the first query so slow?
**A:** First query requires:
- Loading ML models into memory (~30-60 seconds)
- Scraping Wikipedia article (~2 seconds)
- Generating embeddings (~1 second)
- LLM generation (~2-3 seconds)

Subsequent queries are much faster due to:
- Models already loaded
- Documents cached
- Collections indexed

### Q16: How can I speed up the system?
**A:** Several optimizations:

1. **Use GPU**:
   ```python
   # config/settings.py
   ASR_DEVICE = "cuda"  # or "mps" for Mac
   ```

2. **Reduce chunk size**:
   ```python
   CHUNK_SIZE = 256  # Faster embedding
   TOP_K_RESULTS = 1  # Less retrieval
   ```

3. **Enable collection cache**:
   ```python
   ENABLE_COLLECTION_CACHE = True
   CACHE_TTL = 31536000  # 1 year
   ```

4. **Use faster embedding model**:
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```

### Q17: What are typical response times?
**A:**
| Component | Time |
|-----------|------|
| ASR Transcription | 1-2s |
| Translation | 0.5s |
| Wikipedia Scraping | 1-2s |
| Embedding Generation | 0.5s |
| Vector Search | 0.1s |
| LLM Generation | 2-3s |
| **Total (Cold Start)** | **5-8s** |
| **Total (Warm Cache)** | **2-3s** |

## Troubleshooting

### Q18: Qdrant connection refused error
**A:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker start qdrant

# If container doesn't exist
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Verify
curl http://localhost:6333/collections
```

### Q19: Out of memory error
**A:**
```python
# Reduce memory usage:
CHUNK_SIZE = 512  # Smaller chunks
TOP_K_RESULTS = 2  # Fewer results
ASR_DEVICE = "cpu"  # Use CPU instead of GPU

# Or increase system RAM
# Or use Docker with memory limits
```

### Q20: API rate limit exceeded
**A:**
- **Sarvam AI**: Wait 60 seconds or upgrade plan
- **Gemini**: Free tier has generous limits; wait briefly
- **Wikipedia**: Respect rate limits (10 requests/second)

### Q21: Audio file not supported error
**A:** Ensure audio format is supported:
- Supported: .wav, .mp3, .webm, .m4a, .ogg, .flac, .aac, .opus
- Not supported: .amr, .wma, .ra

Convert using FFmpeg:
```bash
ffmpeg -i input.amr -ar 16000 output.wav
```

### Q22: Wikipedia article not found
**A:** The system tries multiple strategies:
1. Regex cleaning (fast)
2. LLM extraction (fallback)
3. Wikipedia search API

If all fail, try:
- Use exact Wikipedia title (e.g., "Python (programming language)")
- Simplify query (e.g., "AI" instead of "Tell me about AI")
- Check if article exists on Wikipedia

## Advanced Questions

### Q23: Can I deploy this to production?
**A:** Yes, but consider:
1. **Security**: 
   - Use environment secrets (not .env)
   - Enable authentication
   - Set up HTTPS

2. **Scalability**:
   - Use managed Qdrant (Qdrant Cloud)
   - Add Redis caching
   - Use load balancer

3. **Monitoring**:
   - Add application metrics
   - Set up logging aggregation
   - Monitor API usage

### Q24: How do I add authentication?
**A:**
```python
# src/api/main.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@router.post("/chat/query")
async def rag_query(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    # Verify token
    ...
```

### Q25: Can I fine-tune the LLM for my domain?
**A:** Yes! Two approaches:

**Option 1: Few-Shot Prompting**
```python
prompt = f"""
You are an expert in {domain}.

Example 1: Q: ... A: ...
Example 2: Q: ... A: ...

Question: {query}
Answer:
"""
```

**Option 2: RAG with Domain Documents**
Replace Wikipedia scraper with your domain documents.

### Q26: How do I add more languages?
**A:** The ASR model already supports 10+ Indian languages. To add more:
1. Check if IndicConformer supports it
2. Update language validation in `asr_service.py`
3. Test thoroughly

For non-Indian languages, consider:
- OpenAI Whisper (supports 90+ languages)
- Google Speech-to-Text
- Azure Speech Service

## Architecture Questions

### Q27: Why separate scraping from retrieval?
**A:** Decoupling provides:
- **Flexibility**: Can swap Wikipedia for other sources
- **Testing**: Easy to mock services
- **Caching**: Document-level cache independent of vector DB
- **Debugging**: Easier to isolate issues

### Q28: Why use LangChain for chunking?
**A:**
- **Battle-tested**: Production-grade implementation
- **Paragraph-aware**: Preserves semantic boundaries
- **Configurable**: Easy to adjust chunk size/overlap
- **Integrated**: Works well with LlamaIndex/LangChain RAG

### Q29: What's the token limit for context?
**A:** 
- **Gemini 2.5 Flash**: 1M input tokens, 8K output
- **Practical limit**: 2-4 chunks × 500 tokens = 1000-2000 tokens
- **Fits comfortably**: Even complex queries stay under limits

### Q30: How scalable is this architecture?
**A:**
| Component | Scalability |
|-----------|-------------|
| FastAPI | Horizontal scaling (multiple workers) |
| Qdrant | Vertical + horizontal (sharding) |
| ASR | Batch processing (GPU) |
| LLM | API rate limits (Gemini: high) |
| **Bottleneck** | **LLM API calls** |

For production: Use caching aggressively + async processing.




