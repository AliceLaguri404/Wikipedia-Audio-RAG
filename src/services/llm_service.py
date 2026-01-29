import json
from typing import List, Optional
from pydantic import BaseModel, Field
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMException

logger = setup_logger(__name__)

class MultiQueryExpansion(BaseModel):
    titles: List[str] = Field(description="List of Wikipedia article titles")
    improvised_queries: List[str] = Field(description="List of optimized RAG search queries")
    
class GeminiLLMService:
    @retry(
        wait=wait_random_exponential(min=1, max=60), 
        stop=stop_after_attempt(5)
    )
    def __init__(self):
        try:
            # We keep the base_llm accessible for structured output calls
            self.base_llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=settings.GEMINI_TEMPERATURE,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
            )

            # This is the wrapped version for standard text generation
            self.llm = self.base_llm.with_retry(
                retry_if_exception_type=(Exception,),
                stop_after_attempt=3,
                wait_exponential_jitter=True,
            )

            logger.info("✅ Gemini LLM initialized")

        except Exception as e:
            logger.exception("Gemini initialization failed")
            raise LLMException(f"Failed to initialize Gemini: {e}")

    async def generate_answer(self, query: str, context: Optional[str] = None) -> str:
        try:
            if context:
                prompt = PromptTemplate.from_template("""
                SYSTEM ROLE:
            You are an Expert Knowledge Synthesizer. Your goal is to provide a professional, 
            objective, and highly accurate response to the user's query using only the 
            verified information provided in the context blocks.

            CORE DIRECTIVES:
            - DIRECTNESS: Start answering the question immediately. Do not use introductory phrases like "Based on the documents..." or "The context states..."
            - FACTUAL INTEGRITY: Use ONLY the provided context. If the context does not contain the answer, reply: "I do not have sufficient information in the current documentation to answer this query."
            - CITATION STYLE: Append the corresponding source ID (e.g., [Source 1]) to every claim or sentence derived from that source.
            - NEUTRALITY: Maintain a Wikipedia-like encyclopedic tone. Avoid personal opinions, flowery adjectives, or "fluff."
            - SYNTHESIS: If multiple sources discuss the same topic, merge the information into a cohesive narrative rather than listing them separately.
            - LISTING: If asked list-type questions, use bullet points for clarity.
            - CLARITY: Use clear, concise language. Avoid jargon unless necessary, and explain technical terms when first introduced.
                                                                                               
            ADAPTIVE RESPONSE STRATEGIES:
            - FOR COMPARISONS: If the user asks to compare/contrast, use a Markdown table or clearly defined "Vs." sections.
            - FOR PROCESSES: If the user asks "How-to" or "Steps," use numbered lists with bolded phase headers.
            - FOR DEFINITIONS: Provide a concise lead-in followed by a "Technical Details" bulleted deep-dive.
            - FOR BROAD QUERIES: Breakdown the answer into logical sub-categories using ### headers.
            - FOR DATES & EVENTS: Present information in chronological order with bolded dates.
                                                      
            FORMATTING STANDARDS:
            - Use ### for section headers.
            - Use bullet points for lists of three or more items.
            - Use bolding for key terms or dates.

            CONTEXT:
            {context}

            USER QUESTION: 
            {query}

            EXPERT RESPONSE:
            """)
                chain = prompt | self.llm | StrOutputParser()
                return await chain.ainvoke({"context": context, "query": query})

            response = await self.llm.ainvoke(query)
            return response.content
        except Exception as e:
            logger.exception("Gemini generation failed")
            raise LLMException(f"Gemini generation failed: {e}")


    async def decompose_query(self, query: str) -> dict:
        try:
            # Use the multi-topic structured output model
            structured_llm = self.base_llm.with_structured_output(MultiQueryExpansion)
            
            prompt = PromptTemplate.from_template("""
                You are a Wikipedia Search Optimization Engine. Your task is to decompose complex user queries into discrete, independent topics for dual-stream retrieval.

                TASK:
                1. If the query contains multiple independent concepts, split them into a list of specific Wikipedia article titles.
                2. For each concept, generate a corresponding "Improvised Query" optimized for Vector RAG retrieval (technical, keyword-dense, and specific).
                3. If the query is singular, return it as both the title and improvised query.
                4. If the query is already a precise Wikipedia title, return it as is.
                5. If the query is X and Y in Z, treat X and Y as separate titles and both understand Z as context.

                CRITICAL RULES:
                - Wikipedia Titles: Use exact English Wikipedia formatting (e.g., "Latent Dirichlet allocation").
                - Improvised Queries: Focus on semantic depth (e.g., "LDA topic modeling algorithm hyperparameters Gibbs sampling").
                - Independence: If the user asks about two unrelated things (e.g., "React hooks and Docker volumes"), provide separate entries for each.
                - Format: Return ONLY valid JSON with keys 'titles' and 'improvised_queries'.

                EXAMPLES:
                - Query: "Comparison between Transformer architecture and CNNs"
                Result: {{
                    "titles": ["Transformer (deep learning architecture)", "Convolutional neural network"],
                    "improvised_queries": ["Transformer self-attention mechanism encoder decoder architecture", "CNN feature extraction convolutional layers pooling"]
                }}

                Query: {query}
            """)
            
            chain = prompt | structured_llm
            result = await chain.ainvoke({"query": query})
            
            return {
                "titles": result.titles,
                "improvised_queries": result.improvised_queries
            }
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return {"titles": [query], "improvised_queries": [query]}

_gemini_service: Optional[GeminiLLMService] = None

def get_llm_service(provider: str = "gemini") -> GeminiLLMService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiLLMService()
    return _gemini_service