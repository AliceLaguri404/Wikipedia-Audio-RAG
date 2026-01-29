"""
Wikipedia Scraper Service - Task 1
Robust Multi-Path Implementation (Regex + LLM Fallback)
Enhanced with comprehensive content extraction using html2text
"""
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import html2text

from config.settings import settings
from src.core.deduplication import get_deduplication_service
from src.utils.logger import setup_logger
from src.utils.exceptions import ScrapingException
from src.services.llm_service import get_llm_service

logger = setup_logger(__name__)


class WikipediaScraperService:
    """
    Wikipedia article scraper with production-grade query understanding.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.SCRAPER_USER_AGENT
        })
        self.dedup_service = get_deduplication_service()
        self.llm = get_llm_service(provider=settings.LLM_PROVIDER)
        
        # Initialize html2text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # Don't wrap text
        self.html_converter.single_line_break = False
        
        logger.info("✅ Scraper Service initialized")
    
    async def _extract_topic_with_llm(self, query: str) -> str:
        """
        Uses LLM to extract the SINGLE best Wikipedia topic.
        """
        try:
            prompt = f"""
            You are a Wikipedia Search Optimizer.
            Extract the SINGLE best Wikipedia Page Title from the user's query.
            
            USER QUERY: "{query}"
            
            RULES:
            1. Return ONLY the topic/title. No extra words.
            2. Remove stopwords (a, an, the) from the start.
            3. If the query mentions multiple topics (e.g. "AQI and IQ"), CHOOSE THE FIRST ONE.
            4. Fix spelling errors.
            
            OUTPUT:
            """
            
            topic = await self.llm.generate_answer(
                query=prompt,
                context="",
                max_retries=1
            )
            
            clean_topic = topic.strip().strip('"').strip("'").strip()
            clean_topic = re.sub(r'^(Topic|Title|Subject):\s*', '', clean_topic, flags=re.IGNORECASE)
            
            return clean_topic
            
        except Exception as e:
            logger.warning(f"⚠️ LLM extraction failed (Rate Limit or Error). Switching to Robust Regex.")
            return self._clean_query_regex(query)

    def _clean_query_regex(self, query: str) -> str:
        """
        Fast Path: Robust Regex-based cleaning.
        Handles: "I have doubt with...", "Help me understand...", "AQI and IQ"
        """
        text = query.strip().lower()
        
        patterns = [
            r"i (?:have|had|has) (?:a )?(?:doubt|question|query|issue|problem) (?:about|regarding|with|on|in)?",
            r"(?:can|could|would) you (?:please )?(?:help|tell|give|show) me (?:to )?(?:understand|know|find|search|about|info|information)?",
            r"(?:help|tell|give|show) me (?:to )?(?:understand|know|find|search|about|info|information)?",
            r"i (?:want|would like) to (?:know|learn|understand) (?:about)?",
            r"what (?:is|are|was|were) (?:the )?",
            r"who (?:is|are|was|were) (?:the )?",
            r"explain (?:to me )?(?:about )?",
            r"describe ",
            r"search for ",
            r"find "
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        text = text.replace(" this", "")
        text = text.strip("?.,! ")
        
        if " and " in text:
            parts = text.split(" and ")
            if len(parts[0]) > 2:
                text = parts[0]
        
        text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
        
        return text.strip()

    def _extract_comprehensive_html(self, page_url: str) -> str:
        """
        Extract comprehensive content from Wikipedia HTML using html2text.
        Preserves tables, lists, formatting, and structure.
        """
        try:
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return ""
            
            # Remove unwanted elements
            for element in content_div.find_all(['script', 'style', 'sup', 'span'], {'class': ['mw-editsection', 'reference']}):
                element.decompose()
            
            # Remove navigation boxes, sister project links, etc.
            for nav in content_div.find_all(['div', 'table'], {'class': lambda x: x and any(c in str(x).lower() for c in ['navbox', 'sistersitebox', 'ambox', 'mbox'])}):
                nav.decompose()
            
            # Convert to markdown-like text (preserves tables, lists, etc.)
            html_content = str(content_div)
            markdown_text = self.html_converter.handle(html_content)
            
            # Clean up the markdown
            markdown_text = self._clean_markdown(markdown_text)
            
            return markdown_text
            
        except Exception as e:
            logger.warning(f"HTML extraction failed: {e}")
            return ""

    def _clean_markdown(self, text: str) -> str:
        """
        Clean up the markdown text from html2text.
        """
        # Remove edit links
        text = re.sub(r'\[edit\]\([^\)]+\)', '', text)
        
        # Clean up citation markers but keep the structure
        text = re.sub(r'\[\d+\](?!\()', '', text)
        
        # Remove empty links
        text = re.sub(r'\[\]\([^\)]*\)', '', text)
        
        # Fix multiple newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines with only whitespace
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()

    async def scrape_wikipedia(
            self,
            query: str,
            force_refresh: bool = False,
            max_retries: int = 3
        ) -> Dict[str, Any]:
            import wikipedia
            
            # 1. Clean the query
            clean_query = self._clean_query_regex(query)
            logger.info(f"🚀 Scraping Wikipedia for: '{clean_query}'")
            
            # 2. Check Cache
            if not force_refresh and self.dedup_service.is_document_cached(clean_query):
                return self._return_cached_doc(query, clean_query)

            # 3. Robust Search Logic
            try:
                search_results = wikipedia.search(clean_query)
                
                if not search_results:
                    logger.info("⚠️ Wikipedia search failed. Asking LLM for a better topic...")
                    llm_topic = await self._extract_topic_with_llm(query)
                    search_results = wikipedia.search(llm_topic)
                
                if not search_results:
                    raise ScrapingException(f"No Wikipedia results found for '{query}'")

                best_match = search_results[0]
                logger.info(f"✅ Found best matching article: '{best_match}'")
                
                try:
                    page = wikipedia.page(best_match, auto_suggest=False)
                except wikipedia.DisambiguationError as e:
                    page = wikipedia.page(e.options[0], auto_suggest=False)

                article_title = page.title
                article_url = page.url
                
                # Get basic content from wikipedia library
                article_text_basic = page.content
                
                # Get comprehensive content with html2text (includes tables, lists, formatting)
                logger.info("📊 Extracting comprehensive content (tables, lists, formatting)...")
                article_text_comprehensive = self._extract_comprehensive_html(article_url)
                
                # Use comprehensive if available, fallback to basic
                article_text = article_text_comprehensive if article_text_comprehensive else article_text_basic

            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                raise e

            # 4. Save & Cache
            file_path = self._save_article(clean_query, article_title, article_text, article_url)
            
            self.dedup_service.cache_document(
                keyword=clean_query,
                article_title=article_title,
                file_path=str(file_path),
                article_url=article_url
            )
            
            logger.info(f"✅ Successfully scraped and cached: {article_title}")
            
            return {
                "query": query,
                "article_title": article_title,
                "article_url": article_url,
                "file_path": str(file_path),
                "text_length": len(article_text),
                "cached": False
            }

    def _return_cached_doc(self, original_query, cached_key):
        logger.info(f"💾 Cache HIT - returning cached document")
        doc_info = self.dedup_service.get_cached_document(cached_key)
        return {
            "query": original_query,
            "article_title": doc_info["article_title"],
            "article_url": doc_info["article_url"],
            "file_path": doc_info["file_path"],
            "text_length": self._get_file_size(doc_info["file_path"]),
            "cached": True
        }

    def _save_article(self, query: str, article_title: str, article_text: str, article_url: str = "N/A") -> Path:
        """
        Saves the article to a text file with the source URL included at the top.
        """
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_').lower()
        filename = f"{safe_query}_wikipedia.txt"
        settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = settings.DOCUMENTS_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Wikipedia Article: {article_title}\n")
            f.write(f"Source URL: {article_url}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(article_text)
            
        logger.info(f"💾 Saved to: {filepath}")
        return filepath

    def _get_file_size(self, filepath: str) -> int:
        try:
            with open(filepath, 'r', encoding='utf-8') as f: 
                return len(f.read())
        except: 
            return 0


# Singleton instance
_scraper_service = None

def get_scraper_service() -> WikipediaScraperService:
    global _scraper_service
    if _scraper_service is None:
        _scraper_service = WikipediaScraperService()
    return _scraper_service