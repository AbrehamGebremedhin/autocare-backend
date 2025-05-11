from app.services.base import BaseService
from app.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class QueryBuilderService(BaseService):
    """
    Service to build optimized queries for search engines, YouTube, and vector search using Gemini API via LangChain.
    """
    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.gemini_api_key = settings.GEMINI_KEY
        self.gemini_model = settings.GEMINI_MODEL_1
        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_model,
            google_api_key=self.gemini_api_key
        )
        self.prompts = {
            "search_engine": ChatPromptTemplate.from_template(
                """
                As an expert search query optimizer, analyze the user query: '{query}'
                
                STEPS:
                1. Identify the primary search intent (informational, navigational, transactional, or commercial)
                2. Extract key concepts and entities from the query
                3. Add relevant modifiers, qualifiers or synonyms to increase precision
                4. Apply appropriate search operators (quotes, site:, filetype:, etc.) where beneficial
                5. Remove unnecessary words that could limit results
                6. Structure the query to prioritize the most important terms first
                
                Generate an optimized search engine query that will maximize relevance while ensuring comprehensive coverage of the topic. The query should be engineered to work effectively with modern search algorithms.
                
                Return ONLY the optimized query string without explanation, quotes or formatting.
                """
            ),
            "youtube": ChatPromptTemplate.from_template(
                """
                As a YouTube search optimization expert, transform the user query: '{query}'
                
                STEPS:
                1. Determine the likely video content type being sought (tutorial, review, entertainment, educational, etc.)
                2. Identify key topic terms that would appear in relevant video titles and descriptions
                3. Include quality indicators (e.g., "best", "official", "complete", "professional") if appropriate
                4. Consider recency requirements (new, latest, 2025, etc.) if time-sensitive information is needed
                5. Incorporate format specifications (long-form, short, livestream) if implied by the query
                
                Generate a YouTube search query that will prioritize high-quality, relevant videos from authoritative sources. The query should be tailored to YouTube's search algorithm, which emphasizes title matches, view counts, and engagement metrics.
                
                Return ONLY the optimized YouTube query string without explanation, quotes or formatting.
                """
            ),
            "vector_search": ChatPromptTemplate.from_template(
                """
                As a semantic search expert, transform the user query: '{query}'
                
                STEPS:
                1. Identify the core conceptual entities in the query
                2. Extract the underlying semantic intention beyond literal keywords
                3. Reformulate using information-dense terminology that captures semantic meaning
                4. Preserve critical context while removing conversational elements
                5. Ensure query maintains semantic richness for embedding-based retrieval
                
                Generate a semantically optimized query for vector database search that captures conceptual meaning rather than just keywords. The query should be engineered for maximum semantic similarity matching in embedding space.
                
                Return ONLY the optimized semantic query string without explanation, quotes or formatting. Keep it concise yet information-rich.
                """
            )
        }

    def perform_action(self, user_query: str, query_type: str = None):
        """
        Build optimized queries from a user query, optionally for a specific search type.
        
        Args:
            user_query: The original user query
            query_type: Optional type of query to generate ('search_engine', 'youtube', 'vector_search', or None for all)
        """
        results = {}
        
        if query_type and query_type in self.prompts:
            chain = self.prompts[query_type] | self.llm
            result = chain.invoke({"query": user_query})
            results[query_type] = result.content if hasattr(result, 'content') else str(result)
        else:
            for key, prompt in self.prompts.items():
                chain = prompt | self.llm
                result = chain.invoke({"query": user_query})
                results[key] = result.content if hasattr(result, 'content') else str(result)
                
        return results
