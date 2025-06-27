from app.services.base_service import BaseService
from app.core.config import get_settings
from langchain_ollama import OllamaEmbeddings
from typing import Optional, Any, List
from app.utils.logger import get_logger_instance, Logger
from app.utils.redis_cache import redis_cache_decorator
import asyncio
import numpy as np

class EmbeddingService(BaseService):
    """
    Service for generating embeddings using a locally running Ollama model (nomic-embed-text).
    Enhanced with batch processing, caching, and performance optimizations.
    """
    def __init__(self, model_name: str = "nomic-embed-text", logger: Optional[Logger] = None, websocket_manager=None):
        """
        Initialize the EmbeddingService.
        Args:
            model_name: Name of the Ollama model to use for embeddings.
            logger: Optional logger instance.
            websocket_manager: Optional WebSocketManager for notifications.
        """
        super().__init__(websocket_manager=websocket_manager)
        self.logger = logger or get_logger_instance("EmbeddingService")
        self.model_name = model_name
        self.embedder = OllamaEmbeddings(model=model_name)
        self._batch_size = 50  # Optimal batch size for API calls
        self._max_retries = 3
        self._retry_delay = 1.0

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self._max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise e
                wait_time = self._retry_delay * (2 ** attempt)
                await self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    @BaseService.cache_result(ttl_seconds=3600)  # Cache embeddings for 1 hour
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string with caching.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: The embedding vector.
        Raises:
            Exception: If embedding fails.
        """
        try:
            await self._rate_limit()
            import asyncio
            loop = asyncio.get_event_loop()
            # Use OllamaEmbeddings' embed_query method
            result = await loop.run_in_executor(
                None, lambda: self.embedder.embed_query(text)
            )
            return result
        except Exception as e:
            await self.logger.error(f"EmbeddingService.embed_text error: {e}")
            raise

    @redis_cache_decorator(expire=1800)  # Cache for 30 minutes
    async def embed_texts_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings with optimized batching.
        Args:
            texts (list): List of text strings to embed.
            batch_size (int): Override default batch size.
        Returns:
            List[List[float]]: List of embedding vectors.
        Raises:
            Exception: If embedding fails.
        """
        try:
            if not texts:
                return []
            
            batch_size = batch_size or self._batch_size
            results = []
            
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                await self._rate_limit()
                
                batch_result = await self._retry_with_backoff(
                    lambda b=batch: asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.embedder.embed_documents(b)
                    )
                )
                results.extend(batch_result)
                
                # Small delay between batches to avoid overwhelming the API
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return results
        except Exception as e:
            await self.logger.error(f"EmbeddingService.embed_texts_batch error: {e}")
            raise

    async def embed_texts(self, texts: list) -> List[List[float]]:
        """
        Legacy method for backward compatibility.
        """
        return await self.embed_texts_batch(texts)

    async def compute_similarity_matrix(self, embeddings1: List[List[float]], embeddings2: List[List[float]]) -> np.ndarray:
        """
        Compute cosine similarity matrix between two sets of embeddings efficiently.
        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
        Returns:
            np.ndarray: Similarity matrix.
        """
        try:
            emb1 = np.array(embeddings1)
            emb2 = np.array(embeddings2)
            
            # Normalize embeddings for cosine similarity
            emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
            emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
            
            # Compute similarity matrix efficiently
            similarity_matrix = np.dot(emb1_norm, emb2_norm.T)
            return similarity_matrix
        except Exception as e:
            await self.logger.error(f"compute_similarity_matrix error: {e}")
            raise

    async def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        Args:
            query_embedding: The query embedding.
            candidate_embeddings: List of candidate embeddings.
            top_k: Number of top results to return.
        Returns:
            List[tuple]: List of (index, similarity_score) tuples.
        """
        try:
            query_emb = np.array(query_embedding).reshape(1, -1)
            candidates = np.array(candidate_embeddings)
            
            # Compute similarities
            similarity_matrix = await self.compute_similarity_matrix(query_emb, candidates)
            similarities = similarity_matrix[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
        except Exception as e:
            await self.logger.error(f"find_most_similar error: {e}")
            raise

    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

    async def _perform_action_impl(self, *args, **kwargs):
        """
        Perform embedding action based on provided arguments.
        Args:
            text: Single string to embed.
            texts: List of strings to embed.
        Returns:
            Embedding(s) for the input.
        Raises:
            ValueError: If neither text nor texts is provided.
        """
        text = kwargs.get('text')
        if text:
            return await self.embed_text(text)
        texts = kwargs.get('texts')
        if texts:
            return await self.embed_texts(texts)
        raise ValueError("No text or texts provided for embedding.")
