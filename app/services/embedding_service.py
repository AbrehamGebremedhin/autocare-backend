from app.services.base_service import BaseService
from app.core.config import get_settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional, Any
from app.utils.logger import Logger

class EmbeddingService(BaseService):
    """
    Service for generating embeddings using GoogleGenerativeAIEmbeddings.
    Dependencies can be injected for testability and flexibility.
    """
    def __init__(self, embedder: Optional[Any] = None, logger: Optional[Logger] = None, settings: Optional[Any] = None):
        """
        Initialize the EmbeddingService.
        Args:
            embedder: Optional custom embedding backend.
            logger: Optional logger instance.
            settings: Optional settings/config object.
        """
        super().__init__()
        self.logger = logger or Logger("EmbeddingService")
        self.settings = settings or get_settings()
        self.embedder = embedder or GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.settings.GEMINI_KEY,
        )

    async def embed_text(self, text: str):
        """
        Generate an embedding for a single text string.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: The embedding vector.
        Raises:
            Exception: If embedding fails.
        """
        try:
            return self.embedder.embed_query(text)
        except Exception as e:
            await self.logger.error(f"EmbeddingService.embed_text error: {e}")
            raise

    async def embed_texts(self, texts: list):
        """
        Generate embeddings for a list of text strings.
        Args:
            texts (list): List of text strings to embed.
        Returns:
            List[List[float]]: List of embedding vectors.
        Raises:
            Exception: If embedding fails.
        """
        try:
            return self.embedder.embed_documents(texts)
        except Exception as e:
            await self.logger.error(f"EmbeddingService.embed_texts error: {e}")
            raise

    async def perform_action(self, *args, **kwargs):
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
