from app.services.base_service import BaseService
from app.core.config import get_settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingService(BaseService):
    """
    Service for generating embeddings using GoogleGenerativeAIEmbeddings.
    """
    def __init__(self):
        super().__init__()
        settings = get_settings()
        google_api_key = settings.GEMINI_KEY
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key,
        )

    async def embed_text(self, text: str):
        """
        Generate an embedding for a single text string.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: The embedding vector.
        """
        return await self.embedder.embed_query(text)

    async def embed_texts(self, texts: list):
        """
        Generate embeddings for a list of text strings.
        Args:
            texts (list): List of text strings to embed.
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return await self.embedder.embed_documents(texts)

    async def perform_action(self, *args, **kwargs):
        """
        Example perform_action implementation for compatibility.
        """
        text = kwargs.get('text')
        if text:
            return await self.embed_text(text)
        texts = kwargs.get('texts')
        if texts:
            return await self.embed_texts(texts)
        raise ValueError("No text or texts provided for embedding.")
