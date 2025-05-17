from app.services.base_service import BaseService
from typing import List, Dict, Any, Optional, Union
import os
import io
import asyncio
from app.utils.logger import Logger

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

class ParserService(BaseService):
    """
    Service for parsing PDFs, large strings, and other documents into structured text chunks.
    Dependencies can be injected for testability and flexibility.
    """
    def __init__(self, pdf_reader: Optional[Any] = None, logger: Optional[Logger] = None):
        """
        Initialize the ParserService.
        Args:
            pdf_reader: Optional PDF reader backend.
            logger: Optional logger instance.
        Raises:
            ImportError: If no PDF reader is available.
        """
        super().__init__()
        self.logger = logger or Logger("ParserService")
        self.pdf_reader = pdf_reader or PdfReader
        if self.pdf_reader is None:
            raise ImportError("pypdf is required for PDF parsing. Please install with 'pip install pypdf'.")

    async def parse_pdf(self, file_path: str, chunk_size: int = 1000) -> List[str]:
        """
        Parse a PDF file and split its text into chunks.
        Args:
            file_path (str): Path to the PDF file.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        Raises:
            Exception: If parsing fails.
        """
        try:
            reader = self.pdf_reader(file_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            return await self.chunk_text(full_text, chunk_size)
        except Exception as e:
            await self.logger.error(f"ParserService.parse_pdf error: {e}")
            raise

    async def parse_string(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split a large string into chunks.
        Args:
            text (str): The input text.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        Raises:
            Exception: If chunking fails.
        """
        try:
            return await self.chunk_text(text, chunk_size)
        except Exception as e:
            await self.logger.error(f"ParserService.parse_string error: {e}")
            raise

    async def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks of a given size, respecting sentence boundaries if possible.
        Args:
            text (str): The input text.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        Raises:
            Exception: If chunking fails.
        """
        try:
            import re
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        except Exception as e:
            await self.logger.error(f"ParserService.chunk_text error: {e}")
            raise

    async def perform_action(self, source: Union[str, bytes], source_type: str = "pdf", chunk_size: int = 1000) -> List[str]:
        """
        Parse a document (PDF, string, or other) and return text chunks.
        Args:
            source (str|bytes): Path to file or raw text.
            source_type (str): 'pdf', 'string', or other supported type.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        Raises:
            Exception: If parsing fails or source_type is unsupported.
        """
        try:
            if source_type == "pdf":
                if isinstance(source, bytes):
                    reader = self.pdf_reader(io.BytesIO(source))
                    full_text = ""
                    for page in reader.pages:
                        full_text += page.extract_text() or ""
                    return await self.chunk_text(full_text, chunk_size)
                else:
                    return await self.parse_pdf(source, chunk_size)
            elif source_type == "string":
                return await self.parse_string(source, chunk_size)
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
        except Exception as e:
            await self.logger.error(f"ParserService.perform_action error: {e}")
            raise
