from app.services.base_service import BaseService
from typing import List, Dict, Any, Optional, Union
import os
import io
import asyncio

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

class ParserService(BaseService):
    """
    Service for parsing PDFs, large strings, and other documents into structured text chunks.
    Inspired by the RAG-Challenge-2 pipeline (Docling, chunking, etc.).
    """
    def __init__(self):
        super().__init__()
        if PdfReader is None:
            raise ImportError("pypdf is required for PDF parsing. Please install with 'pip install pypdf'.")

    async def parse_pdf(self, file_path: str, chunk_size: int = 1000) -> List[str]:
        """
        Parse a PDF file and split its text into chunks.
        Args:
            file_path (str): Path to the PDF file.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        return await self.chunk_text(full_text, chunk_size)

    async def parse_string(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split a large string into chunks.
        Args:
            text (str): The input text.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
        return await self.chunk_text(text, chunk_size)

    async def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks of a given size, respecting sentence boundaries if possible.
        Args:
            text (str): The input text.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
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

    async def perform_action(self, source: Union[str, bytes], source_type: str = "pdf", chunk_size: int = 1000) -> List[str]:
        """
        Parse a document (PDF, string, or other) and return text chunks.
        Args:
            source (str|bytes): Path to file or raw text.
            source_type (str): 'pdf', 'string', or other supported type.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
        if source_type == "pdf":
            if isinstance(source, bytes):
                # Parse PDF from bytes
                reader = PdfReader(io.BytesIO(source))
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
