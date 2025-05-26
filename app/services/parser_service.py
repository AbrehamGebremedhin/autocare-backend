from app.services.base_service import BaseService
from typing import List, Dict, Any, Optional, Union
import os
import io
import re
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

    async def chunk_text_optimized(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Enhanced text chunking with intelligent boundary detection for better semantic coherence.
        Args:
            text (str): The input text.
            chunk_size (int): Target number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
        try:
            import re
            
            if not text or not text.strip():
                return []
            
            # Clean and normalize text
            text = ' '.join(text.split())  # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
            
            # Enhanced sentence splitting with better patterns
            sentence_patterns = [
                r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence boundaries
                r'(?<=\.)\s+(?=\d+\.)',      # Numbered lists
                r'(?<=:)\s+(?=[A-Z])',       # After colons before sentences
                r'(?<=;)\s+(?=[A-Z])',       # After semicolons before sentences
            ]
            
            sentences = [text]
            for pattern in sentence_patterns:
                new_sentences = []
                for sentence in sentences:
                    new_sentences.extend(re.split(pattern, sentence))
                sentences = new_sentences
            
            # Filter out empty sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    # If current chunk has content, save it
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # If single sentence is larger than chunk_size, split it
                    if len(sentence) > chunk_size:
                        # Split long sentence by commas, then by words if necessary
                        sub_parts = sentence.split(', ')
                        for part in sub_parts:
                            if len(part) <= chunk_size:
                                if not current_chunk:
                                    current_chunk = part + " "
                                elif len(current_chunk) + len(part) + 1 <= chunk_size:
                                    current_chunk += part + " "
                                else:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = part + " "
                            else:
                                # Split by words as last resort
                                words = part.split()
                                temp_chunk = ""
                                for word in words:
                                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                                        temp_chunk += word + " "
                                    else:
                                        if temp_chunk.strip():
                                            chunks.append(temp_chunk.strip())
                                        temp_chunk = word + " "
                                if temp_chunk.strip():
                                    current_chunk = temp_chunk
                    else:
                        current_chunk = sentence + " "
            
            # Add remaining content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Filter out very short chunks (less than 50 characters) unless they're the only content
            if len(chunks) > 1:
                chunks = [chunk for chunk in chunks if len(chunk) >= 50]
            
            return chunks
        except Exception as e:
            await self.logger.error(f"ParserService.chunk_text_optimized error: {e}")
            return await self.chunk_text(text, chunk_size)  # Fallback to original method

    @BaseService.cache_result(ttl_seconds=1800)  # Cache parsed PDFs for 30 minutes
    async def parse_pdf_optimized(self, file_path: str, chunk_size: int = 1000) -> List[str]:
        """
        Parse a PDF file with enhanced processing and caching.
        Args:
            file_path (str): Path to the PDF file.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
        try:
            loop = asyncio.get_running_loop()
            
            def _parse_pdf_sync():
                reader = self.pdf_reader(file_path)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        # Clean up common PDF artifacts
                        page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                        page_text = re.sub(r'(\w)-\s+(\w)', r'\1\2', page_text)  # Fix hyphenated words
                        full_text += page_text + " "
                    except Exception as e:
                        asyncio.run(self.logger.warning(f"Error extracting page {page_num}: {e}"))
                        continue
                
                return full_text.strip()
            
            full_text = await loop.run_in_executor(None, _parse_pdf_sync)
            return await self.chunk_text_optimized(full_text, chunk_size)
        except Exception as e:
            await self.logger.error(f"ParserService.parse_pdf_optimized error: {e}")
            return await self.parse_pdf(file_path, chunk_size)  # Fallback

    async def parse_pdf_bytes_optimized(self, pdf_bytes: bytes, chunk_size: int = 1000) -> List[str]:
        """
        Parse PDF from bytes with optimized processing.
        Args:
            pdf_bytes (bytes): PDF file content as bytes.
            chunk_size (int): Number of characters per chunk.
        Returns:
            List[str]: List of text chunks.
        """
        try:
            import re
            loop = asyncio.get_running_loop()
            
            def _parse_pdf_bytes_sync():
                reader = self.pdf_reader(io.BytesIO(pdf_bytes))
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        # Clean up PDF artifacts
                        page_text = re.sub(r'\s+', ' ', page_text)
                        page_text = re.sub(r'(\w)-\s+(\w)', r'\1\2', page_text)
                        full_text += page_text + " "
                    except Exception as e:
                        asyncio.run(self.logger.warning(f"Error extracting page {page_num}: {e}"))
                        continue
                
                return full_text.strip()
            
            full_text = await loop.run_in_executor(None, _parse_pdf_bytes_sync)
            return await self.chunk_text_optimized(full_text, chunk_size)
        except Exception as e:
            await self.logger.error(f"ParserService.parse_pdf_bytes_optimized error: {e}")
            # Fallback to original method
            reader = self.pdf_reader(io.BytesIO(pdf_bytes))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            return await self.chunk_text(full_text, chunk_size)

    async def perform_action(self, file_path: str = None, pdf_bytes: bytes = None, text: str = None, chunk_size: int = 1000, **kwargs) -> List[str]:
        """
        Perform parsing action based on input type.
        Args:
            file_path (str): Path to file to parse
            pdf_bytes (bytes): PDF bytes to parse
            text (str): Text to chunk
            chunk_size (int): Size of chunks
        Returns:
            List[str]: Parsed and chunked text
        """
        try:
            if pdf_bytes:
                return await self.parse_pdf_bytes_optimized(pdf_bytes, chunk_size)
            elif file_path:
                return await self.parse_pdf_optimized(file_path, chunk_size)
            elif text:
                return await self.chunk_text_optimized(text, chunk_size)
            else:
                raise ValueError("Must provide either file_path, pdf_bytes, or text")
        except Exception as e:
            await self.logger.error(f"ParserService.perform_action error: {e}")
            return []
