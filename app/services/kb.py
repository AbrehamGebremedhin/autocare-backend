import asyncio
import re
from pathlib import Path
from typing import Literal, TypedDict

from pypdf import PdfReader

from app.db import pool
from app.utils.logger import get_logger_instance

logger = get_logger_instance("kb")

_STOPWORDS = frozenset({
    "and", "the", "that", "with", "this", "for", "are", "was", "were", "has", "have", "had",
    "not", "but", "you", "your", "its", "can", "will", "when", "what", "why", "how", "from",
    "does", "did", "get", "got", "just", "very", "been", "than", "then", "there", "they",
    "them", "into", "about", "also", "while", "some", "any", "out",
})
_TOKEN = re.compile(r"[a-z0-9]+")
_INSERT_BATCH = 500


class Passage(TypedDict):
    source_type: str
    title: str
    page: int
    content: str


def build_tsquery(text: str, max_terms: int = 12) -> str:
    terms: dict[str, None] = {}
    for token in _TOKEN.findall(text.lower()):
        if len(token) >= 3 and token not in _STOPWORDS:
            terms.setdefault(token)
    return " | ".join(list(terms)[:max_terms])


def chunk_pages(pages: list[str], size: int = 1500) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for number, page in enumerate(pages, start=1):
        current = ""
        pieces: list[str] = []
        for para in (p.strip() for p in re.split(r"\n\s*\n", page or "")):
            if not para:
                continue
            pieces.extend(para[i:i + size] for i in range(0, len(para), size))
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > size:
                chunks.append((number, current))
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
        if current:
            chunks.append((number, current))
    return chunks


async def ingest_pdf(
    path: Path,
    *,
    source_type: Literal["manual", "book"],
    title: str,
    car_id: str | None = None,
) -> int:
    # Parse first: no database connection is held while the PDF is read.
    reader = await asyncio.to_thread(PdfReader, path)
    pages = await asyncio.to_thread(lambda: [(p.extract_text() or "").replace("\x00", "") for p in reader.pages])
    rows = [(source_type, car_id, title, page, text) for page, text in chunk_pages(pages)]
    if not rows:
        return 0
    # Replace the document atomically: a failed batch rolls the delete back too.
    async with pool.transaction() as tx:
        await tx.execute(
            "DELETE FROM kb_passages WHERE source_type = $1 AND title = $2 AND car_id IS NOT DISTINCT FROM $3",
            source_type, title, car_id,
        )
        for i in range(0, len(rows), _INSERT_BATCH):
            await tx.executemany(
                "INSERT INTO kb_passages (source_type, car_id, title, page, content) VALUES ($1, $2, $3, $4, $5)",
                rows[i:i + _INSERT_BATCH],
            )
    await logger.info(f"Ingested '{title}': {len(rows)} passages")
    return len(rows)


async def search(query: str, car_id: str, k: int) -> list[Passage]:
    tsq = build_tsquery(query)
    if not tsq:
        return []
    rows = await pool.fetch("SELECT * FROM search_kb($1, $2, $3)", tsq, car_id, k)
    return [
        Passage(source_type=r["source_type"], title=r["title"], page=r["page"], content=r["content"])
        for r in rows
    ]
