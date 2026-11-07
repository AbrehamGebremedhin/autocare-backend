"""Load PDFs from a directory into kb_passages.

`<car_id>_manual.pdf` is ingested as that car's manual; every other PDF is a book.
Usage: uv run scripts/ingest_kb.py [--dir car_data]
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import pool  # noqa: E402
from app.services import kb  # noqa: E402

MANUAL_SUFFIX = "_manual"


def parse_pdf_name(path: Path) -> dict:
    """Map a PDF filename to ingest_pdf kwargs (source_type, title, car_id)."""
    stem = path.stem
    if stem.endswith(MANUAL_SUFFIX):
        car_id = stem[: -len(MANUAL_SUFFIX)]
        return {"source_type": "manual", "title": f"{car_id} owner's manual", "car_id": car_id}
    return {"source_type": "book", "title": stem.replace("-", " ").title(), "car_id": None}


async def main(directory: Path) -> None:
    await pool.init_pool()
    try:
        for path in sorted(directory.glob("*.pdf")):
            args = parse_pdf_name(path)
            count = await kb.ingest_pdf(path, **args)
            print(f"{args['title']}: {count} passages")
    finally:
        await pool.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="car_data", type=Path)
    asyncio.run(main(parser.parse_args().dir))
