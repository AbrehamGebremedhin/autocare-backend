import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin, urlparse

from app.core.config import get_settings
from app.db import pool
from app.db.crud import BaseCRUD, _ident
from app.services import kb, manual_search
from app.services.fetch_car_data_service import FetchCarDataService
from app.utils.logger import get_logger_instance

MAX_LOGGED_ID_CHARS = 64


def log_id(car_id: str) -> str:
    """A car id as it may appear in a log line: client-supplied ids can be arbitrarily long."""
    return car_id[:MAX_LOGGED_ID_CHARS]


class CarCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Car')
        self.logger = get_logger_instance("car_crud")

    def unique_logic(self, make, model, year):
        """Generate a unique car id using make, model, and year."""
        if not (make and model and year):
            raise ValueError("Make, model, and year are required for unique car id.")
        return f"{make.strip().lower().replace(' ', '-')}-{model.strip().lower().replace(' ', '-')}-{year}"

    def ensure_list(self, value):
        """Ensure the value is a list of proper CarGuideLink objects, parsing from JSON if needed."""
        if isinstance(value, list):
            # If it's already a list, ensure each item has the right structure
            result = []
            for item in value:
                if isinstance(item, dict) and 'link' in item:
                    result.append(item)
                elif isinstance(item, str):
                    # If it's a string, treat it as a link with no summary
                    result.append({'link': item, 'summary': None})
            return result
        if value is None:
            return []
        try:
            # Try to parse as JSON
            parsed = json.loads(value)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict) and 'link' in item:
                        result.append(item)
                    elif isinstance(item, str):
                        result.append({'link': item, 'summary': None})
                return result
            return []
        except Exception:
            return []

    async def create_or_get_car(self, data):
        """
        Create a car record if it doesn't exist, or return existing car.
        Ensures cars are unique.
        """
        make = data.get('make')
        model = data.get('model')
        year = data.get('year')
        
        if not (make and model and year):
            raise ValueError("Make, model, and year are required.")
        
        # Generate unique car ID
        if not data.get('id'):
            data['id'] = self.unique_logic(make, model, year)
        car_id = data['id']
        
        # Check if car already exists
        existing = await self.read({'id': car_id})
        if existing and isinstance(existing, list) and existing:
            existing_car = existing[0]
            await self.logger.info(f"Car {log_id(car_id)} already exists, returning existing car")
            return [existing_car]
        
        # Initialize required fields
        data['owner_manual_url'] = data.get('owner_manual_url') or ""
        data['service_manual_url'] = data.get('service_manual_url') or ""
        data['car_guide_links'] = self.ensure_list(data.get('car_guide_links'))
        
        # Two requests can add the same new car at once: the loser's insert is a no-op, and both read it back.
        columns = ", ".join(_ident(key) for key in data)
        params = ", ".join(f"${n}" for n in range(1, len(data) + 1))
        await pool.execute(
            f'INSERT INTO {self._table} ({columns}) VALUES ({params}) ON CONFLICT ("id") DO NOTHING', *data.values()
        )
        return [(await self.read({'id': car_id}))[0]]

    async def read(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Override read to properly parse car_guide_links JSON field"""
        result = await super().read(filters)
        
        # Process each car record to parse car_guide_links
        if result and isinstance(result, list):
            for car in result:
                if isinstance(car, dict) and 'car_guide_links' in car:
                    car['car_guide_links'] = self.ensure_list(car['car_guide_links'])
        
        return result

    async def create(self, data):
        """Create a car record - now redirects to create_or_get_car for uniqueness"""
        return await self.create_or_get_car(data)

    async def get_many(self, ids: List[str]) -> List[Dict[str, Any]]:
        """The cars with these ids in one query, in the order of `ids`; unknown ids are left out."""
        rows = {r["id"]: r for r in await pool.fetch(f"SELECT * FROM {self._table} WHERE id = ANY($1::text[])", ids)}
        cars = [rows[i] for i in ids if i in rows]
        for car in cars:
            if "car_guide_links" in car:
                car["car_guide_links"] = self.ensure_list(car["car_guide_links"])
        return cars

    async def get_car_by_id(self, car_id: str):
        """Retrieve a car by its unique id."""
        # Normalize car_id to handle different formats
        normalized_id = self._normalize_car_id(car_id)
        
        # First try with the original car_id
        cars = await self.read({'id': car_id})
        
        # If not found, try with normalized ID
        if not cars or not isinstance(cars, list) or not cars:
            if normalized_id != car_id:
                # Try again with normalized ID
                cars = await self.read({'id': normalized_id})
                
                # Log attempt with normalized ID
                if cars and isinstance(cars, list) and cars:
                    await self.logger.info(
                        f"Found car with normalized ID: {log_id(normalized_id)} (original: {log_id(car_id)})"
                    )
                else:
                    await self.logger.warning(
                        f"Car not found with original ID '{log_id(car_id)}' or normalized ID '{log_id(normalized_id)}'"
                    )
        
        if cars and isinstance(cars, list) and cars:
            return cars[0]
            
        return None
        
    def _normalize_car_id(self, car_id: str) -> str:
        """
        Normalize car ID to handle different formats and common variations.
        For example: "echo-toyota-2001" -> "toyota-echo-2001"
        """
        if not car_id:
            return car_id
            
        # Try to parse parts from the car_id
        parts = car_id.lower().strip().split('-')
        if len(parts) >= 3:
            # Check for common pattern where make and model are reversed
            # Common car makes that we can detect
            common_makes = ['toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'audi', 'mercedes', 'nissan', 
                           'mazda', 'subaru', 'hyundai', 'kia', 'lexus', 'acura', 'volkswagen', 'vw']
            
            # If the first part isn't a known make but the second is, swap them
            if parts[0] not in common_makes and parts[1] in common_makes:
                model, make = parts[0], parts[1]
                remaining = '-'.join(parts[2:])
                return f"{make}-{model}-{remaining}"
        
        # If we couldn't normalize it, return the original
        return car_id


_logger = get_logger_instance("car_ingest")


async def _set_kb_status(car_id: str, status: str, count: int = 0) -> None:
    try:
        await CarCRUD().update({"id": car_id}, {"kb_status": status, "kb_passage_count": count})
    except Exception as e:
        await _logger.error(f"Could not set kb_status={status} for {log_id(car_id)}: {type(e).__name__}")


async def _startmycar_manual(fetcher: FetchCarDataService, car: dict, pdf_path: Path) -> str | None:
    """Download startmycar's owner manual to pdf_path; the title to index it under, or None when it has none."""
    url = fetcher.build_url(car["make"], car["model"], car["year"])["Owner_Manual"]
    link = await fetcher.scrape_links(url)
    if not link or not await manual_search.fetch_pdf(urljoin(fetcher.BASE_URL, link), pdf_path):
        return None
    return f"{car['make']} {car['model']} {car['year']} owner's manual"


_ingesting: set[str] = set()  # car ids whose manual ingest is running in this process


async def ingest_car_manual(car: dict) -> None:
    """Run the manual ingest unless one is already running for this car (two quick adds start one, not two)."""
    if car["id"] in _ingesting:
        return
    _ingesting.add(car["id"])
    try:
        await _ingest_car_manual(car)
    finally:
        _ingesting.discard(car["id"])


async def _ingest_car_manual(car: dict) -> None:
    """Background task: index up to MANUAL_MAX_SOURCES manuals (startmycar, then Brave Search hits) into kb_passages.

    A source that fails or is not about this car is skipped. Never raises.
    """
    car_id = car["id"]
    settings = get_settings()
    max_sources = settings.MANUAL_MAX_SOURCES
    fetcher = FetchCarDataService()
    tmp_dir = None
    total = indexed = 0
    failed = False
    try:
        tmp_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="autocare-manual-"))
        label = f"{car['make']} {car['model']} {car['year']}"

        async def index(title: str, pdf_path: Path) -> None:
            nonlocal total, indexed
            count = await kb.ingest_pdf(pdf_path, source_type="manual", title=title, car_id=car_id)
            if count > 0:
                total, indexed = total + count, indexed + 1

        try:
            title = await _startmycar_manual(fetcher, car, tmp_dir / "startmycar.pdf")
            if title:
                await index(title, tmp_dir / "startmycar.pdf")
        except Exception as e:
            failed = True
            await _logger.error(f"startmycar manual failed for {log_id(car_id)}: {type(e).__name__}")

        if indexed < max_sources and settings.BRAVE_API_KEY:
            aliases = await manual_search.sibling_names(car["make"], car["model"], car["year"])
            models = [car["model"], *aliases]
            urls = await manual_search.search_manual_urls(car["make"], car["model"], car["year"], max_sources * 2, aliases)
            for n, url in enumerate(urls):
                if indexed >= max_sources:
                    break
                path = tmp_dir / f"web{n}.pdf"
                try:
                    if await manual_search.download_manual(url, path, models, car["year"]):
                        await index(f"{label} manual ({urlparse(url).hostname})", path)
                except Exception as e:
                    failed = True
                    await _logger.error(f"web manual failed for {log_id(car_id)}: {type(e).__name__}")

        await _set_kb_status(car_id, "ready" if indexed else "failed" if failed else "no_manual", total)
    except Exception as e:
        await _logger.error(f"Manual ingest failed for {log_id(car_id)}: {type(e).__name__}")
        await _set_kb_status(car_id, "failed")
    finally:
        await fetcher.cleanup()
        if tmp_dir:
            await asyncio.to_thread(shutil.rmtree, tmp_dir, True)
