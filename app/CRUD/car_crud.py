from app.db.crud import BaseCRUD
from app.services.fetch_car_data_service import FetchCarDataService
from app.db.bucket_operations import SupabaseBucketManager
from app.utils.logger import get_logger_instance, Logger
from app.services.parser_service import ParserService
from app.services.embedding_service import EmbeddingService
from app.utils.redis_cache import get_redis_cache, RedisCache
import os
import uuid
import json
import asyncio

class CarCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Car')
        self.fetch_car_data_service = FetchCarDataService()
        # Always use SupabaseBucketManager, never override from parent
        self.bucket_manager = SupabaseBucketManager()
        self.logger = get_logger_instance("car_crud")
        self.parser_service = ParserService()
        self.embedding_service = EmbeddingService()

    def unique_logic(self, make, model, year):
        """Generate a unique car id using make, model, and year."""
        if not (make and model and year):
            raise ValueError("Make, model, and year are required for unique car id.")
        return f"{make.strip().lower().replace(' ', '-')}-{model.strip().lower().replace(' ', '-')}-{year}"

    def ensure_list(self, value):
        """Ensure the value is a list, parsing from JSON if needed."""
        if isinstance(value, list):
            return value
        if value is None:
            return []
        try:
            return json.loads(value)
        except Exception:
            return []

    async def update_car_with_links(self, car_obj, manual_link, guide_links):
        """Update car record with manual URL and guide links, checking for existing manual in bucket. Adds error handling."""
        make = car_obj.get('make')
        model = car_obj.get('model')
        year = car_obj.get('year')
        car_id = car_obj.get('id')
        pdf_filename = f"{make}-{model}_{year}_EN_US.pdf"
        bucket_name = "manuals"
        logger = self.logger
        try:
            # Check if bucket exists
            buckets = await self.bucket_manager.list_buckets()
            bucket_names = [getattr(b, 'name', None) for b in buckets] if buckets else []
            manual_exists = False
            if bucket_name in bucket_names:
                # List files in the bucket and check for the manual
                try:
                    client = await self.bucket_manager.client
                    files = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: client.storage.from_(bucket_name).list()
                    )
                    file_names = [f['name'] if isinstance(f, dict) else getattr(f, 'name', None) for f in files]
                    manual_exists = pdf_filename in file_names
                except Exception as e:
                    manual_exists = False
                    await logger.error(f"Error listing files in bucket: {e}")
            else:
                try:
                    await self.bucket_manager.create_bucket(bucket_name)
                except Exception as e:
                    await logger.error(f"Error creating bucket: {e}")
                    return [car_obj]
            vector = None
            if not manual_exists and manual_link:
                try:
                    await self.fetch_car_data_service.download_pdf(manual_link, pdf_filename)
                    # Vectorize the PDF before uploading
                    try:
                        chunks = await self.parser_service.parse_pdf(pdf_filename)
                        if chunks:
                            vectors = await self.embedding_service.embed_texts(chunks)
                            # Use the average vector for the car (or first vector if only one chunk)
                            if vectors:
                                import numpy as np
                                if len(vectors) == 1:
                                    vector = vectors[0]
                                else:
                                    vector = np.mean(np.array(vectors), axis=0).tolist()
                                car_obj['vector'] = vector
                    except Exception as e:
                        await logger.error(f"Error vectorizing PDF: {e}")
                    await self.bucket_manager.upload_file(bucket_name, pdf_filename, pdf_filename)
                    if os.path.exists(pdf_filename):
                        os.remove(pdf_filename)
                except Exception as e:
                    await logger.error(f"Error downloading/uploading manual: {e}")
            # If manual already exists, try to vectorize if not already present
            elif manual_exists and os.path.exists(pdf_filename):
                try:
                    chunks = await self.parser_service.parse_pdf(pdf_filename)
                    if chunks:
                        vectors = await self.embedding_service.embed_texts(chunks)
                        if vectors:
                            import numpy as np
                            if len(vectors) == 1:
                                vector = vectors[0]
                            else:
                                vector = np.mean(np.array(vectors), axis=0).tolist()
                            car_obj['vector'] = vector
                except Exception as e:
                    await logger.error(f"Error vectorizing existing PDF: {e}")
            # Update car record
            update_data = {
                'owner_manual_url': f"{bucket_name}/{pdf_filename}" if manual_link else "",
                'car_guide_links': self.ensure_list(guide_links) if guide_links else [],
                'vector': car_obj.get('vector')
            }
            try:
                await self.update({'id': car_id}, update_data)
                car_obj.update(update_data)
                car_obj['car_guide_links'] = self.ensure_list(car_obj.get('car_guide_links'))
            except Exception as e:
                await logger.error(f"Error updating car record: {e}")
            return [car_obj]
        except Exception as e:
            await logger.error(f"update_car_with_links unexpected error: {e}")
            return [car_obj]

    async def create(self, data):
        """Create a car record, fetch and attach manual and guide links."""
        make = data.get('make')
        model = data.get('model')
        year = data.get('year')
        if not data.get('id'):
            data['id'] = self.unique_logic(make, model, year)
        car_id = data['id']
        # Check if car with this id already exists
        existing = await self.read({'id': car_id})
        if existing and isinstance(existing, list) and existing:
            return existing
        # Ensure NOT NULL fields are not None
        data['owner_manual_url'] = data.get('owner_manual_url') or ""
        data['service_manual_url'] = data.get('service_manual_url') or ""
        data['car_guide_links'] = self.ensure_list(data.get('car_guide_links'))
        # Ensure vector is not null
        if data.get('vector') is None:
            data['vector'] = []
        # 1. Create the car record first
        car = await super().create(data)
        if not car or not isinstance(car, list) or not car[0]:
            return car
        car_obj = car[0]
        make = car_obj.get('make')
        model = car_obj.get('model')
        year = car_obj.get('year')
        car_id = car_obj.get('id')
        if not (make and model and year and car_id):
            return car
        # 2. Fetch manual link and guide links
        try:
            manual_url_dict = self.fetch_car_data_service.build_url(make, model, year)
            manual_link = await self.fetch_car_data_service.scrape_links(
                manual_url_dict['Owner_Manual'], req_type="owner_manual"
            )
            guide_links = await self.fetch_car_data_service.scrape_links(
                manual_url_dict['Car_guide_link'], req_type="car_guide_link"
            )
            return await self.update_car_with_links(car_obj, manual_link, guide_links)
        except Exception as e:
            # Log error, but return car as created
            logger = self.logger
            await logger.error(f"Car post-create logic error: {e}")
            return car

    async def get_car_by_make_model_year(self, make: str, model: str, year: int):
        """Retrieve a car by make, model, and year."""
        car_id = self.unique_logic(make, model, year)
        cars = await self.read({'id': car_id})
        if cars and isinstance(cars, list) and cars:
            return cars[0]
        return None

    async def get_car_by_id(self, car_id: str, cache: RedisCache = None):
        """Retrieve a car by its unique id, with optional Redis caching."""
        cache_key = f"car:{car_id}"
        if cache:
            cached = await cache.get(cache_key)
            if cached:
                return cached
        cars = await self.read({'id': car_id})
        if cars and isinstance(cars, list) and cars:
            car = cars[0]
            if cache:
                await cache.set(cache_key, car)
            return car
        return None
