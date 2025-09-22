from app.db.crud import BaseCRUD
from app.services.fetch_car_data_service import FetchCarDataService
from app.services.car_vectorization_service import CarVectorizationService
from app.db.bucket_operations import SupabaseBucketManager
from app.utils.logger import get_logger_instance, Logger
from app.services.parser_service import ParserService
from app.services.embedding_service import EmbeddingService
from app.utils.redis_cache import get_redis_cache, RedisCache
from typing import List, Dict, Optional, Any
import os
import uuid
import json
import asyncio

class CarCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Car')
        self.fetch_car_data_service = FetchCarDataService()
        self.vectorization_service = CarVectorizationService()
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

    async def create_or_get_car(self, data, websocket=None, session_id=None):
        """
        Create a car record if it doesn't exist, or return existing car.
        Ensures cars are unique and handles vectorization.
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
            await self.logger.info(f"Car {car_id} already exists, returning existing car")
            
            # Check vectorization status
            status = await self.vectorization_service.get_car_vectorization_status(car_id)
            existing_car['is_vectorized'] = status['is_vectorized']
            existing_car['vector_chunk_count'] = status['chunk_count']
            
            return [existing_car]
        
        # Initialize required fields
        data['owner_manual_url'] = data.get('owner_manual_url') or ""
        data['service_manual_url'] = data.get('service_manual_url') or ""
        data['car_guide_links'] = self.ensure_list(data.get('car_guide_links'))
        data['is_vectorized'] = False
        data['vector_chunk_count'] = 0
        
        # Create the car record
        car = await super().create(data)
        if not car or not isinstance(car, list) or not car[0]:
            return car
        
        car_obj = car[0]
        
        # Fetch and vectorize manual synchronously (wait for completion)
        await self.logger.info(f"Starting synchronous vectorization for car {car_obj.get('id')}")
        vectorization_result = await self._fetch_and_vectorize_manual(car_obj, websocket, session_id)
        
        if vectorization_result.get("success"):
            await self.logger.info(f"Car {car_obj.get('id')} successfully vectorized with {vectorization_result.get('chunk_count', 0)} chunks")
        else:
            await self.logger.warning(f"Car {car_obj.get('id')} vectorization failed: {vectorization_result.get('error', 'Unknown error')}")
        
        return car

    async def _fetch_and_vectorize_manual(
        self, 
        car_obj=None, 
        websocket=None, 
        session_id=None, 
        car_id=None, 
        make=None, 
        model=None, 
        year=None,
        chunk_size=800,
        overlap=200
    ):
        """Background task to fetch and vectorize car manual with configurable chunking"""
        # Support both car_obj and individual parameters
        if car_obj:
            make = car_obj.get('make')
            model = car_obj.get('model')  
            year = car_obj.get('year')
            car_id = car_obj.get('id')
        elif car_id and make and model and year:
            # Individual parameters provided
            pass
        else:
            return {"success": False, "error": "Either car_obj or all individual parameters must be provided"}
        
        try:
            # Fetch manual link and guide links
            manual_url_dict = self.fetch_car_data_service.build_url(make, model, year)
            manual_link = await self.fetch_car_data_service.scrape_links(
                manual_url_dict['Owner_Manual'], req_type="owner_manual"
            )
            guide_links = await self.fetch_car_data_service.scrape_links(
                manual_url_dict['Car_guide_link'], req_type="car_guide_link"
            )
            
            # Update car with links
            update_data = {
                'owner_manual_url': manual_link or "",
                'car_guide_links': self.ensure_list(guide_links) if guide_links else []
            }
            
            # Handle manual PDF and vectorization
            if manual_link:
                pdf_filename = f"{make}-{model}_{year}_EN_US.pdf"
                bucket_name = "manuals"
                
                # Ensure bucket exists
                await self._ensure_bucket_exists(bucket_name)
                
                # Check if PDF already exists in bucket
                pdf_exists = await self._check_pdf_exists(bucket_name, pdf_filename)
                
                if not pdf_exists:
                    # Download and upload PDF
                    await self.fetch_car_data_service.download_pdf(manual_link, pdf_filename)
                    await self.bucket_manager.upload_file(bucket_name, pdf_filename, pdf_filename)
                    if os.path.exists(pdf_filename):
                        os.remove(pdf_filename)
                
                # Vectorize manual from PDF
                result = await self.vectorization_service.vectorize_car_manual_from_pdf(
                    car_id=car_id,
                    make=make,
                    model=model,
                    year=year,
                    pdf_filename=pdf_filename,
                    websocket=websocket,
                    session_id=session_id,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                if result['success']:
                    update_data.update({
                        'is_vectorized': True,
                        'vector_chunk_count': result['chunk_count']
                    })
                    await self.logger.info(f"Successfully vectorized car {car_id} with {result['chunk_count']} chunks")
                else:
                    await self.logger.error(f"Failed to vectorize car {car_id}: {result.get('error')}")
            
            # Update car record
            await self.update({'id': car_id}, update_data)
            
            # Return result for API endpoint use
            if 'result' in locals() and result['success']:
                return result
            else:
                return {"success": False, "error": "Manual vectorization failed"}
            
        except Exception as e:
            await self.logger.error(f"Error in background vectorization for car {car_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure bucket exists, create if not"""
        try:
            buckets = await self.bucket_manager.list_buckets()
            bucket_names = [getattr(b, 'name', None) for b in buckets] if buckets else []
            
            if bucket_name not in bucket_names:
                await self.bucket_manager.create_bucket(bucket_name)
                await self.logger.info(f"Created bucket: {bucket_name}")
        except Exception as e:
            await self.logger.error(f"Error ensuring bucket {bucket_name}: {str(e)}")

    async def _check_pdf_exists(self, bucket_name: str, pdf_filename: str) -> bool:
        """Check if PDF exists in bucket"""
        try:
            client = self.bucket_manager.client
            files = await asyncio.get_running_loop().run_in_executor(
                None, lambda: client.storage.from_(bucket_name).list()
            )
            file_names = [f['name'] if isinstance(f, dict) else getattr(f, 'name', None) for f in files]
            return pdf_filename in file_names
        except Exception as e:
            await self.logger.error(f"Error checking if PDF exists: {str(e)}")
            return False

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

    async def search_car_manual(self, query: str, car_id: str = None, top_k: int = 5) -> List[Dict]:
        """Search car manual using vectorized chunks in Milvus"""
        return await self.vectorization_service.search_car_manual(
            query=query, 
            car_id=car_id, 
            top_k=top_k
        )

    async def get_car_by_make_model_year(self, make: str, model: str, year: int):
        """Retrieve a car by make, model, and year."""
        car_id = self.unique_logic(make, model, year)
        cars = await self.read({'id': car_id})
        if cars and isinstance(cars, list) and cars:
            return cars[0]
        return None

    async def get_car_by_id(self, car_id: str, cache: RedisCache = None):
        """Retrieve a car by its unique id, with optional Redis caching."""
        # Normalize car_id to handle different formats
        normalized_id = self._normalize_car_id(car_id)
        
        # First try with the original car_id
        cache_key = f"car:{car_id}"
        if cache:
            cached = await cache.get(cache_key)
            if cached:
                return cached
                
        cars = await self.read({'id': car_id})
        
        # If not found, try with normalized ID
        if not cars or not isinstance(cars, list) or not cars:
            if normalized_id != car_id:
                # Try again with normalized ID
                normalized_cache_key = f"car:{normalized_id}"
                if cache:
                    cached = await cache.get(normalized_cache_key)
                    if cached:
                        return cached
                
                cars = await self.read({'id': normalized_id})
                
                # Log attempt with normalized ID
                if cars and isinstance(cars, list) and cars:
                    await self.logger.info(f"Found car with normalized ID: {normalized_id} (original: {car_id})")
                else:
                    await self.logger.warning(f"Car not found with original ID '{car_id}' or normalized ID '{normalized_id}'")
        
        if cars and isinstance(cars, list) and cars:
            car = cars[0]
            if cache:
                await cache.set(cache_key, car)
                if normalized_id != car_id:
                    await cache.set(f"car:{normalized_id}", car)
            return car
            
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

    async def get_owner_manual_chunks(self, car_id: str, query: str = None, top_k: int = 5) -> List[Dict]:
        """
        Retrieve vectorized manual chunks for a car from Milvus.
        This replaces the old get_owner_manual_text method.
        """
        try:
            # If query provided, do semantic search
            if query:
                results = await self.search_car_manual(query=query, car_id=car_id, top_k=top_k)
                await self.logger.info(f"Found {len(results)} relevant chunks for car {car_id} with query")
                return results
            
            # Otherwise get limited chunks for the car (not all)
            status = await self.vectorization_service.get_car_vectorization_status(car_id)
            
            if not status['is_vectorized']:
                await self.logger.warning(f"Car {car_id} is not vectorized yet")
                return []
            
            # Get limited chunks (not all) - respecting the top_k parameter
            # This prevents performance issues when retrieving all 500+ chunks
            results = await asyncio.get_running_loop().run_in_executor(
                None, 
                lambda: self.vectorization_service.milvus_handler.car_collection.query(
                    expr=f'car_id == "{car_id}"',
                    output_fields=["id", "car_id", "make", "model", "year", "content_chunk", "chunk_index"],
                    limit=min(top_k, 100)  # Respect top_k but cap at 100 to prevent performance issues
                )
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id", ""),
                    "car_id": result.get("car_id", ""),
                    "make": result.get("make", ""),
                    "model": result.get("model", ""),
                    "year": result.get("year", 0),
                    "chunk": result.get("content_chunk", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "score": 1.0,  # No semantic scoring for all chunks
                    "source": "owner_manual"
                })
            
            # Sort by chunk_index
            formatted_results.sort(key=lambda x: x.get("chunk_index", 0))
            
            await self.logger.info(f"Retrieved {len(formatted_results)} chunks for car {car_id}")
            return formatted_results
            
        except Exception as e:
            await self.logger.error(f"Error retrieving manual chunks for car {car_id}: {str(e)}")
            return []

    async def batch_get_cars_by_ids(self, car_ids: List[str], cache: RedisCache = None) -> Dict[str, dict]:
        """
        Retrieve multiple cars by their IDs in a single operation for better performance.
        Returns a dictionary mapping car_id to car data.
        """
        results = {}
        uncached_ids = []
        
        # Check cache first if provided
        if cache:
            for car_id in car_ids:
                cache_key = f"car:{car_id}"
                cached = await cache.get(cache_key)
                if cached:
                    results[car_id] = cached
                else:
                    uncached_ids.append(car_id)
        else:
            uncached_ids = car_ids
        
        # Fetch uncached cars in batch
        if uncached_ids:
            # Use batch query - this is more efficient than individual reads
            batch_cars = await self.read({'id': {'in': uncached_ids}}) if len(uncached_ids) > 1 else await self.read({'id': uncached_ids[0]})
            
            if batch_cars:
                for car in batch_cars:
                    car_id = car.get('id')
                    if car_id:
                        results[car_id] = car
                        # Cache the result
                        if cache:
                            cache_key = f"car:{car_id}"
                            await cache.set(cache_key, car)
        
        return results
