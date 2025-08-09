#!/usr/bin/env python3
"""
Migration script to vectorize existing cars in the database.
This script will:
1. Fetch all existing cars from Supabase
2. For cars with manual text, vectorize and store in Milvus
3. Update car records to reflect vectorization status
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.CRUD.car_crud import CarCRUD
from app.services.car_vectorization_service import CarVectorizationService
from app.db.base import SupabaseDBHandler
from app.utils.logger import get_logger_instance
from typing import List, Dict, Any


class CarVectorizationMigration:
    def __init__(self):
        self.car_crud = CarCRUD()
        self.vectorization_service = CarVectorizationService()
        self.db_handler = SupabaseDBHandler()
        self.logger = get_logger_instance("migration")
        
    async def get_all_cars(self) -> List[Dict[str, Any]]:
        """Get all cars from the database"""
        try:
            async with self.db_handler.get_connection() as db:
                result = db.table('Car').select('*').execute()
                
                if result.data:
                    await self.logger.info(f"Found {len(result.data)} cars in database")
                    return result.data
                else:
                    await self.logger.info("No cars found in database")
                    return []
                    
        except Exception as e:
            await self.logger.error(f"Error fetching cars: {str(e)}")
            return []
    
    async def migrate_car(self, car: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a single car to vectorized storage"""
        car_id = car.get('id')
        make = car.get('make', '')
        model = car.get('model', '')
        year = car.get('year', 0)
        manual_text = car.get('text', '')
        
        await self.logger.info(f"Processing car {car_id} ({make} {model} {year})")
        
        # Check if already vectorized
        status = await self.vectorization_service.get_car_vectorization_status(car_id)
        if status['is_vectorized'] and status['chunk_count'] > 0:
            await self.logger.info(f"Car {car_id} already vectorized with {status['chunk_count']} chunks, skipping")
            return {
                "car_id": car_id,
                "status": "already_vectorized",
                "chunk_count": status['chunk_count']
            }
        
        # Skip if no manual text
        if not manual_text or not manual_text.strip():
            await self.logger.warning(f"Car {car_id} has no manual text, skipping vectorization")
            return {
                "car_id": car_id,
                "status": "no_text",
                "chunk_count": 0
            }
        
        # Vectorize the manual text
        try:
            result = await self.vectorization_service.vectorize_car_manual_text(
                car_id=car_id,
                make=make,
                model=model,
                year=year,
                manual_text=manual_text
            )
            
            if result['success']:
                # Update car record
                update_data = {
                    'is_vectorized': True,
                    'vector_chunk_count': result['chunk_count']
                }
                
                await self.car_crud.update({'id': car_id}, update_data)
                
                await self.logger.info(f"Successfully vectorized car {car_id} with {result['chunk_count']} chunks")
                return {
                    "car_id": car_id,
                    "status": "success",
                    "chunk_count": result['chunk_count']
                }
            else:
                await self.logger.error(f"Failed to vectorize car {car_id}: {result.get('error')}")
                return {
                    "car_id": car_id,
                    "status": "failed",
                    "error": result.get('error'),
                    "chunk_count": 0
                }
                
        except Exception as e:
            await self.logger.error(f"Exception vectorizing car {car_id}: {str(e)}")
            return {
                "car_id": car_id,
                "status": "exception",
                "error": str(e),
                "chunk_count": 0
            }
    
    async def run_migration(self, batch_size: int = 5, max_cars: int = None):
        """Run the complete migration"""
        await self.logger.info("Starting car vectorization migration")
        
        # Get all cars
        cars = await self.get_all_cars()
        if not cars:
            await self.logger.info("No cars to migrate")
            return
        
        # Limit cars if specified
        if max_cars:
            cars = cars[:max_cars]
            await self.logger.info(f"Limited migration to first {max_cars} cars")
        
        total_cars = len(cars)
        await self.logger.info(f"Migrating {total_cars} cars in batches of {batch_size}")
        
        results = {
            "total_cars": total_cars,
            "success": 0,
            "failed": 0,
            "already_vectorized": 0,
            "no_text": 0,
            "exception": 0,
            "details": []
        }
        
        # Process cars in batches
        for i in range(0, total_cars, batch_size):
            batch = cars[i:i + batch_size]
            await self.logger.info(f"Processing batch {i // batch_size + 1}/{(total_cars + batch_size - 1) // batch_size}")
            
            # Process batch concurrently
            tasks = [self.migrate_car(car) for car in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    await self.logger.error(f"Exception in batch processing: {str(result)}")
                    results["exception"] += 1
                else:
                    status = result["status"]
                    results[status] = results.get(status, 0) + 1
                    results["details"].append(result)
            
            # Small delay between batches to avoid overwhelming services
            await asyncio.sleep(1)
        
        # Print summary
        await self.logger.info("Migration completed!")
        await self.logger.info(f"Summary:")
        await self.logger.info(f"  Total cars: {results['total_cars']}")
        await self.logger.info(f"  Successful: {results['success']}")
        await self.logger.info(f"  Failed: {results['failed']}")
        await self.logger.info(f"  Already vectorized: {results['already_vectorized']}")
        await self.logger.info(f"  No text: {results['no_text']}")
        await self.logger.info(f"  Exceptions: {results['exception']}")
        
        # Show failed cars
        failed_cars = [r for r in results["details"] if r["status"] in ["failed", "exception"]]
        if failed_cars:
            await self.logger.warning(f"Failed cars ({len(failed_cars)}):")
            for car in failed_cars:
                await self.logger.warning(f"  {car['car_id']}: {car.get('error', 'Unknown error')}")
        
        return results


async def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate cars to vectorized storage")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing cars")
    parser.add_argument("--max-cars", type=int, help="Maximum number of cars to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't actually vectorize)")
    
    args = parser.parse_args()
    
    migration = CarVectorizationMigration()
    
    if args.dry_run:
        await migration.logger.info("DRY RUN MODE - No actual vectorization will occur")
        cars = await migration.get_all_cars()
        if args.max_cars:
            cars = cars[:args.max_cars]
        
        cars_with_text = [car for car in cars if car.get('text', '').strip()]
        
        print(f"Would migrate {len(cars_with_text)} cars out of {len(cars)} total cars")
        for car in cars_with_text[:10]:  # Show first 10
            print(f"  {car.get('id')} ({car.get('make')} {car.get('model')} {car.get('year')}) - {len(car.get('text', ''))} chars")
        
        if len(cars_with_text) > 10:
            print(f"  ... and {len(cars_with_text) - 10} more")
    else:
        await migration.run_migration(
            batch_size=args.batch_size,
            max_cars=args.max_cars
        )


if __name__ == "__main__":
    asyncio.run(main())
