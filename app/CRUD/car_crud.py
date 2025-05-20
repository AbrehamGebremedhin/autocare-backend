from app.db.crud import BaseCRUD
from app.services.fetch_car_data_service import FetchCarDataService
from app.db.bucket_operations import SupabaseBucketManager
import os
import uuid
import json

class CarCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Car')
        self.fetch_car_data_service = FetchCarDataService()
        # Always use SupabaseBucketManager, never override from parent
        self.bucket_manager = SupabaseBucketManager()

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
        """Update car record with manual URL and guide links."""
        make = car_obj.get('make')
        model = car_obj.get('model')
        year = car_obj.get('year')
        car_id = car_obj.get('id')
        pdf_filename = f"{make}-{model}_{year}_EN_US.pdf"
        bucket_name = "manuals"
        # Download and upload manual
        await self.fetch_car_data_service.download_pdf(manual_link, pdf_filename)
        await self.bucket_manager.create_bucket(bucket_name)
        await self.bucket_manager.upload_file(bucket_name, pdf_filename, pdf_filename)
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        # Update car record
        update_data = {
            'owner_manual_url': f"{bucket_name}/{pdf_filename}",
            'car_guide_links': self.ensure_list(guide_links)
        }
        await self.update({'id': car_id}, update_data)
        car_obj.update(update_data)
        car_obj['car_guide_links'] = self.ensure_list(car_obj.get('car_guide_links'))
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
        print('DEBUG: Data to insert into Car table:', data)
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
            logger = getattr(getattr(self, 'fetch_car_data_service', None), 'logger', None)
            if logger:
                await logger.error(f"Car post-create logic error: {e}")
            return car
