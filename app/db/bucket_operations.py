from .base import SupabaseDBHandler
from typing import Optional
import asyncio

class SupabaseBucketManager(SupabaseDBHandler):
    async def create_bucket(self, bucket_name: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.storage.create_bucket(bucket_name))

    async def delete_bucket(self, bucket_name: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.storage.delete_bucket(bucket_name))

    async def list_buckets(self) -> Optional[list]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.storage.list_buckets())

    async def upload_file(self, bucket_name: str, file_path: str, dest_path: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        def _upload():
            with open(file_path, "rb") as f:
                return self.client.storage.from_(bucket_name).upload(dest_path, f)
        return await loop.run_in_executor(None, _upload)

    async def download_file(self, bucket_name: str, file_path: str) -> Optional[bytes]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.storage.from_(bucket_name).download(file_path))

    async def delete_file(self, bucket_name: str, file_path: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.storage.from_(bucket_name).remove([file_path]))
