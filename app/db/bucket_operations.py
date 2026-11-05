from .base import SupabaseDBHandler
from typing import Optional
import asyncio

class SupabaseBucketManager(SupabaseDBHandler):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SupabaseBucketManager, cls).__new__(cls)
            # Do NOT call SupabaseDBHandler.__new__ directly, let Python handle MRO
        return cls._instance

    async def create_bucket(self, bucket_name: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        client = self.client
        return await loop.run_in_executor(None, lambda: client.storage.create_bucket(bucket_name))

    async def delete_bucket(self, bucket_name: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        client = self.client
        return await loop.run_in_executor(None, lambda: client.storage.delete_bucket(bucket_name))

    async def list_buckets(self) -> Optional[list]:
        loop = asyncio.get_running_loop()
        client = self.client
        return await loop.run_in_executor(None, lambda: client.storage.list_buckets())

    async def upload_file(
        self, bucket_name: str, file_path: str, dest_path: str, content_type: Optional[str] = None
    ) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        client = self.client
        # Without a content type storage3 labels every object text/plain.
        options = {"content-type": content_type} if content_type else None
        def _upload():
            with open(file_path, "rb") as f:
                return client.storage.from_(bucket_name).upload(dest_path, f, options)
        return await loop.run_in_executor(None, _upload)

    async def delete_prefix(self, bucket_name: str, prefix: str, page: int = 100) -> int:
        """Remove every object under `prefix` (recursively); returns how many were removed."""
        loop = asyncio.get_running_loop()
        storage = self.client.storage.from_(bucket_name)

        def _files(folder: str) -> list[str]:
            # list() returns one folder level, a page at a time; sub-folders come back with id None.
            found: list[str] = []
            offset = 0
            while True:
                entries = storage.list(folder, {"limit": page, "offset": offset})
                for e in entries:
                    path = f"{folder}/{e['name']}"
                    found += _files(path) if e.get("id") is None else [path]
                if len(entries) < page:
                    return found
                offset += page

        def _delete() -> int:
            files = _files(prefix.strip("/"))
            for i in range(0, len(files), page):
                storage.remove(files[i : i + page])
            return len(files)

        return await loop.run_in_executor(None, _delete)

    async def download_file(self, bucket_name: str, file_path: str) -> Optional[bytes]:
        loop = asyncio.get_running_loop()
        client = self.client
        return await loop.run_in_executor(None, lambda: client.storage.from_(bucket_name).download(file_path))

    async def delete_file(self, bucket_name: str, file_path: str) -> Optional[dict]:
        loop = asyncio.get_running_loop()
        client = self.client
        return await loop.run_in_executor(None, lambda: client.storage.from_(bucket_name).remove([file_path]))
