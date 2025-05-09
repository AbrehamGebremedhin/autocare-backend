from .base import SupabaseDBHandler
from typing import Optional

class SupabaseBucketManager(SupabaseDBHandler):
    def create_bucket(self, bucket_name: str) -> Optional[dict]:
        return self.client.storage.create_bucket(bucket_name)

    def delete_bucket(self, bucket_name: str) -> Optional[dict]:
        return self.client.storage.delete_bucket(bucket_name)

    def list_buckets(self) -> Optional[list]:
        return self.client.storage.list_buckets()

    def upload_file(self, bucket_name: str, file_path: str, dest_path: str) -> Optional[dict]:
        with open(file_path, "rb") as f:
            return self.client.storage.from_(bucket_name).upload(dest_path, f)

    def download_file(self, bucket_name: str, file_path: str) -> Optional[bytes]:
        return self.client.storage.from_(bucket_name).download(file_path)

    def delete_file(self, bucket_name: str, file_path: str) -> Optional[dict]:
        return self.client.storage.from_(bucket_name).remove([file_path])
