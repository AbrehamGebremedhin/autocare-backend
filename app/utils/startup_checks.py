import asyncio
from app.db.milvus_handler import MilvusHandler
from app.db.base import SupabaseDBHandler
from app.utils.redis_cache import redis_cache

async def check_milvus_connection():
    try:
        handler = MilvusHandler()
        # Try a simple operation: list collections
        from pymilvus import utility
        collections = utility.list_collections()
        return True, None
    except Exception as e:
        return False, str(e)

async def check_supabase_connection():
    try:
        db_handler = SupabaseDBHandler()
        client = await db_handler.client
        # Try a simple operation: list tables
        tables = client.table('Car').select('*').limit(1).execute()
        return True, None
    except Exception as e:
        return False, str(e)

async def check_redis_connection():
    try:
        conn = await redis_cache.connect()
        pong = await conn.ping()
        if pong:
            return True, None
        return False, 'No PONG from Redis'
    except Exception as e:
        return False, str(e)
