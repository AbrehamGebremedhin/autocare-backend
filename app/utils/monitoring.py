import time
import functools
import uuid
import traceback
import asyncio
from typing import Callable, Any, Dict
from app.utils.logger import get_logger_instance

# Decorator for performance monitoring and error handling
# Tracks: response time, success, token usage (if provided in kwargs/result), error info

def monitor_and_handle(agent_name: str):
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                logger = get_logger_instance(agent_name)
                start_time = time.perf_counter()
                error_id = None
                metrics: Dict[str, Any] = {}
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.perf_counter() - start_time
                    metrics['response_time'] = elapsed
                    metrics['success'] = result.get('success', True) if isinstance(result, dict) else True
                    metrics['token_usage'] = result.get('token_usage') if isinstance(result, dict) and 'token_usage' in result else None
                    await logger.info(f"[PERF] {func.__name__} metrics: {metrics}")
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    error_id = str(uuid.uuid4())
                    tb = traceback.format_exc()
                    metrics['response_time'] = elapsed
                    metrics['success'] = False
                    metrics['error_type'] = type(e).__name__
                    metrics['error_id'] = error_id
                    await logger.error(f"[ERROR] {func.__name__} failed (ID: {error_id}): {e}\n{tb}")
                    # User-friendly error message
                    user_message = "An internal error occurred. Please try again later. (Error ID: %s)" % error_id
                    return {
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'error_id': error_id,
                        'user_message': user_message
                    }
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                logger = get_logger_instance(agent_name)
                start_time = time.perf_counter()
                error_id = None
                metrics: Dict[str, Any] = {}
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start_time
                    metrics['response_time'] = elapsed
                    metrics['success'] = result.get('success', True) if isinstance(result, dict) else True
                    metrics['token_usage'] = result.get('token_usage') if isinstance(result, dict) and 'token_usage' in result else None
                    logger.info(f"[PERF] {func.__name__} metrics: {metrics}")
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    error_id = str(uuid.uuid4())
                    tb = traceback.format_exc()
                    metrics['response_time'] = elapsed
                    metrics['success'] = False
                    metrics['error_type'] = type(e).__name__
                    metrics['error_id'] = error_id
                    logger.error(f"[ERROR] {func.__name__} failed (ID: {error_id}): {e}\n{tb}")
                    user_message = "An internal error occurred. Please try again later. (Error ID: %s)" % error_id
                    return {
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'error_id': error_id,
                        'user_message': user_message
                    }
            return wrapper
    return decorator
