import os
import importlib.util
import inspect
import asyncio
from app.db.base import SupabaseDBHandler
from pydantic import BaseModel
from typing import get_args, get_origin, List, Dict
from app.utils.logger import get_logger_instance

logger = get_logger_instance("migration").logger

PY_TO_PG = {
    str: 'text',
    int: 'integer',
    float: 'double precision',
    bool: 'boolean',
    dict: 'jsonb',
}

def python_type_to_pg(py_type, field_name=None):
    origin = get_origin(py_type)
    # Special case for 'vector' field: List[float] with 1024 dimension
    if field_name == 'vector' and (origin is list or origin is List):
        elem_type = get_args(py_type)[0]
        if elem_type == float:
            return 'double precision[1024]'
    if origin is list or origin is List:
        return 'jsonb'  # Store lists as JSONB
    if origin is dict or origin is Dict:
        return 'jsonb'
    if hasattr(py_type, '__name__') and py_type.__name__ == 'EmailStr':
        return 'text'
    if hasattr(py_type, '__name__') and py_type.__name__ == 'datetime':
        return 'timestamp'
    return PY_TO_PG.get(py_type, 'text')

async def get_schema_models(schemas_path):
    models = []
    loop = asyncio.get_running_loop()
    for fname in os.listdir(schemas_path):
        if fname.endswith('.py') and fname != '__init__.py':
            module_name = f'app.schemas.{fname[:-3]}'
            spec = importlib.util.find_spec(module_name)
            if not spec:
                continue
            module = importlib.util.module_from_spec(spec)
            await loop.run_in_executor(None, spec.loader.exec_module, module)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                    models.append(obj)
    return models

def generate_create_table_sql(model):
    table_name = model.__name__.replace('Base', '').capitalize()
    fields = []
    primary_key = None

    for field, info in model.model_fields.items():
        pg_type = python_type_to_pg(info.annotation, field_name=field)
        nullable = 'NULL' if info.is_required is False else 'NOT NULL'
        fields.append(f'"{field}" {pg_type} {nullable}')
        if field == "id":
            primary_key = field

    fields_sql = ', '.join(fields)
    if primary_key:
        fields_sql += f', PRIMARY KEY ("{primary_key}")'
    return f'CREATE TABLE IF NOT EXISTS "{table_name}" ({fields_sql});'

async def migrate_all_schemas():
    schemas_path = os.path.join(os.path.dirname(__file__), '../schemas')
    models = await get_schema_models(schemas_path)
    db_handler = SupabaseDBHandler()
    async with db_handler.get_connection() as db:
        for model in models:
            sql = generate_create_table_sql(model)
            logger.info(f"Executing SQL for {model.__name__}:\n{sql}\n")
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: db.rpc('execute_sql', { 'sql': sql }).execute())
            logger.info(f"Response: {response}\n")
            if response and isinstance(response, dict) and response.get('error'):
                logger.error(f"Error migrating {model.__name__}: {response['error']}")
                raise Exception(f"Error migrating {model.__name__}: {response['error']}")
            logger.info(f'Migrated: {model.__name__}')

if __name__ == "__main__":
    asyncio.run(migrate_all_schemas())