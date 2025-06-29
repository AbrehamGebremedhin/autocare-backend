import datetime

def serialize_datetimes(obj):
    """
    Recursively convert all datetime objects in a dict/list to ISO strings.
    """
    if isinstance(obj, dict):
        return {k: serialize_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetimes(i) for i in obj]
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    return obj
