import redis
import json

"""Importing redis as cache for the models and data"""
redis_client = redis.Redis(host='redis', port=6379)

def get_cache(key):
    """Get value cache in Redis by key"""
    try:
        value = redis_client.get(key)
        return json.loads(value) if value else None
    except:
        return None
    
def set_cache(key, value, ttl: int = 300):
    """Set value in cache with TTL"""
    try:
        redis_client.set(key, json.dumps(value), ex=ttl)
    except:
        pass