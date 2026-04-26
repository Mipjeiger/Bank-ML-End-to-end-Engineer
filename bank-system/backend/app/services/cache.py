import redis
import json

"""Importing redis as cache for the models and data"""
redis_client = redis.Redis(host='redis', port=6379)

def get_cache(key):
    value = redis_client.get(key)
    return json.loads(value) if value else None
    
def set_cache(key, value):
    redis_client.set(key, json.dumps(value), ex=300)