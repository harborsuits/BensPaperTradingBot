import functools
import time
from typing import Callable, Any

class SimpleCache:
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Any:
        if key in self.cache and (time.time() - self.timestamps[key]) < self.ttl:
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def cached(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            result = self.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        return wrapper
