import time
from typing import Any

class AdapterError(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_time=60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = 0
        self.open = False

    def call(self, func, *args, **kwargs) -> Any:
        if self.open:
            if time.time() - self.last_failure_time > self.recovery_time:
                self.open = False
                self.failure_count = 0
            else:
                raise AdapterError("Circuit breaker open. Skipping call.")
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.open = True
            raise AdapterError(f"Adapter call failed: {e}")
