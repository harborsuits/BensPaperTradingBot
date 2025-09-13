import time
from typing import Dict

class ProviderHealth:
    def __init__(self):
        self.status: Dict[str, str] = {}
        self.last_checked: Dict[str, float] = {}
        self.error_counts: Dict[str, int] = {}
        self.latencies: Dict[str, float] = {}

    def update(self, provider: str, status: str, latency: float = 0):
        self.status[provider] = status
        self.last_checked[provider] = time.time()
        self.latencies[provider] = latency
        if status == "error":
            self.error_counts[provider] = self.error_counts.get(provider, 0) + 1
        else:
            self.error_counts[provider] = 0

    def get_status(self, provider: str) -> str:
        return self.status.get(provider, "unknown")

    def get_latency(self, provider: str) -> float:
        return self.latencies.get(provider, 0)

    def get_error_count(self, provider: str) -> int:
        return self.error_counts.get(provider, 0)

    def is_healthy(self, provider: str) -> bool:
        return self.get_status(provider) == "ok" and self.get_error_count(provider) < 3
