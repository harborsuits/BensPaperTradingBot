from fastapi import APIRouter, Response

# Safe import: if prometheus_client is missing, define no-op metrics so the app still starts
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    _PROM_OK = True
except Exception:  # pragma: no cover - fallback path
    _PROM_OK = False

    class _Noop:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def time(self):
            class _Timer:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Timer()

    # Dummy stand-ins so imports elsewhere keep working
    Counter = _Noop  # type: ignore
    Histogram = _Noop  # type: ignore

    def generate_latest():  # type: ignore
        return b""

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


# Counters (real or no-op based on availability)
benbot_logs_total = Counter("benbot_logs_total", "Total logs fetched/seen", ["source"])  # source: rest|ws
benbot_context_requests_total = Counter("benbot_context_requests_total", "Context requests served", ["path"])  # path: /api/context

# Optional latency histogram
benbot_context_latency_seconds = Histogram(
    "benbot_context_latency_seconds",
    "Context handler latency (s)",
)

# Prometheus text-format endpoint (separate from existing JSON /metrics)
metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics/prom")
def metrics_prom() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


