from fastapi.testclient import TestClient
from trading_bot.api.app_new import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_context_contract():
    r = client.get("/api/context")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_logs_contract():
    r = client.get("/api/logs", params={"level": "INFO", "limit": 1})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


