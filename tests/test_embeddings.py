"""bge-m3 语义兜底：配置装载、两种接口、缓存、失败降级、规则接线。"""
import pytest
import requests

from src import embeddings
from src.embeddings import Settings
from src.metrics_generation import fact_covered
from src.metrics_retrieval import evidence_cover


def _vec(text: str):
    return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset():
    embeddings.reset()
    yield
    embeddings.reset()


def _cfg(**kw):
    base = {"enabled": True, "base_url": "http://127.0.0.1:9997/v1", "model": "bge-m3"}
    base.update(kw)
    return base


def test_disabled_by_default():
    assert not embeddings.enabled()
    assert embeddings.similarity("甲", "甲") is None
    assert embeddings.best_similarity("甲", ["甲"]) is None
    assert not fact_covered("给水泵出口压力0.6MPa", "泵出口压力为0.6MPa")


def test_enabled_requires_base_url():
    settings = embeddings.configure({"enabled": True, "base_url": ""})
    assert settings.enabled
    assert not embeddings.enabled()


def test_openai_interface_and_cache(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return _Resp({"data": [{"index": i, "embedding": _vec(t)}
                               for i, t in enumerate(json["input"])]})

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    embeddings.configure(_cfg(api_key="secret"))

    assert embeddings.similarity("水压试验压力", "水压试验压力") == pytest.approx(1.0)
    assert embeddings.similarity("水压试验压力", "水压试验压力") == pytest.approx(1.0)
    assert len(calls) == 1
    assert calls[0]["url"] == "http://127.0.0.1:9997/v1/embeddings"
    assert calls[0]["json"]["model"] == "bge-m3"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_tei_interface(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["url"] = url
        return _Resp([_vec(t) for t in json["inputs"]])

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    embeddings.configure(_cfg(interface="tei", base_url="http://127.0.0.1:8080"))

    assert embeddings.similarity("过热器出口温度", "过热器出口温度") == pytest.approx(1.0)
    assert seen["url"] == "http://127.0.0.1:8080/embed"


def test_http_error_degrades_to_none(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(embeddings.requests, "post", boom)
    embeddings.configure(_cfg())

    assert embeddings.similarity("甲", "乙") is None
    assert "connection refused" in embeddings.last_error()


def test_fact_covered_semantic_fallback(monkeypatch):
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "settings", lambda: Settings(enabled=True))
    monkeypatch.setattr(embeddings, "best_similarity", lambda text, cands: 0.9)

    assert fact_covered("给水泵出口压力0.6MPa", "泵出口压力为0.6MPa")

    monkeypatch.setattr(embeddings, "best_similarity", lambda text, cands: 0.5)
    assert not fact_covered("给水泵出口压力0.6MPa", "泵出口压力为0.6MPa")


def test_fact_covered_semantic_never_overrides_number_mismatch(monkeypatch):
    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "settings", lambda: Settings(enabled=True))
    monkeypatch.setattr(embeddings, "best_similarity", lambda text, cands: 0.99)

    assert not fact_covered("给水泵出口压力0.6MPa", "泵出口压力为0.8MPa")


def test_evidence_cover_semantic_fallback(monkeypatch):
    evidence = "给水泵出口压力不低于0.6MPa"
    contents = ["泵的出口压力须保持在0.6mpa以上"]
    assert evidence_cover(evidence, contents) == 0.0

    monkeypatch.setattr(embeddings, "enabled", lambda: True)
    monkeypatch.setattr(embeddings, "settings", lambda: Settings(enabled=True))
    monkeypatch.setattr(embeddings, "best_similarity", lambda text, cands: 0.85)
    assert evidence_cover(evidence, contents) == 1.0


def test_api_key_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "from-env")
    assert Settings.from_dict({"api_key": ""}).api_key == "from-env"
