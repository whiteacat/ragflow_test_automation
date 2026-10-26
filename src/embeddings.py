"""bge-m3 语义兜底（可选）。

配置位于 config/test.yaml 的 `embedding:` 段。**默认关闭**：未启用或未填
base_url 时，sim/best_similarity 一律返回 None，所有上层规则判定与关闭前完全一致。

支持两种接口：
  - openai：POST {base_url}/embeddings   {"model":..., "input":[...]}  （vLLM / Ollama / TEI 的 OpenAI 兼容端点）
  - tei   ：POST {base_url}/embed        {"inputs":[...], "model":...}
"""
from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests

DEFAULT_DIM = 1024          # bge-m3 输出维度
MAX_CACHE = 20000
_SENT_SPLIT = re.compile(r"[。；;！？!?\n]+")


def sentences(text: str, limit: int = 40) -> List[str]:
    """按中文/英文句末标点切句，过滤过短片段，最多返回 limit 句。"""
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "")]
    return [p for p in parts if len(p) >= 2][:limit]


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    interface: str = "openai"          # openai | tei
    base_url: str = ""
    model: str = "bge-m3"
    api_key: str = ""
    dim: int = DEFAULT_DIM
    timeout_s: float = 30.0
    batch_size: int = 32
    # 语义兜底阈值：仅在字面/模糊规则都没命中时才使用，不会放宽已达成的命中
    fact_label_threshold: float = 0.75
    evidence_threshold: float = 0.80

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "Settings":
        d = dict(raw or {})
        api_key = d.get("api_key") or os.environ.get("EMBEDDING_API_KEY") or ""
        return cls(
            enabled=bool(d.get("enabled", False)),
            interface=str(d.get("interface") or "openai").lower(),
            base_url=str(d.get("base_url") or "").rstrip("/"),
            model=str(d.get("model") or "bge-m3"),
            api_key=str(api_key),
            dim=int(d.get("dim") or DEFAULT_DIM),
            timeout_s=float(d.get("timeout_s") or 30.0),
            batch_size=max(1, int(d.get("batch_size") or 32)),
            fact_label_threshold=float(d.get("fact_label_threshold") or 0.75),
            evidence_threshold=float(d.get("evidence_threshold") or 0.80),
        )


class _Embedder:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._last_error = ""

    # -- HTTP --
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}

    def _post(self, batch: Sequence[str]) -> List[List[float]]:
        if self.cfg.interface == "tei":
            url = f"{self.cfg.base_url}/embed"
            payload: Dict[str, Any] = {"inputs": list(batch), "model": self.cfg.model}
        else:
            url = f"{self.cfg.base_url}/embeddings"
            payload = {"model": self.cfg.model, "input": list(batch)}
        resp = requests.post(url, json=payload, headers=self._headers(),
                             timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        return _parse_vectors(resp.json())

    def embed(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        uniq = [t for t in dict.fromkeys(texts) if t]
        if not uniq:
            return []
        with self._lock:
            missing = [t for t in uniq if t not in self._cache]
            for i in range(0, len(missing), self.cfg.batch_size):
                batch = missing[i:i + self.cfg.batch_size]
                try:
                    vectors = self._post(batch)
                except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    return None
                if len(vectors) != len(batch):
                    self._last_error = f"向量数与输入数不一致: {len(vectors)} != {len(batch)}"
                    return None
                for text, vec in zip(batch, vectors):
                    if vec:
                        self._cache[text] = vec
            if len(self._cache) > MAX_CACHE:
                self._cache.clear()
            out = [self._cache.get(t) for t in texts]
        if any(v is None for v in out):
            return None
        return out  # type: ignore[return-value]

    def last_error(self) -> str:
        return self._last_error


def _parse_vectors(data: Any) -> List[List[float]]:
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            rows = sorted(data, key=lambda r: r.get("index", 0))
            return [[float(x) for x in r["embedding"]] for r in rows]
        return [[float(x) for x in vec] for vec in data]
    if isinstance(data, dict):
        for key in ("embeddings", "data", "vectors"):
            if key in data:
                return _parse_vectors(data[key])
    raise ValueError(f"无法解析 embedding 响应: {type(data).__name__}")


def _cos(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if not n:
        return None
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if not na or not nb:
        return None
    return dot / (na * nb)


_settings = Settings()
_embedder: Optional[_Embedder] = None
_state_lock = threading.Lock()


def configure(raw: Optional[Dict[str, Any]]) -> Settings:
    """从配置字典装载。未启用/缺 base_url 时置为不可用（不影响规则判定）。"""
    global _settings, _embedder
    s = Settings.from_dict(raw)
    with _state_lock:
        _settings = s
        _embedder = _Embedder(s) if s.enabled and s.base_url else None
    return s


def reset() -> None:
    configure(None)


def settings() -> Settings:
    return _settings


def enabled() -> bool:
    return _embedder is not None


def last_error() -> str:
    return _embedder.last_error() if _embedder else ""


def similarity(a: str, b: str) -> Optional[float]:
    """两段文本的余弦相似度；未启用或请求失败返回 None。"""
    if _embedder is None or not a or not b:
        return None
    vectors = _embedder.embed([a, b])
    if not vectors or len(vectors) < 2:
        return None
    return _cos(vectors[0], vectors[1])


def best_similarity(text: str, candidates: Sequence[str]) -> Optional[float]:
    """text 与候选集的最大余弦相似度；未启用或请求失败返回 None。"""
    if _embedder is None or not text:
        return None
    cands = [c for c in candidates if c]
    if not cands:
        return None
    vectors = _embedder.embed([text] + cands)
    if not vectors or len(vectors) != len(cands) + 1:
        return None
    base = vectors[0]
    sims = [s for s in (_cos(base, v) for v in vectors[1:]) if s is not None]
    return max(sims) if sims else None


def status() -> Dict[str, Any]:
    s = _settings
    return {
        "enabled": enabled(),
        "interface": s.interface,
        "base_url": s.base_url,
        "model": s.model,
        "dim": s.dim,
        "fact_label_threshold": s.fact_label_threshold,
        "evidence_threshold": s.evidence_threshold,
        "last_error": last_error(),
    }
