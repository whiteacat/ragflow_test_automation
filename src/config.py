"""配置加载。"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 环境变量覆盖敏感项：RAGFLOW_API_KEY / RAGFLOW_BASE_URL
    cfg.setdefault("ragflow", {})
    if os.environ.get("RAGFLOW_API_KEY"):
        cfg["ragflow"]["api_key"] = os.environ["RAGFLOW_API_KEY"]
    if os.environ.get("RAGFLOW_BASE_URL"):
        cfg["ragflow"]["base_url"] = os.environ["RAGFLOW_BASE_URL"]
    return cfg
