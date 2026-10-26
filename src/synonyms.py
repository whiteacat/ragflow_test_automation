"""同义词表：术语等价组 + 可选核心词，支持热加载。

配置文件默认 `config/synonyms.yaml`（可用 `SYNONYMS_PATH` 覆盖），两种写法：

```yaml
水压试验: [水压强度试验, 打压试验]          # 简写：只声明等价词
过热器:
  synonyms: [过热段]                        # 显式写法
  core: [过热, 蒸汽温度]                    # 可选：核心词共现也算近似命中
```

说明：
- 等价组会做传递闭包（A~B、B~C ⇒ A~C），比较前统一走 `textnorm.normalize`；
- `core` 不自动生成（中文分词不可靠），只在配置里显式给出时才启用，
  匹配口径为「核心词全部出现在答案中」，只给 0.8 分，不给满分；
- 文件缺失/解析失败不报错，等价表为空，判定退回纯字面口径。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from .textnorm import normalize

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "synonyms.yaml"
RELOAD_INTERVAL_S = 1.0


class Table:
    """等价组与核心词表；不可变，热加载时整体替换。"""

    def __init__(self, path: Optional[str], mtime: float,
                 groups: Dict[str, Tuple[str, ...]],
                 cores: Dict[str, Tuple[str, ...]],
                 error: str = "") -> None:
        self.path = path
        self.mtime = mtime
        self.groups = groups
        self.cores = cores
        self.error = error

    def expand(self, label: str) -> Tuple[str, ...]:
        return self.groups.get(normalize(label), ())

    def core(self, label: str) -> Tuple[str, ...]:
        return self.cores.get(normalize(label), ())

    def synonyms_of(self, label: str) -> List[str]:
        return list(self.expand(label))

    def variants(self, label: str) -> Tuple[str, ...]:
        """标签的等价写法：整词同义 + 长标签内部的术语替换。

        例：组内 汽轮机↔汽机 时，``汽轮机润滑油压`` → ``汽机润滑油压``。
        """
        base = normalize(label)
        out: List[str] = [base]
        for word, alts in self.groups.items():
            if not word or word == base or word not in base:
                continue
            for alt in alts:
                candidate = base.replace(word, alt)
                if candidate not in out:
                    out.append(candidate)
        return tuple(out)


def _merge(classes: Dict[str, int], words: Iterable[str], next_id: List[int]) -> None:
    members = [w for w in dict.fromkeys(words) if w]
    if len(members) < 2:
        return
    ids = {classes.get(w) for w in members if w in classes}
    group_id = min(ids) if ids else next_id[0]
    if not ids:
        next_id[0] += 1
    for word in members:
        classes[word] = group_id


def parse(raw: object) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]]]:
    """把 yaml 映射解析为 (等价组, 核心词)。"""
    if not isinstance(raw, dict):
        return {}, {}
    classes: Dict[str, int] = {}
    next_id = [0]
    cores: Dict[str, Tuple[str, ...]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        head = normalize(key)
        aliases: List[str] = []
        core_words: List[str] = []
        if isinstance(value, list):
            aliases = [str(v) for v in value if str(v).strip()]
        elif isinstance(value, dict):
            raw_aliases = value.get("synonyms") or value.get("aliases") or []
            if isinstance(raw_aliases, str):
                raw_aliases = [raw_aliases]
            aliases = [str(v) for v in raw_aliases if str(v).strip()]
            raw_core = value.get("core") or []
            if isinstance(raw_core, str):
                raw_core = [raw_core]
            core_words = [normalize(str(v)) for v in raw_core if str(v).strip()]
        members = [head] + [normalize(a) for a in aliases]
        _merge(classes, members, next_id)
        if core_words:
            for word in members:
                cores[word] = tuple(core_words)
    grouped: Dict[int, List[str]] = {}
    for word, group_id in classes.items():
        grouped.setdefault(group_id, []).append(word)
    index: Dict[str, Tuple[str, ...]] = {}
    for members in grouped.values():
        for word in members:
            index[word] = tuple(w for w in members if w != word)
    return index, cores


def load(path: Optional[str] = None) -> Table:
    target = path or os.environ.get("SYNONYMS_PATH") or str(DEFAULT_PATH)
    if not os.path.exists(target):
        return Table(target, 0.0, {}, {})
    try:
        with open(target, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        mtime = os.path.getmtime(target)
    except (OSError, yaml.YAMLError) as exc:
        return Table(target, 0.0, {}, {}, error=f"{type(exc).__name__}: {exc}")
    groups, cores = parse(raw)
    return Table(target, mtime, groups, cores)


_table: Optional[Table] = None
_lock = threading.Lock()
_last_check = 0.0


def configure(path: Optional[str] = None) -> Table:
    """显式装载（测试或切换文件时用）；path=None 走默认路径/环境变量。"""
    global _table, _last_check
    with _lock:
        _table = load(path)
        _last_check = time.monotonic()
    return _table


def reset() -> None:
    global _table, _last_check
    with _lock:
        _table = None
        _last_check = 0.0


def table() -> Table:
    """取当前表；文件 mtime 变化时自动重载（默认最多每秒检查一次）。"""
    global _table, _last_check
    with _lock:
        if _table is None:
            _table = load()
            _last_check = time.monotonic()
            return _table
        if time.monotonic() - _last_check < RELOAD_INTERVAL_S:
            return _table
        _last_check = time.monotonic()
        path = _table.path
        try:
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else 0.0
        except OSError:
            mtime = 0.0
        if mtime != _table.mtime:
            _table = load(path)
        return _table


def expand(label: str) -> Tuple[str, ...]:
    return table().expand(label)


def core(label: str) -> Tuple[str, ...]:
    return table().core(label)


def variants(label: str) -> Tuple[str, ...]:
    return table().variants(label)
