"""检索基准：只跑 RAG-01~05/09/10 + 延迟（无需 chat_id）。

实现：复制 config 为临时文件并置 retrieval_only=true，复用 src.runner。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml  # noqa: E402
from src.config import load_config  # noqa: E402
from src.runner import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["run"]["retrieval_only"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
        tmp = f.name
    try:
        rep = run(tmp)
    finally:
        os.unlink(tmp)
    print(f"模式 retrieval_only n={rep['n_questions']}")
    for k, v in rep["verdicts"].items():
        print(f" - {k}: {v['desc']}")
    print(f"报告目录: {rep['_output_dir']}")


if __name__ == "__main__":
    main()
