"""生成基准：全链路（检索 + 问答），输出规则版 GEN 分 + 复核队列。"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.runner import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    args = ap.parse_args()
    rep = run(args.config)
    print(f"模式 {rep['mode']} n={rep['n_questions']} veto疑似={rep['veto_count']}")
    for k, v in rep["verdicts"].items():
        print(f" - {k}: {v['desc']}")
    print(f"报告目录: {rep['_output_dir']}  待复核: {rep['_review_items']} 条")


if __name__ == "__main__":
    main()
