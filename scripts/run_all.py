"""一键全量：python scripts/run_all.py --config config/test.yaml"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.runner import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    args = ap.parse_args()
    rep = run(args.config)
    print(json.dumps({k: v for k, v in rep.items()
                      if not k.startswith("_")}, ensure_ascii=False, indent=2)[:4000])
    print(f"\n报告目录: {rep['_output_dir']}  待复核: {rep['_review_items']} 条")


if __name__ == "__main__":
    main()
