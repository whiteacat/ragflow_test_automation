"""健康检查：连通 + 鉴权 + 最小检索/问答冒烟。"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config  # noqa: E402
from src.ragflow_client import RAGFlowClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rag = cfg["ragflow"]
    client = RAGFlowClient(rag["base_url"], rag["api_key"])
    h = client.health()
    print("health:", h["http_status"], "ok=",
          h["ok"])
    if not h["ok"]:
        print(h["body"])
        sys.exit(1)
    ds = rag.get("dataset_ids") or []
    if ds and ds[0] != "CHANGE_ME":
        ret = client.retrieve("冒烟测试：受电弓额定电压是多少？", ds, page_size=5)
        print(f"retrieval ok: {len(ret['chunks'])} chunks, "
              f"{ret['latency_ms']}ms")
        if ret["chunks"]:
            c = ret["chunks"][0]
            print(f"  top1: {c.document_name}#{c.chunk_id} "
                  f"score={c.score:.3f}")
    else:
        print("跳过检索冒烟：请先填写 dataset_ids")
    if rag.get("chat_id") and rag["chat_id"] != "CHANGE_ME":
        qa = client.chat(rag["chat_id"], "你好，请简述你的知识范围。")
        print("chat ok:", (qa.answer or "")[:120], "err=", qa.error or "无")
    else:
        print("跳过问答冒烟：请先填写 chat_id")


if __name__ == "__main__":
    main()
