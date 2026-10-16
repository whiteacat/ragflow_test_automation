"""LLM 复核执行器：消费 review_prompts.jsonl，调模型打分并回填。

只在 judge.mode=openai_compatible 时工作；off 模式直接退出。
特性：失败重试、断点续跑（已打分题默认跳过）、任务/风险子集、限速。
输出：review_scores.jsonl（逐条打分）+ review_queue_scored.jsonl（队列回填）。

用法：
    python scripts/judge_review.py --dir results/exp_xlsx_001 --config config/test.yaml
    python scripts/judge_review.py --dir results/exp_xlsx_001 --tasks GEN-01 GEN-04
    python scripts/judge_review.py --dir results/exp_xlsx_001 --risk critical high --limit 20
    python scripts/judge_review.py --dir results/exp_xlsx_001 --no-resume  # 全量重打
    python scripts/judge_review.py --dir results/exp_xlsx_001 --dry-run    # 只打印首条 prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config  # noqa: E402
from src.judge import call_openai_compatible  # noqa: E402


def load_inputs(d: str):
    prompts = {}
    for line in open(os.path.join(d, "review_prompts.jsonl"), encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            prompts[(p["question_id"], p["judge_task"])] = p["prompt"]
    queue = [json.loads(l) for l in
             open(os.path.join(d, "review_queue.jsonl"), encoding="utf-8")
             if l.strip()]
    perq = {}
    pq_path = os.path.join(d, "per_question.jsonl")
    if os.path.exists(pq_path):
        for line in open(pq_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                perq[r.get("question_id")] = r
    return prompts, queue, perq


def load_done(d: str):
    done = {}
    path = os.path.join(d, "review_scores.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                s = json.loads(line)
                done[(s["question_id"], s["judge_task"])] = s
    return done


def select(queue, perq, tasks, risks):
    out = []
    for q in queue:
        if tasks and q.get("judge_task") not in tasks:
            continue
        if risks:
            row = perq.get(q.get("question_id"), {})
            if str(row.get("risk_level", "")) not in risks:
                continue
        out.append(q)
    return out


def call_with_retry(base_url, api_key, model, prompt, timeout_s, retries,
                    response_format=""):
    last = None
    for i in range(retries + 1):
        try:
            return call_openai_compatible(base_url, api_key, model, prompt,
                                          timeout_s, response_format)
        except Exception as exc:  # noqa: BLE001 — 重试后仍失败则记 error
            last = str(exc)
            time.sleep(min(2 ** i, 30))
    return {"score": None, "reason": f"调用失败: {last}"[:200], "error": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--risk", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--qps", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config).get("judge", {}) or {}
    if cfg.get("mode") != "openai_compatible":
        print("judge.mode 不是 openai_compatible（当前 %r），不执行。"
              % cfg.get("mode"))
        return 2
    base_url, model = cfg.get("base_url"), cfg.get("model")
    if not base_url or not model:
        print("需配置 judge.base_url 与 judge.model")
        return 2
    api_key = cfg.get("api_key") or os.environ.get("JUDGE_API_KEY", "none")
    response_format = str(cfg.get("response_format") or "")

    prompts, queue, perq = load_inputs(args.dir)
    items = select(queue, perq, set(args.tasks), set(args.risk))
    done = {} if args.no_resume else load_done(args.dir)
    todo = [q for q in items
            if (q.get("question_id"), q.get("judge_task")) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"队列 {len(queue)} 条 → 本次 {len(todo)} 条（已完成 {len(done)} 条跳过）")
    if args.dry_run:
        if todo:
            q = todo[0]
            print(prompts.get((q.get("question_id"), q.get("judge_task")),
                              "")[:1500])
        return 0
    if not todo:
        print("无待打分条目。")
        return 0

    out_path = os.path.join(args.dir, "review_scores.jsonl")
    if args.no_resume and os.path.exists(out_path):
        os.remove(out_path)  # 全量重打：旧分量表已失效，清掉防重复
    scored = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for q in todo:
            key = (q.get("question_id"), q.get("judge_task"))
            res = call_with_retry(base_url, api_key, model,
                                  prompts.get(key, ""), args.timeout, args.retries,
                                  response_format)
            rec = {"question_id": q.get("question_id"),
                   "judge_task": q.get("judge_task"),
                   "model": model, **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[key] = rec
            scored += 1
            if scored % 10 == 0:
                print(f"已打分 {scored}/{len(todo)}")
            if args.qps > 0:
                time.sleep(1.0 / args.qps)

    merged = []
    for q in queue:
        key = (q.get("question_id"), q.get("judge_task"))
        q2 = dict(q)
        if key in done:
            for k in ("score", "reason", "evidence_idx", "model", "error"):
                if done[key].get(k) is not None:
                    q2[k] = done[key][k]
        merged.append(q2)
    with open(os.path.join(args.dir, "review_queue_scored.jsonl"), "w",
              encoding="utf-8") as f:
        for q in merged:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    ok = sum(1 for s in done.values() if s.get("score") is not None)
    print(f"完成：{ok}/{len(done)} 条有分 → review_scores.jsonl + review_queue_scored.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
