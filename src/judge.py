"""LLM-as-Judge 框架（半自动指标用）。

覆盖：GEN-01(0/1/2)、GEN-03、GEN-04、GEN-06、GEN-08、RAG-06、RAG-07、TRACE-02。
默认 mode=off：不调模型，只输出待复核队列（review_queue.jsonl），由人工按
judge_prompts 逐条打分后回填，再跑 report.py 聚合。

启用：config judge.mode=openai_compatible（需 OpenAI 兼容 endpoint，如本地 vLLM/Ollama）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

STEP_TYPES_FOR_GEN08 = ("procedure", "accident", "accident_action",
                        "accident_cause", "accident_symptom")

JUDGE_TASKS = {
    "GEN-01": "答案准确率：对照 golden_answer + expected_answer_points，0=错误/关键事实错，1=基本正确但缺次要信息，2=正确完整。2分计正确。",
    "GEN-03": "答案完整率：expected_answer_points 是否全部完整表达（含条件/例外）。",
    "GEN-04": "幻觉率：答案中每条事实是否有检索证据或 golden 支持；无支持即幻觉。",
    "GEN-06": "错误事实率：答案中是否有与 golden/证据矛盾的事实。",
    "GEN-08": "操作步骤正确率：顺序、前置条件、状态确认是否正确（procedure/accident 题）。",
    "RAG-06": "引用准确率：每条引用是否真正支持对应结论（抽检）。",
    "RAG-07": "证据覆盖率：关键结论是否有有效源文件/条款支撑。",
    "TRACE-02": "证据可追溯率：关键结论能否追溯到具体文档/片段。",
}

PROMPT_TEMPLATE = """你是 RAG 问答质量评审员。只依据【证据】和【参考答案】评审，不引入外部知识。

【问题】{question}
【系统答案】{answer}
【参考答案】{golden}
【必须覆盖事实】{required}
【禁止事项】{forbidden}
【检索证据】{evidence}

 评审任务：{task_desc}
仅输出一行 JSON，不要输出任何解释、前言或代码块标记。
输出 JSON：{{"score": <按任务标度打分>, "reason": "<20字内理由>", "evidence_idx": [<支撑证据编号>]}}"""


def build_prompt(question: str, answer: str, golden: str, required: List[str],
                 forbidden: List[str], evidence: str, task: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=question, answer=answer, golden=golden,
        required=json.dumps(required, ensure_ascii=False),
        forbidden=json.dumps(forbidden, ensure_ascii=False),
        evidence=evidence[:6000], task_desc=JUDGE_TASKS[task])


def build_review_queue(items: List[Dict[str, Any]],
                       tasks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """为每道题生成待复核条目（离线人工/LLM 复核共用同一队列格式）。"""
    tasks = tasks or list(JUDGE_TASKS)
    queue = []
    for it in items:
        for t in tasks:
            if t in ("GEN-08",) and it.get("question_type") not in STEP_TYPES_FOR_GEN08:
                continue
            queue.append({
                "question_id": it.get("question_id"),
                "judge_task": t,
                "question": it.get("question"),
                "answer": it.get("predicted_answer", ""),
                "golden_answer": it.get("golden_answer", ""),
                "required_facts": it.get("expected_answer_points") or it.get("required_facts", []),
                "forbidden_facts": it.get("forbidden_claims") or it.get("forbidden_facts", []),
                "evidence_snippet": (it.get("evidence_snippet", "") or "")[:2000],
                "prompt": build_prompt(
                    it.get("question", ""), it.get("predicted_answer", ""),
                    it.get("golden_answer", ""),
                    it.get("expected_answer_points") or it.get("required_facts", []),
                    it.get("forbidden_claims") or it.get("forbidden_facts", []),
                    it.get("evidence_snippet", ""), t),
                "score": None,
            })
    return queue


def call_openai_compatible(base_url: str, api_key: str, model: str,
                           prompt: str, timeout_s: int = 120,
                           response_format: str = "") -> Dict[str, Any]:
    import requests  # 延迟导入：off 模式下无依赖
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0}
    if response_format:  # 如 "json_object"（Ollama/vLLM 支持，强制 JSON 输出）
        body["response_format"] = {"type": response_format}
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=body, timeout=timeout_s)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return parse_judge_json(content)


def parse_judge_json(content: str) -> Dict[str, Any]:
    """从模型输出中提取打分 JSON：去代码围栏后取首尾花括号段，
    失败则正则兜底抓 score，仍无则返回无分（带原文备查）。"""
    import re
    text = re.sub(r"```(?:json)?", "", content or "").strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except ValueError:
        pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text)
    if m:
        return {"score": float(m.group(1)), "reason": text[:200],
                "partial": True}
    return {"score": None, "reason": text[:200], "raw": text[:1000]}
