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
    "GEN-01": "答案准确率：对照参考答案与必须覆盖事实。0=错误/关键事实错，"
              "1=基本正确但缺次要信息，2=正确完整。只取0/1/2，2分计正确。",
    "GEN-03": "答案完整率：必须覆盖事实是否全部完整表达（含条件/例外）。"
              "0=缺关键内容，1=基本完整，2=完整。只取0/1/2。",
    "GEN-04": "幻觉率：答案中是否有检索证据或参考答案不支持的事实。"
              "0=无幻觉，1=有幻觉。只取0/1，越小越好。",
    "GEN-06": "错误事实率：答案中是否有与参考答案/证据矛盾的事实。"
              "0=无矛盾，1=有矛盾。只取0/1，越小越好。",
    "GEN-08": "操作步骤正确率：顺序、前置条件、状态确认是否正确。"
              "0=错误，1=基本正确，2=正确。只取0/1/2。",
    "RAG-06": "引用准确率：每条引用是否真正支持对应结论。"
              "0=不支持，1=支持。只取0/1。",
    "RAG-07": "证据覆盖率：关键结论是否有有效源文件/条款支撑。"
              "0=无支撑，1=有支撑。只取0/1。",
    "TRACE-02": "证据可追溯率：关键结论能否追溯到具体文档/片段。"
                "0=不能，1=能。只取0/1。",
}

# 任务 -> 分数方向（聚合时用：higher 为越高越好，lower 为越小越好）
SCORE_DIRECTIONS = {"GEN-01": "higher", "GEN-03": "higher", "GEN-04": "lower",
                    "GEN-06": "lower", "GEN-08": "higher", "RAG-06": "higher",
                    "RAG-07": "higher", "TRACE-02": "higher"}

REFUSAL_LABELS = {"no_answer": "库内无答案", "insufficient": "依据不足",
                  "ambiguous": "问题含糊", "false_premise": "错误前提",
                  "stale_version": "版本失效", "no_permission": "权限不足"}

PROMPT_TEMPLATE = """你是 RAG 问答质量评审员。只依据【证据】和【参考答案】评审，不引入外部知识。

【问题】{question}
【系统答案】{answer}
【参考答案】{golden}
【必须覆盖事实】{required}
【禁止事项】{forbidden}
【检索证据】{evidence}
【拒答要求】{refusal}
{refusal_rule}
评分铁律：
1. 表述差异不算错：数值与单位写法（MPa 与 兆帕、有无空格）、语序、同义词、是否复述题干中的部件名称，均不扣分。
2. 逐条核对【必须覆盖事实】与【参考答案】：缺失次要条目的最高评 1 分；只有关键数值错误、与参考答案矛盾、答非所问，或应拒答却直接作答，才评 0 分。
3. 【证据】能支撑但【参考答案】未写出的内容，不判为幻觉，也不据此扣分。
 评审任务：{task_desc}
仅输出一行 JSON，不要输出任何解释、前言或代码块标记。
输出 JSON：{{"score": <按任务标度打分>, "reason": "<20字内理由>", "evidence_idx": [<支撑证据编号>]}}"""


def refusal_block(should_refuse: bool, refusal_reason: str) -> tuple[str, str]:
    if not should_refuse:
        return "无（应正常回答）", ""
    label = REFUSAL_LABELS.get(refusal_reason, refusal_reason or "应拒答")
    return (f"应拒答（{label}）",
            "拒答题判定铁律：系统若直接给出操作/参数答案而未拒答、未要求补充信息，"
            "则 GEN-01/03/08 一律 0 分，GEN-04/06 按有幻觉/矛盾记 1 分；"
            "只有正确拒答或要求补充关键信息才算正确。")


def build_prompt(question: str, answer: str, golden: str, required: List[str],
                 forbidden: List[str], evidence: str, task: str,
                 should_refuse: bool = False, refusal_reason: str = "") -> str:
    refusal, rule = refusal_block(bool(should_refuse), str(refusal_reason or ""))
    return PROMPT_TEMPLATE.format(
        question=question, answer=answer, golden=golden,
        required=json.dumps(required, ensure_ascii=False),
        forbidden=json.dumps(forbidden, ensure_ascii=False),
        evidence=evidence[:6000], refusal=refusal, refusal_rule=rule,
        task_desc=JUDGE_TASKS[task])


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
                    it.get("evidence_snippet", ""), t,
                    bool(it.get("should_refuse") or it.get("must_abstain", False)
                         or it.get("answerable") is False or it.get("refusal_reason")),
                    str(it.get("refusal_reason", ""))),
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
