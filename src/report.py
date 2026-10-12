"""聚合 per-question 结果 → 指标总览 → 阶段门判定（只判有数据的指标）。

聚合口径（与 xlsx 一致）：
- RAG-01/02/03/09/10、GEN-05/09/10：命中题数/适用题总数（None 不计入分母）
- RAG-04/05：适用题均值；GEN-02：事实级（sum covered / sum total）+ 题均双口径
- GEN-04/06：规则版只能给“疑似率”（forbidden/矛盾初筛），真值待复核 → 标半自动
- 一票否决：forbidden_fact_hit>0 / 高风险题疑似幻觉 → veto 告警，不平均稀释
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .metrics_generation import latency_stats
from .metrics_retrieval import paraphrase_group_recall
from .thresholds import STAGE_GATES, THRESHOLDS, VETO, check


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(rows: List[Dict[str, Any]], key: str) -> tuple[Optional[float], int, int]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if not vals:
        return None, 0, 0
    return sum(vals) / len(vals), len(vals), sum(1 for v in vals if v == 1)


def build_report(rows: List[Dict[str, Any]], lat: Dict[str, Any],
                 retrieval_only: bool,
                 dataset_info: Optional[Dict[str, Any]] = None,
                 retrieval_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    answerable = [r for r in rows if r.get("answerable", True)]
    m: Dict[str, Any] = {}
    # RAG（全自动）
    for k, mid in [("Top@5", "RAG-01"), ("Top@10", "RAG-02"), ("Top@20", "RAG-03")]:
        v, n, _ = _rate(answerable, k)
        m[mid] = {"value": v, "n": n}
    m["RAG-04"] = {"value": _mean([r.get("MRR") for r in answerable
                                   if r.get("MRR") is not None]),
                   "n": sum(1 for r in answerable if r.get("MRR") is not None)}
    m["RAG-05"] = {"value": _mean([r.get("NDCG@10") for r in answerable
                                   if r.get("NDCG@10") is not None]),
                   "n": sum(1 for r in answerable if r.get("NDCG@10") is not None)}
    v, n, _ = _rate(rows, "attachment_recall")
    m["RAG-09"] = {"value": v, "n": n}
    v, n, _ = _rate(rows, "cross_doc_complete")
    m["RAG-10"] = {"value": v, "n": n}
    for key, mid in [("evidence_recall@5", "RAG-evidence@5"),
                     ("evidence_recall@10", "RAG-evidence@10")]:
        v, n, _ = _rate(answerable, key)
        if n:
            m[mid] = {"value": v, "n": n,
                      "note": "内容口径规则版：证据分句在TopK内容中的覆盖率；"
                              "ID映射完成后以RAG-01/02为准"}

    if not retrieval_only:
        # GEN 规则版（全自动部分）
        v, n, _ = _rate(rows, "abstention_correct")
        m["GEN-05"] = {"value": v, "n": n}
        forb = [r for r in rows if r.get("forbidden_fact_hit") is not None]
        hits = sum(1 for r in forb if r.get("forbidden_fact_hit") == 1)
        m["GEN-09-rule"] = {"value": (1 - hits / len(forb)) if forb else None,
                            "n": len(forb), "hits": hits,
                            "note": "规则版：未出现禁止事项的题比例；语义复核见 review_queue"}
        v, n, _ = _rate(rows, "citation_complete_rule")
        m["GEN-10-rule"] = {"value": v, "n": n, "note": "规则版：引用标记存在性"}
        # GEN-02 事实级 + 题均
        tot_c, tot_t, per_q = 0, 0, []
        for r in rows:
            fr = r.get("fact_recall_rule")
            if fr is None:
                continue
            per_q.append(fr)
            # 近似还原计数：fact_recall_rule 是比例，用 expected_answer_points 长度还原
            n_req = len(r.get("expected_answer_points") or r.get("required_facts") or [])
            tot_t += n_req
            tot_c += round(fr * n_req)
        m["GEN-02-fact"] = {"value": (tot_c / tot_t) if tot_t else None,
                            "covered": tot_c, "total": tot_t}
        m["GEN-02-题均"] = {"value": _mean(per_q), "n": len(per_q)}
        # TRACE-01（本轮调用层面）
        tr = [r for r in rows if "trace_ok" in r]
        m["TRACE-01-call"] = {"value": (sum(1 for r in tr if r.get("trace_ok")) / len(tr))
                              if tr else None, "n": len(tr),
                              "note": "调用字段完整率；全链路串联需 OpenTelemetry/Langfuse 另行核查"}
        # 一票否决扫描
        vetos = []
        for r in rows:
            if r.get("forbidden_fact_hit") == 1:
                vetos.append({"question_id": r["question_id"],
                              "veto": "GEN-09 疑似触发禁止事项"})
            if r.get("risk_level") == "critical" and r.get("abstention_correct") == 0:
                vetos.append({"question_id": r["question_id"],
                              "veto": "高风险题拒答/转人工疑似失败"})
        m["_vetos"] = vetos
        # 要点化/顺序/单位（§7.1 规则版）
        for key, mid in [("order_correct_rule", "GEN-ORDER"),
                         ("unit_complete_rule", "GEN-UNIT")]:
            v, n, _ = _rate(rows, key)
            if n:
                m[mid + "-rule"] = {"value": v, "n": n}
        # 拒答分原因统计（§7.2 Table7）
        ref_rows = [r for r in rows if r.get("should_refuse") or r.get("must_abstain")
                    or r.get("refusal_reason")]
        if ref_rows:
            by_reason: dict = {}
            for r in ref_rows:
                by_reason.setdefault(str(r.get("refusal_reason") or "?"), []).append(r)
            m["REFUSAL-by-reason"] = {
                k: {"n": len(v), "abstention_correct": _rate(v, "abstention_correct")[0]}
                for k, v in by_reason.items()}
        # 失败归因分布（§12.1 八类）
        attr: dict = {}
        for r in rows:
            a = r.get("failure_attribution")
            if a:
                attr[a] = attr.get(a, 0) + 1
        if attr:
            m["_attribution"] = attr
        # 改写组召回（§8.2-10：同组多种表达按组统计，避免虚增覆盖）
        grp = paraphrase_group_recall(rows)
        if grp:
            m["paraphrase_group_recall"] = {
                "n_groups": len(grp),
                "mean": sum(grp.values()) / len(grp),
                "min_group": min(grp, key=lambda k: grp[k]),
                "min_value": min(grp.values())}
        # 审核态分布（§10 Table9）
        if any("review_status" in r for r in rows):
            st: dict = {}
            for r in rows:
                st[str(r.get("review_status", "?"))] = st.get(
                    str(r.get("review_status", "?")), 0) + 1
            m["_review_status"] = st
        # 重排 Top3（测试规则：正确依据排在干扰前列；只统计 rerank 环节已召回题）
        rer = [r for r in rows if r.get("test_stage") == "rerank"
               and isinstance(r.get("hit_rank"), int)]
        if rer:
            m["RER-Top3"] = {"value": sum(1 for r in rer if r["hit_rank"] <= 3) / len(rer),
                             "n": len(rer),
                             "note": "测试规则口径：正确依据进前3；未召回题不计"}
        # 场景闭环汇总（同一 scenario_id 跨环节归组；任一环节失败即该场景修复优先）
        veto_ids = {v["question_id"] for v in m.get("_vetos", [])}
        scn: dict = {}
        for r in rows:
            sid = r.get("scenario_id") or ""
            if not sid:
                continue
            s = scn.setdefault(sid, {"n": 0, "stages": set(), "failed": set()})
            s["n"] += 1
            s["stages"].add(str(r.get("test_stage", "?")))
            if (r.get("question_id") in veto_ids or r.get("failure_attribution")
                    or r.get("abstention_correct") == 0):
                s["failed"].add(r["question_id"])
        if scn:
            m["_scenarios"] = {
                sid: {"n": v["n"], "stages": sorted(v["stages"]),
                      "pass": not v["failed"], "failed": sorted(v["failed"])}
                for sid, v in sorted(scn.items())}

    # 阈值判定（只判有值项）
    verdicts = {}
    key_map = {"RAG-01": "RAG-01", "RAG-02": "RAG-02", "RAG-03": "RAG-03",
               "RAG-04": "RAG-04", "RAG-05": "RAG-05", "RAG-09": "RAG-09",
               "RAG-10": "RAG-10", "GEN-05": "GEN-05"}
    for k, mid in key_map.items():
        if mid in m and m[mid].get("value") is not None:
            ok, desc = check(mid, m[mid]["value"])
            verdicts[mid] = {"pass": ok, "desc": desc, "n": m[mid].get("n")}
    if "GEN-02-fact" in m and m["GEN-02-fact"]["value"] is not None:
        ok, desc = check("GEN-02", m["GEN-02-fact"]["value"])
        verdicts["GEN-02(fact规则版)"] = {"pass": ok, "desc": desc}
    if "GEN-10-rule" in m and m["GEN-10-rule"]["value"] is not None:
        ok, desc = check("GEN-10", m["GEN-10-rule"]["value"])
        verdicts["GEN-10(规则版)"] = {"pass": ok, "desc": desc}
    if "GEN-ORDER-rule" in m and m["GEN-ORDER-rule"]["value"] is not None:
        ok, desc = check("GEN-ORDER", m["GEN-ORDER-rule"]["value"])
        verdicts["GEN-ORDER(规则版)"] = {"pass": ok, "desc": desc}
    if "GEN-UNIT-rule" in m and m["GEN-UNIT-rule"]["value"] is not None:
        ok, desc = check("GEN-UNIT", m["GEN-UNIT-rule"]["value"])
        verdicts["GEN-UNIT(规则版)"] = {"pass": ok, "desc": desc}

    # 阶段门：stage1（检索基线）可直接判定；其余阶段列出需补的证据
    stage1 = {}
    for mid in STAGE_GATES["stage1_rag"]:
        if mid in verdicts:
            stage1[mid] = verdicts[mid]
        else:
            stage1[mid] = {"pass": None, "desc": "需半自动复核/人工确认",
                           "n": 0}
    return {
        "mode": "retrieval_only" if retrieval_only else "full",
        "n_questions": len(rows),
        "dataset": dataset_info,
        "retrieval": retrieval_config,
        "metrics": m,
        "verdicts": verdicts,
        "stage1_rag_gate": stage1,
        "latency": lat,
        "veto_count": len(m.get("_vetos", [])),
        "note": "规则版结论可直接用于 RAG-01~05/09/10、GEN-02/05/09/10 初筛；顺序/单位为规则版；"
                "GEN-01/03/04/06/08、RAG-06/07/08 以 review_queue 复核为准；"
                "失败归因见 _attribution（§12.1 八类）；"
                "SEC/OPS/DATA/AGENT/SAFE 见平台核查清单；解析层见 parse_summary.json（§5）。",
    }


def render_markdown(rep: Dict[str, Any]) -> str:
    L = [f"# RAGFlow 自动化测试报告（{rep['mode']}，n={rep['n_questions']}）", ""]
    if rep.get("dataset"):
        d = rep["dataset"]
        L.append(f"数据集: {d.get('path')}（sha256={str(d.get('sha256'))[:12]}…，"
                 f"n={d.get('n_items')}）")
        L.append("")
    L.append("## 指标结果（全自动部分）")
    for k, v in rep["metrics"].items():
        if k.startswith("_"):
            continue
        L.append(f"- {k}: {v}")
    L.append("")
    L.append("## 阈值判定")
    for k, v in rep["verdicts"].items():
        icon = "✅" if v["pass"] else ("❌" if v["pass"] is False else "⬜")
        L.append(f"- {icon} {k}: {v['desc']} (n={v.get('n')})")
    L.append("")
    L.append("## 阶段1 RAG离线基准门")
    for k, v in rep["stage1_rag_gate"].items():
        icon = "✅" if v["pass"] else ("❌" if v["pass"] is False else "⬜")
        L.append(f"- {icon} {k}: {v['desc']}")
    if rep.get("veto_count"):
        L.append("")
        L.append(f"## ⚠️ 一票否决疑似 {rep['veto_count']} 项（须人工确认）")
        for x in rep["metrics"].get("_vetos", []):
            L.append(f"- {x['question_id']}: {x['veto']}")
    scns = rep["metrics"].get("_scenarios")
    if scns:
        n_fail = sum(1 for v in scns.values() if not v["pass"])
        L.append("")
        L.append(f"## 场景闭环（{len(scns)} 个场景，{n_fail} 个有环节失败）")
        for sid, v in scns.items():
            icon = "✅" if v["pass"] else "❌"
            L.append(f"- {icon} {sid}: n={v['n']} 环节={','.join(v['stages'])}"
                     + ("" if v["pass"] else f" 失败={','.join(v['failed'])}"))
    L.append("")
    L.append(f"## 延迟 P50/P95\n- {rep['latency']}")
    L.append("")
    L.append("## 说明\n- " + rep["note"])
    return "\n".join(L) + "\n"
