"""全环节一键流水线：转换 → 校验 → 解析 → 映射 → 分环节执行 → 复核 → 写回。

问答侧 5 个环节（召回/重排/回答/引用/拒答）本就由 run_all 一次跑完；
本脚本只是把有依赖关系的旁路串起来，任何一步缺条件就跳过并说明，
不中断后续步骤。

用法：
    python scripts/run_pipeline.py --config config/test.yaml \\
        --xlsx 企业RAG全环节测试集_600MW.xlsx --batch EXP-001
常用开关：
    --stages retrieval rerank qa refusal   只跑子集（默认全环节）
    --skip-import / --skip-parse / --skip-mapping / --skip-export  跳过某步
    --reimport                             强制重转（会丢失已应用的 Chunk 映射）
    --auto-map [--map-threshold 0.8]       证据对齐自动映射后应用
    --apply-mapping mapping.jsonl          应用已有映射文件后继续
    --judge [--judge-tasks GEN-01 GEN-04]  全链路后跑 LLM 复核
    --dataset-id KB --document-id DOC      解析步骤覆盖配置
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
PY = sys.executable

from src.suite import exp_dir_from_config, record  # noqa: E402

EXP_DIR = ""  # main() 按 config 的 output_dir/experiment_id 设置


def mark(stage: str, status: str, elapsed: float, note: str = "",
         artifact: str = "") -> None:
    """把阶段结果写进 results/<exp>/pipeline_stages.jsonl，供 S6 汇总。"""
    if EXP_DIR:
        record(EXP_DIR, stage, status, elapsed, artifact=artifact, note=note)


def step(stage: str, title: str, *cmd: str) -> bool:
    t0 = time.time()
    ok = sh(title, *cmd)
    mark(stage, "ok" if ok else "fail", time.time() - t0, title)
    return ok


def step_capture(stage: str, title: str, *cmd: str) -> tuple:
    t0 = time.time()
    ok, out = sh_capture(title, *cmd)
    mark(stage, "ok" if ok else "fail", time.time() - t0, title)
    return ok, out


def skip(stage: str, note: str) -> None:
    mark(stage, "skip", 0.0, note)


def sh(step: str, *cmd: str) -> bool:
    print(f"\n===== {step} =====")
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"----- {step}: {'OK' if r.returncode == 0 else 'FAIL(' + str(r.returncode) + ')'} -----")
    return r.returncode == 0


def sh_capture(step: str, *cmd: str) -> tuple[bool, str]:
    print(f"\n===== {step} =====")
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:])
    print(f"----- {step}: {'OK' if r.returncode == 0 else 'FAIL(' + str(r.returncode) + ')'} -----")
    return r.returncode == 0, r.stdout


def report_dir_from_output(output: str) -> str:
    import re
    m = re.search(r"报告目录:\s*(\S+)", output)
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--xlsx", default="企业RAG全环节测试集_600MW.xlsx")
    ap.add_argument("--golden", default="datasets/xlsx_golden.jsonl")
    ap.add_argument("--baseline", default="datasets/xlsx_parse_baseline.jsonl")
    ap.add_argument("--batch", default="EXP-001")
    ap.add_argument("--tester", default="")
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--dataset-id", default="")
    ap.add_argument("--document-id", default="")
    ap.add_argument("--apply-mapping", default="")
    ap.add_argument("--auto-map", action="store_true")
    ap.add_argument("--map-threshold", type=float, default=0.8)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-tasks", nargs="*", default=[])
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--reimport", action="store_true")
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument("--skip-mapping", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    global EXP_DIR
    import yaml  # noqa: E402
    with open(args.config, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}
    EXP_DIR = exp_dir_from_config(_cfg)
    print(f"运行目录：{EXP_DIR}")

    golden = args.golden
    baseline = args.baseline
    ok = True

    if args.skip_import:
        skip("S0", "指定 --skip-import")
    elif args.xlsx and os.path.exists(args.xlsx):
        fresh = (not os.path.exists(golden) or not os.path.exists(baseline)
                 or os.path.getmtime(args.xlsx) > min(os.path.getmtime(golden),
                                                      os.path.getmtime(baseline)))
        if args.reimport or fresh:
            if not step("S0", "00. 转换xlsx", PY, "scripts/import_xlsx_cases.py",
                        "--xlsx", args.xlsx, "--out-golden", golden,
                        "--out-baseline", baseline):
                return 1
        else:
            skip("S0", "转换产物比 xlsx 新，跳过（--reimport 强制重转）")
            print("转换产物比 xlsx 新，跳过转换（加 --reimport 强制重转；"
                  "注意重转会丢失已应用的 Chunk 映射）。")

    if not step("S0", "0. 校验问答集", PY, "scripts/validate_dataset.py", golden):
        return 1
    step("S0", "0b. 覆盖率", PY, "scripts/coverage_check.py", golden)

    if args.skip_parse:
        skip("S1", "指定 --skip-parse")
    else:
        cmd = [PY, "scripts/parse_benchmark.py", "--baseline", baseline,
               "--config", args.config]
        if args.dataset_id:
            cmd += ["--dataset-id", args.dataset_id]
        if args.document_id:
            cmd += ["--document-id", args.document_id]
        if not step("S1", "1. 解析+切分", *cmd):
            print("提示：解析需 RAGFlow 已入库且配置 document-id，跳过继续。")
    if args.skip_mapping:
        skip("S2", "指定 --skip-mapping")
    else:
        apply_path = args.apply_mapping
        if args.auto_map:
            apply_path = apply_path or os.path.join(
                "/tmp", f"mapping_auto_{args.batch}.jsonl")
            if not step("S2", "2a. 证据对齐自动映射", PY,
                        "scripts/assist_chunk_mapping.py",
                        "--auto-map", apply_path, "--golden", golden,
                        "--config", args.config, "--threshold", str(args.map_threshold)):
                return 1
        if apply_path:
            if not step("S2", "2. 应用Chunk映射", PY,
                        "scripts/assist_chunk_mapping.py",
                        "--apply", apply_path,
                        "--in-golden", golden, "--out-golden", golden):
                return 1
        else:
            step("S2", "2. 生成Chunk候选（人工确认后下次加 --apply-mapping）",
                 PY, "scripts/assist_chunk_mapping.py", "--golden", golden,
                 "--config", args.config, "--out", "/tmp/chunk_candidates.jsonl")
    if not step("S3", "3. 健康检查", PY, "scripts/health_check.py",
                "--config", args.config):
        print("服务不通，后续在线步骤将失败，如需只做离线校验请 Ctrl-C。")

    cfg = dict(_cfg)
    cfg.setdefault("run", {})["dataset"] = golden
    run_cfg = args.config
    if args.stages:
        cfg["run"]["stages"] = args.stages
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                      encoding="utf-8")
    yaml.safe_dump(cfg, tmp, allow_unicode=True)
    tmp.close()
    run_cfg = tmp.name
    print(f"本次问答集：{golden}"
          + (f"，环节子集：{args.stages}" if args.stages else "（全环节）"))
    run_ok, run_out = step_capture("S4", "4. 全链路执行", PY, "scripts/run_all.py",
                                   "--config", run_cfg)
    if not run_ok:
        ok = False
        print("第 4 步失败：跳过写回，避免旧数据套新批次号；修好服务/配置后重跑。")
    try:
        os.unlink(run_cfg)
    except OSError:
        pass

    out_dir = report_dir_from_output(run_out)
    if not args.judge:
        skip("S5", "未指定 --judge")
    elif out_dir and os.path.exists(
            os.path.join(ROOT, out_dir, "review_prompts.jsonl")):
        cmd = [PY, "scripts/judge_review.py", "--dir", out_dir,
               "--config", args.config]
        if args.judge_tasks:
            cmd += ["--tasks"] + args.judge_tasks
        if not step("S5", "4b. LLM 复核", *cmd):
            print("提示：复核失败不影响已有规则分；修好 judge 配置后可单独重跑本步。")
    else:
        skip("S5", "第 4 步无 review_prompts.jsonl，跳过 LLM 复核")
        print("提示：第 4 步无产物，跳过 LLM 复核。")

    if ok and not args.skip_export and os.path.exists(args.xlsx):
        perq, summ = "", ""
        out_dir = report_dir_from_output(run_out)
        if out_dir and os.path.exists(os.path.join(ROOT, out_dir,
                                                   "per_question.jsonl")):
            perq = os.path.join(out_dir, "per_question.jsonl")
            summ = os.path.join(out_dir, "summary.json")
        else:
            import glob
            outs = sorted(glob.glob("results/*/per_question.jsonl"),
                          key=os.path.getmtime)
            if outs:
                print(f"提示：未定位本次产物目录，回退到最新 {outs[-1]}")
                perq = outs[-1]
                summ = os.path.join(os.path.dirname(perq), "summary.json")
            else:
                print("跳过写回：找不到任何 per_question.jsonl（第 4 步无产物）。")
        if perq:
            step("S6", "5a. 写回业务xlsx", PY, "scripts/export_xlsx_record.py",
                 "--xlsx", args.xlsx, "--per-question", perq,
                 "--summary", summ, "--batch", args.batch,
                 "--tester", args.tester)
        else:
            skip("S6", "无 per_question.jsonl，跳过业务 xlsx 写回")
    elif not ok:
        skip("S6", "第 4 步失败，跳过业务 xlsx 写回")
        print("跳过写回：第 4 步失败（见上），无本次产物可写。")
    elif args.skip_export:
        skip("S6", "指定 --skip-export")
        print("跳过写回：指定了 --skip-export。")
    elif not os.path.exists(args.xlsx):
        skip("S6", "找不到业务 xlsx")
        print(f"跳过写回：找不到 {args.xlsx}。")

    # S6 收口：无论前面成败，只要有产物就汇成一份 suite.xlsx（过程留痕）
    if os.path.isdir(EXP_DIR) and os.path.exists(
            os.path.join(EXP_DIR, "pipeline_stages.jsonl")):
        if step("S6", "6. 汇总结果xlsx", PY, "scripts/export_suite_xlsx.py",
                "--dir", EXP_DIR, "--config", args.config,
                "--batch", args.batch):
            print(f"单一交付物：{os.path.join(EXP_DIR, 'suite.xlsx')}")
    else:
        skip("S6", "无运行产物，跳过 suite.xlsx")

    print("\n流水线结束：" + ("全部成功" if ok else "执行有失败，见上"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
