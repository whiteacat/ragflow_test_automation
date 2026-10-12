"""全环节一键流水线：校验 → 解析 → 映射 → 分环节执行 → 写回。

问答侧 5 个环节（召回/重排/回答/引用/拒答）本就由 run_all 一次跑完；
本脚本只是把有依赖关系的旁路串起来，任何一步缺条件就跳过并说明，
不中断后续步骤。

用法：
    python scripts/run_pipeline.py --config config/test.yaml \\
        --xlsx 企业RAG全环节测试集_600MW.xlsx --batch EXP-001
常用开关：
    --stages retrieval rerank qa refusal   只跑子集（默认全环节）
    --skip-parse / --skip-mapping / --skip-export   跳过某步
    --apply-mapping mapping.jsonl          自动应用 Chunk 映射后继续
    --dataset-id KB --document-id DOC      解析步骤覆盖配置
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
PY = sys.executable


def sh(step: str, *cmd: str) -> bool:
    print(f"\n===== {step} =====")
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"----- {step}: {'OK' if r.returncode == 0 else 'FAIL(' + str(r.returncode) + ')'} -----")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--xlsx", default="企业RAG全环节测试集_600MW.xlsx")
    ap.add_argument("--batch", default="EXP-001")
    ap.add_argument("--tester", default="")
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--dataset-id", default="")
    ap.add_argument("--document-id", default="")
    ap.add_argument("--apply-mapping", default="")
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument("--skip-mapping", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    golden = "datasets/xlsx_golden.jsonl"
    baseline = "datasets/xlsx_parse_baseline.jsonl"
    ok = True

    if not sh("0. 校验问答集", PY, "scripts/validate_dataset.py", golden):
        return 1
    sh("0b. 覆盖率", PY, "scripts/coverage_check.py", golden)

    if not args.skip_parse:
        cmd = [PY, "scripts/parse_benchmark.py", "--baseline", baseline,
               "--config", args.config]
        if args.dataset_id:
            cmd += ["--dataset-id", args.dataset_id]
        if args.document_id:
            cmd += ["--document-id", args.document_id]
        if not sh("1. 解析+切分", *cmd):
            print("提示：解析需 RAGFlow 已入库且配置 document-id，跳过继续。")
    if not args.skip_mapping:
        if args.apply_mapping:
            if not sh("2. 应用Chunk映射", PY, "scripts/assist_chunk_mapping.py",
                      "--apply", args.apply_mapping,
                      "--in-golden", golden, "--out-golden", golden):
                return 1
        else:
            sh("2. 生成Chunk候选（人工确认后下次加 --apply-mapping）",
               PY, "scripts/assist_chunk_mapping.py", "--golden", golden,
               "--config", args.config, "--out", "/tmp/chunk_candidates.jsonl")
    if not sh("3. 健康检查", PY, "scripts/health_check.py", "--config", args.config):
        print("服务不通，后续在线步骤将失败，如需只做离线校验请 Ctrl-C。")

    env = os.environ.copy()
    run_cfg = args.config
    if args.stages:
        import yaml  # noqa: E402
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("run", {})["stages"] = args.stages
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                          encoding="utf-8")
        yaml.safe_dump(cfg, tmp, allow_unicode=True)
        tmp.close()
        run_cfg = tmp.name
        print(f"本次问答环节子集：{args.stages}（已覆写 run.stages）")
    if not sh("4. 全链路执行", PY, "scripts/run_all.py", "--config", run_cfg):
        ok = False

    if not args.skip_export and os.path.exists(args.xlsx):
        import glob
        outs = sorted(glob.glob("results/*/per_question.jsonl"),
                      key=os.path.getmtime)
        if outs:
            perq = outs[-1]
            summ = os.path.join(os.path.dirname(perq), "summary.json")
            sh("5. 写回执行记录", PY, "scripts/export_xlsx_record.py",
               "--xlsx", args.xlsx, "--per-question", perq,
               "--summary", summ, "--batch", args.batch,
               "--tester", args.tester)
    print("\n流水线结束：" + ("全部成功" if ok else "执行有失败，见上"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
