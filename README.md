# RAGFlow 自动化测试（对齐《智能体完整测试指标体系.xlsx》43 指标）

本地 RAGFlow 问答系统的自动化测试方案，**优先交付可全自动计算的指标**，
半自动/人工项以复核队列 + 核查清单承接，不伪造分数。

## 指标覆盖

详见 [docs/METRICS_COVERAGE.md](docs/METRICS_COVERAGE.md)。

- ✅ 全自动（已实现）：RAG-01/02/03/04/05/09/10、GEN-02/05/09/10（规则版）、
  OPS-03/04（P50/P95）、TRACE-01（调用层）、DATA-02（元数据扫描）
- 🟡 半自动：GEN-01/03/04/06/07/08、RAG-06/07/08、TRACE-02、SEC-02/03 切片、
  DATA-03、AGENT-01/02、SAFE-01 计数 → `review_queue.jsonl` 待人工/LLM 复核
- ⬜ 人工·演练·监控：SEC-01/04/05/06、OPS-01/02/05/06、COST-01、USER-01、
  DATA-01、AGENT-03 → `src/platform_checks.py` 核查清单

## 安装

```bash
pip install -e .   # 或 pip install requests pyyaml
```

## 配置

```bash
cp config/test.yaml.example config/test.yaml
# 编辑 config/test.yaml，至少填写：
#   ragflow.base_url / api_key / dataset_ids / chat_id
# 也可用环境变量 RAGFLOW_API_KEY / RAGFLOW_BASE_URL 覆盖敏感项
```

 golden 测试集按 xlsx「测试集字段映射」+《企业RAG测试集构建要求》Table9 组织，模板见
`datasets/golden_sample.jsonl`（8 道：parameter/procedure/
版本冲突/multi_document/附件依赖/negative + 600MW 表格 PAR-TABLE-001/模糊拒答）。
字段规范见 [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md)，七层结构说明见 [docs/TESTSET_BUILD.md](docs/TESTSET_BUILD.md)。
先标注 10~20 道人工确认的黄金题，验证通过后按 §11 配额先扩到 120 条核心集，再到 300~500。

## 运行

```bash
python scripts/validate_dataset.py datasets/golden_sample.jsonl
python scripts/coverage_check.py datasets/golden_sample.jsonl  # §11 配额/章节/表格占比缺口
python scripts/extract_docx_baseline.py --docx 600MW规程.docx --out datasets/parse_baseline.jsonl  # §5.1 基线
python scripts/parse_benchmark.py --baseline datasets/parse_baseline.jsonl --chunks-file /tmp/chunks.jsonl  # §5 解析比对
python scripts/health_check.py --config config/test.yaml
python scripts/retrieval_benchmark.py --config config/test.yaml  # 阶段1：RAG 离线基准
python scripts/run_all.py --config config/test.yaml              # 全链路 + 阶段验收报告
```

单项运行：

```bash
python scripts/health_check.py --config config/test.yaml
python scripts/retrieval_benchmark.py --config config/test.yaml
python scripts/generation_benchmark.py --config config/test.yaml
```

产物（`results/<experiment_id>/`）：`per_question.jsonl`（对齐 xlsx「测试结果模板」列）、
`summary.json`、`report.md`（阶段1门判定 + 一票否决扫描）、
`review_queue.jsonl` + `review_prompts.jsonl`（半自动复核）。

## 验证（无需 RAGFlow 服务）

```bash
python -m pytest tests/ -q
```

## 注意

正式投入测试前，应先用 10~20 道人工确认过的黄金问题验证
`document_ids` / `chunk_ids` / `required_facts` / `forbidden_facts` 标注，
再扩展到 500~1000 题。
