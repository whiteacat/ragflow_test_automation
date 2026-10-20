# RAGFlow 自动化测试（对齐《智能体完整测试指标体系.xlsx》43 指标）

本地 RAGFlow 问答系统的自动化测试方案，**优先交付可全自动计算的指标**，
半自动/人工项以复核队列 + 核查清单承接，不伪造分数。

## 指标覆盖

详见 [docs/METRICS_COVERAGE.md](docs/METRICS_COVERAGE.md)。

- ✅ 全自动（已实现）：RAG-01/02/03/04/05/09/10、GEN-02/05/09/10（规则版）、
  OPS-03/04（P50/P95）、TRACE-01（调用层）、DATA-02（元数据扫描）
- 🟡 半自动：GEN-01/03/04/06/07/08、RAG-06/07/08、TRACE-02、SEC-02/03 切片、
  DATA-03、AGENT-01/02、SAFE-01 计数 → `review_queue.jsonl` 待人工/LLM 复核；
  DATA-01（解析）→ `parse_benchmark.py` 自动比对 + 人工复核异常清单
- ⬜ 人工·演练·监控：SEC-01/04/05/06、OPS-01/02/05/06、COST-01、USER-01、
  DATA-01 复核结论、AGENT-03 → `src/platform_checks.py` 核查清单

## 安装

```bash
pip install -e .   # 或 pip install requests pyyaml openpyxl
```

## 配置

```bash
cp config/test.yaml.example config/test.yaml
# 编辑 config/test.yaml，至少填写：
#   ragflow.base_url / api_key / dataset_ids / chat_id
# 也可用环境变量 RAGFLOW_API_KEY / RAGFLOW_BASE_URL 覆盖敏感项
```

关键开关（`config/test.yaml`）：

| 位置 | 说明 |
|---|---|
| `run.dataset` | 本次运行的问答数据集（二选一：`datasets/golden_sample.jsonl` 手写样例，或 `datasets/xlsx_golden.jsonl` xlsx 转换集） |
| `run.stages` | 环节子集过滤，`null` 跑全环节；如 `["retrieval", "rerank"]` 只跑召回/重排 |
| `run.retrieval_only` | `true` 只跑检索（RAG 指标，无需 `chat_id`）；`false` 跑全链路 |
| `retrieval.rerank_id` | 重排模型 id，必须用 RAGFlow 的**全限定引用** `模型名@实例名@provider名`（如 `Qwen3-Reranker-8B@default@VLLM`）；没配就 `null` |
| `retrieval.rerank_compare` | `§6.2` 重排 A/B，如 `[null, "Qwen3-Reranker-8B@default@VLLM"]`，必须有一侧等于 `rerank_id` |
| `judge.response_format` | 填 `json_object` 强制模型输出 JSON（Ollama/vLLM，思考模型如 Qwen3 强烈建议开） |
| `judge.timeout / qps` | 复核单条超时（默认 120s）与限速（默认 1 qps），27B 等慢模型可调 |
| `parse.baseline/document_id` | 解析比对的基线文件与 RAGFlow 侧文档 |
| `tier.active` | 测试集分层：`core` 核心金标准 / `extended` 扩展回归 / `adversarial` 对抗 |

### 本地 rerank（vLLM）配置要点

`retrieval.rerank_id` 会原样透传给 RAGFlow 的 `/api/v1/retrieval`，**格式必须与 RAGFlow 注册的一致**：

1. **全限定引用**：`模型名@实例名@provider名`。实例名一般 `default`，provider 是注册时选的厂商（`VLLM` / `OpenAI-API-Compatible` / `Xinference` 等）。
   例：`"Qwen3-Reranker-8B@default@VLLM"`。
2. **只写裸模型名会报错**：`Provider  not found for model <名字>`（provider 段为空）——这不是没注册，是少了 `@实例@provider`。
3. 老版本（Python 后端）用的是 `___VLLM` 后缀，新版不适用；以 UI「Model providers」或数据库里查到的为准：
   ```sql
   SELECT m.model_name, i.instance_name, p.provider_name
   FROM tenant_model m
   JOIN tenant_model_instance i ON i.id = m.instance_id
   JOIN tenant_model_provider p ON p.id = m.provider_id
   WHERE m.model_type LIKE '%rerank%';
   ```
4. **vLLM 侧**：Qwen3-Reranker 必须以 pooling + rerank overrides 启动（`--runner pooling --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'`），不能加 `--is-embedding`，否则 `/v1/rerank` 算不出分数。
5. RAGFlow 跑在 Docker、vLLM 在宿主机时，provider 的 Base URL 用 `host.docker.internal:<端口>`（Linux 用宿主机局域网 IP），不能用 `127.0.0.1`。
6. 验证顺序：curl 直连 `/api/v1/retrieval` 带 `rerank_id` → `qa_single.py --rerank-compare` 看排序是否变化 → `retrieval_benchmark.py` 看 `rerank_gain` / RER-Top3。

## 测试数据：两种输入

- **手写 jsonl**：`datasets/golden_sample.jsonl`（8 道样例），字段规范见
  [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md)，七层结构见 [docs/TESTSET_BUILD.md](docs/TESTSET_BUILD.md)。
  先标注 10~20 道人工确认的黄金题，验证通过后按 §11 配额先扩到 120 条核心集，再到 300~500。
- **xlsx 全环节集**：`企业RAG全环节测试集_600MW.xlsx`（测试用例总表，120 条，30 场景）
  为业务基准源，运行时不直读，需先转换（见流程步骤 0）。xlsx 本体为受控文件，不入库 git。
- **小规模冒烟集**：`datasets/smoke_small.jsonl`（8 条：6 条问答 + 2 条拒答，均无需 Chunk 映射），
  用于改完配置/模型后 1~2 分钟验证全链路是否通。检索的 chunk 级指标（hit@k/MRR/NDCG）不计分母，
  仍可看 evidence_recall、rerank_gain、生成规则分与拒答判定。

```bash
python scripts/run_pipeline.py --config config/test.yaml \
    --golden datasets/smoke_small.jsonl \
    --skip-import --skip-parse --skip-mapping --skip-export --batch SMOKE-8
```

## 完整测试流程

一键流水线（校验 → 解析 → 映射 → 全链路 → 写回，缺条件自动跳过）：

```bash
python scripts/run_pipeline.py --config config/test.yaml \
    --xlsx 企业RAG全环节测试集_600MW.xlsx --batch EXP-001
# 子集：--stages retrieval rerank qa refusal；跳步：--skip-import/--skip-parse/--skip-mapping/--skip-export
# 映射：--auto-map（--map-threshold 0.8）/ --apply-mapping mapping.jsonl；复核：--judge [--judge-tasks GEN-01]
# 换数据集：--golden datasets/golden_sample.jsonl --baseline datasets/parse_golden_sample.jsonl
# 解析覆盖：--dataset-id KB --document-id DOC；强制重转：--reimport（会丢失已应用映射）
```

各步骤也可单独执行：

```bash
# A. 手写 jsonl：直接编辑 datasets/golden_sample.jsonl，旧键可先迁移再补证据链
python scripts/migrate_dataset.py datasets/golden_sample.jsonl --write
# B. xlsx 转换：120 条 → xlsx_golden.jsonl（RET/RER/ANS/CIT/REF 84 条）
#                + xlsx_parse_baseline.jsonl（PAR/CHK 36 条）
python scripts/import_xlsx_cases.py --xlsx 企业RAG全环节测试集_600MW.xlsx
```

### 步骤 1：校验与覆盖率（无需服务）

```bash
python scripts/validate_dataset.py datasets/xlsx_golden.jsonl   # 字段/拒答/行列/顺序/审核校验
python scripts/coverage_check.py datasets/xlsx_golden.jsonl     # §11 配额/章节/表格占比缺口
```

### 步骤 2：原文解析评测（§5）

```bash
# 自建基线（二选一）：从 DOCX 独立提取，或直接用 xlsx 转换的 parse 基线
python scripts/extract_docx_baseline.py --docx 600MW规程.docx --out datasets/parse_baseline.jsonl
# 比对：在线拉取 RAGFlow Chunk，或离线用 --chunks-file
python scripts/parse_benchmark.py --baseline datasets/xlsx_parse_baseline.jsonl \
    --config config/test.yaml --dataset-id <KB_ID> --document-id <DOC_ID>
# 产物 results/<exp>/parse_summary.json（DATA-01 四项）+ parse_anomalies.jsonl（待§5.3人工复核）
```

### 步骤 3：入库映射 Chunk（xlsx 转换集必需，手写集已内置）

```bash
# 1) 对 needs_chunk_mapping=true 的用例跑检索，输出候选清单
python scripts/assist_chunk_mapping.py --golden datasets/xlsx_golden.jsonl \
    --config config/test.yaml --out /tmp/chunk_candidates.jsonl
# 2) 人工确认后写回（mapping.jsonl: question_id/expected_chunk/acceptable_chunks）
python scripts/assist_chunk_mapping.py --apply mapping.jsonl \
    --in-golden datasets/xlsx_golden.jsonl --out-golden datasets/xlsx_golden.jsonl
```

### 步骤 4：健康检查与冒烟

```bash
python scripts/health_check.py --config config/test.yaml
```

### 步骤 5：检索/重排基准（阶段 1，可分环节）

```bash
# run.dataset 指向 xlsx_golden.jsonl，run.stages 取 ["retrieval", "rerank"]，retrieval_only=true
python scripts/retrieval_benchmark.py --config config/test.yaml
```

### 步骤 6：全链路问答（阶段 2，含拒答）

```bash
# run.stages 取 ["qa", "refusal"]（或 null 全量），retrieval_only=false
python scripts/run_all.py --config config/test.yaml
# 单项运行：
python scripts/generation_benchmark.py --config config/test.yaml
```

### 步骤 7：复核与写回

```bash
# 半自动复核：results/<experiment_id>/review_queue.jsonl + review_prompts.jsonl
# LLM 自动复核（需 judge.mode=openai_compatible，支持断点续跑/子集/限量）：
python scripts/judge_review.py --dir results/<experiment_id> --config config/test.yaml
python scripts/judge_review.py --dir results/<experiment_id> --risk critical high --limit 20
python scripts/judge_review.py --dir results/<experiment_id> --no-resume   # 量表/prompt 变更后全量重打
python scripts/judge_review.py --dir results/<experiment_id> --dry-run    # 先验一条 prompt 再开跑
# 写回 xlsx 执行记录（默认写副本，--in-place 覆盖前请备份）：
python scripts/export_xlsx_record.py --xlsx 企业RAG全环节测试集_600MW.xlsx \
    --per-question results/<experiment_id>/per_question.jsonl \
    --summary results/<experiment_id>/summary.json --batch EXP-001 --tester 张三
```

### 辅助脚本（不走流水线，按需单点）

```bash
# 单条问答全流程：检索→重排对比→问答→四项规则分
python scripts/qa_single.py --config config/test.yaml --question "水冷壁超压试验压力是多少？" \
    --expect "23.55MPa" --forbidden "再热器5.7MPa" --evidence "省煤器、水冷壁、过热器 | 23.55MPa" \
    --must-cite --full
# 类 Ragas 四指标（Context Precision/Recall、Faithfulness、Answer Relevancy，规则版下限）
python scripts/ragas_like.py --per-question results/<experiment_id>/per_question.jsonl \
    --golden datasets/xlsx_golden.jsonl --k 5 --out /tmp/ragas.jsonl
```

## 产物（`results/<experiment_id>/`）

`per_question.jsonl`（对齐 xlsx「测试结果模板」列，含失败归因与场景编号）、
`summary.json`、`report.md`（阶段门判定 + 一票否决扫描 + **场景闭环**汇总 + RER-Top3 + 运行异常节）、
`review_queue.jsonl` + `review_prompts.jsonl`（半自动复核）、
`review_scores.jsonl` + `review_queue_scored.jsonl`（LLM 复核逐条分与回填）、
`parse_summary.json` / `parse_anomalies.jsonl`（解析比对与异常清单，`parse_benchmark` 产出）。

## 验证（无需 RAGFlow 服务）

```bash
python -m pytest tests/ -q
```

## 注意

正式投入测试前，应先用 10~20 道人工确认过的黄金问题验证
`document_ids` / `chunk_ids` / `expected_answer_points` / `forbidden_claims` 标注，
再按 §11 配额扩展到 120 条核心集（表格≥20%），最后到 300~500 题。
xlsx 转换集全量题均为 `pending` 待审核，安全关键题须专业人员确认后才进入正式金标准；
RET/RER 题须先完成 Chunk 映射，否则检索指标不计分母。
