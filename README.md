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
| `config/synonyms.yaml` | 术语等价表（GEN-02 标签匹配用），保存后 1 秒内热加载；只收录**经确认的同义**术语 |
| `embedding.enabled` | 可选语义兜底（本地 bge-m3），默认 `false`；`interface` 选 `openai`（`/v1/embeddings`）或 `tei`（`/embed`），填好 `base_url`/`model`/`dim` 即生效，关闭时判定与未引入 embedding 完全一致 |

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

**推荐：xlsx 人工勾选**（拿到真实检索结果后人工判定，最可靠）

```bash
# 1) 对未映射的题跑检索，导出候选清单 xlsx（一行一个候选；可传多个数据集）
python scripts/export_chunk_candidates_xlsx.py --config config/test.yaml \
    --golden datasets/xlsx_golden.jsonl datasets/smoke_small.jsonl \
    --out chunk_review.xlsx --topn 10
# 2) 在 Excel 的「候选清单」页，「选中(填1)」列给正确 chunk 填 1（多个共同作答就多填）
# 3) 回收勾选 → mapping.jsonl，并直接写回 golden
python scripts/export_chunk_candidates_xlsx.py --apply-xlsx chunk_review.xlsx \
    --mapping-out mapping.jsonl \
    --in-golden datasets/xlsx_golden.jsonl --out-golden datasets/xlsx_golden.jsonl
```

`--all` 连已有映射的题一起导出（复核已有标注用）。同一问题文本只请求一次检索，结果复用。

**备选：JSONL 候选 + 自动映射**

```bash
# 1) 对 needs_chunk_mapping=true 的用例跑检索，输出候选清单
python scripts/assist_chunk_mapping.py --golden datasets/xlsx_golden.jsonl \
    --config config/test.yaml --out /tmp/chunk_candidates.jsonl
# 2) 人工确认后写回（mapping.jsonl: question_id/expected_chunk/acceptable_chunks）
python scripts/assist_chunk_mapping.py --apply mapping.jsonl \
    --in-golden datasets/xlsx_golden.jsonl --out-golden datasets/xlsx_golden.jsonl
```

`--auto-map` 的覆盖率是拿「同一批检索结果」自证，只能当粗筛，正式金标准请人工勾选。

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
# 人工评审 xlsx：客观指标结果 + 主观指标逐题评分表 + 评分规则（详见下节）
python scripts/export_review_xlsx.py \
    --per-question results/<experiment_id>/per_question.jsonl \
    --summary results/<experiment_id>/summary.json \
    --out results/<experiment_id>/review.xlsx
# 阈值校准：① 从跑批结果抽候选并预标 → ② 人工确认 → ③ 扫描阈值（详见下节）
python scripts/make_calibration_set.py --per-question results/<experiment_id>/per_question.jsonl \
    --out datasets/calibration_candidates.jsonl --review-out datasets/calibration_review.jsonl
python scripts/calibrate_thresholds.py --config config/test.yaml \
    --data datasets/calibration_candidates.jsonl --kind all --out results/calibration
```

## 人工评审 xlsx

主观指标（GEN-01/03/04/06/07/08、RAG-06/07、TRACE-02）无法纯规则判定，
`export_review_xlsx.py` 把它们展开成逐题待评分行，附完整上下文和打分规则，
输出 4 个 sheet 的 xlsx 供人工审查打分：

| Sheet | 内容 | 用途 |
|---|---|---|
| **评分规则** | 全部指标（主观+客观）的口径、打分规则、方向、阈值、是否一票否决 | 参考手册 |
| **人工评分表** | 逐题 × 主观指标：问题/参考答案/系统答案/检索证据/事实点/禁止事项 + 空白评分列 | **主表，人工填分** |
| **客观指标结果** | 逐题自动计算指标（Top@k/MRR/NDCG/fact_recall/abstention 等） | 对照参考 |
| **汇总** | 聚合指标 + 阈值判定（需 `--summary`） | 总览 |

人工评分表列说明：
- **绿底**（输入）：问题、参考答案、必须覆盖事实、禁止事项、检索证据 — 评审依据
- **黄底**（输出）：系统答案 — 待评审对象
- **橙底**（人工填写）：人工评分（下拉选择，如 0/1/2）、评分理由

打分规则（以 GEN-01 为例）：`0=错误或关键事实错误；1=基本正确但缺次要信息；2=正确完整`。
GEN-08 仅对步骤/事故类题型展开，RAG-06 仅对 `must_cite=true` 的题展开。

## 判定口径与阈值校准

### 规则判定口径（已落地）

| 环节 | 口径 | 说明 |
|---|---|---|
| 归一化 | NFKC + 去零宽/隐藏空格 + 单位别名（兆帕→MPa、摄氏度→°C）+ 千分位 + `至/到/～/—`→`~` | 两侧统一归一化后再比较，`18.84 MPa` 与 `18.84MPa` 等价 |
| 禁止项 GEN-09 | 命中词 + **同分句**否定语境判定 | 「严禁按 5.7MPa 执行」「不得使用 XX」判为警示，不触发否决；真违规仍记 1 |
| 数值匹配 GEN-02 | **十进制精确相等，无容差** | `18.84≠18.8`、`18.84≡18.840`；同量纲换算后精确相等才算命中（`0.6MPa≡600kPa`、`1kW≡1000W`、`0.6kgf/cm²≡0.0588399MPa`）；区间只认端点，中间值不覆盖 |
| 标签匹配 GEN-02 | 字面 → 同义词 → 核心词 → 语义 | 同义词表 `config/synonyms.yaml`，支持长标签内术语替换（`汽轮机润滑油压`→`汽机润滑油压`）；核心词需显式配置；语义兜底仅在 `embedding.enabled` 时生效 |
| 邻近窗口 GEN-02 | 数值与标签 ≤40 字共现 | 超出记 0.5（`uncertain`，进复核），不直接判错 |
| 分档 GEN-02 | 1.0 / 0.8 / 0.5 / 0 | 严格口径 `fact_covered = credit ≥ 0.8`；`fact_recall_relaxed` 是分档均值，`fact_uncertain_count` 记待复核条数 |
| 顺序 GEN-ORDER | 取最早**连续段**锚点 | 「如前所述，水压试验…」的顺带提及不算正文位置；数值缺失时用标签位置，不再整题 `None` |
| 单位 GEN-UNIT | 只统计**出现过的**事实的单位 | 整条缺席的事实不参与，避免无关事实的单位拉低该项 |
| 引用 GEN-10 | 强/弱双档 | 强模式（`【】`、`《》`、裸文件名、`来源：`、`依据/根据+条款号`、`GB/DL/NB`）全文匹配；弱模式（`[1]`、`[^1]`、`(1)`）只在**末尾 200 字**生效，避免正文序号误判 |
| 拒答 GEN-05 | 三态 `abstention_state` | 1=拒答 / 0=作答 / `None`=长答案先拒后答等混合情形（不进分母、进复核队列）；`不确定度/不确定性` 等计量术语不判拒答 |
| 证据覆盖 | 归一化 + 字二元组 ≥0.85 兜底 | 换行/空格造成的切句差异不再掉分；跨段断裂的近似表述也能命中 |
| 跨文档 RAG-10 | gold 纳入 `acceptable_chunks` | 只填了可接受 chunk 的题不再被当 `None` 跳过 |

> 验收阈值（`THRESHOLDS` / `STAGE_GATES` / `VETO`）本轮**未改动**：口径修正只影响分子/分母判定，不放宽验收线。

### 语义兜底（可选，默认关闭）

配置 `embedding:` 段后生效，**只补漏、不改判**（已命中的规则结论不会被推翻）：

```yaml
embedding:
  enabled: true
  interface: "openai"          # openai=POST {base_url}/embeddings | tei=POST {base_url}/embed
  base_url: "http://127.0.0.1:9997/v1"
  model: "bge-m3"
  dim: 1024
  fact_label_threshold: 0.75
  evidence_threshold: 0.80
```

生效位置：
- `fact_covered`：标签字面缺席时，取「标签 vs 答案单句」的最大余弦，所有标签都需达阈值
- `evidence_cover`：分句字面与二元组都没命中时，取「分句 vs TopK 单句」的最大余弦

数值判定不参与语义兜底（有测试锁定）；服务不可用/超时自动降级为 `None` 并记 `last_error()`，跑批不中断。

### 术语同义词表（`config/synonyms.yaml`）

GEN-02 判定标签时，除字面匹配外还会查这张表（保存即生效，最多 1 秒延迟）：

```yaml
汽轮机: [汽机]              # 简写：等价词
水压试验:                    # 显式写法：可加核心词
  synonyms: [打压试验]
  core: [水压, 试验]
```

规则：
- 等价词必须是**同义**而非同族（`给水前置泵` ≠ `主给水泵`），否则会把错答判成对答；
- 组内做传递闭包（`A~B`、`B~C` ⇒ `A~C`）；
- 核心词不自动生成，只按配置生效；核心词全命中按 **0.8 分**计（不打满）；
- 长标签内的术语也会替换（`汽轮机润滑油压` → `汽机润滑油压`），因此不必为每个组合单独配置；
- 表里留空 = 纯字面口径，与未引入该功能一致。

宽松口径（`fact_recall_relaxed`）只在报告里与严格口径并列展示，**不参与阶段判定**；`0.5 分档`的事实请走复核（报告中 `GEN-02-relaxed` 的 `uncertain` 计数）。

### 阈值校准

分三步：**抽候选并预标 → 人工确认 label → 扫描阈值**。

```bash
# 1) 从跑批结果抽候选样本（自动预标 + 待确认清单）
python scripts/make_calibration_set.py \
    --per-question results/<experiment_id>/per_question.jsonl \
    --out datasets/calibration_candidates.jsonl \
    --review-out datasets/calibration_review.jsonl \
    --max-per-kind 80 --cross-negatives 1

# 2) 人工确认：只改 review 文件里 label 为 null 的行（改成 0/1），回填候选文件
#    预览标注格式：
python scripts/calibrate_thresholds.py --emit-template

# 2) 用人工标注样本扫描（fact/evidence 需 embedding.enabled=true；fuzzy 不需要）
python scripts/calibrate_thresholds.py --config config/test.yaml \
    --data datasets/calibration.jsonl --kind all \
    --grid 0.60:0.95:0.05 --min-precision 0.95 --out results/calibration
```

**候选抽取的预标判据**（只用与待校准信号无关的判据，避免自证）：

| basis | 含义 | 预标 |
|---|---|---|
| `literal` | 归一化后字面包含 | 1 |
| `number_mismatch` | 数值十进制精确比较后对不上 | 0 |
| `cross_question` | 与另一题（不同改写组/文档）的答案或片段配对 | 0 |
| `needs_review` | 规则无法判断（改述/语义相近，正是要人判的样本） | `null` |

`label` 为 `null` 的行会被校准脚本跳过并提示条数；每个 kind 默认限量 80 条，且优先保留 `needs_review` 与 `cross_question`（信息量最大）。`--chunks /tmp/chunk_candidates.jsonl`（`assist_chunk_mapping.py` 的产出）可提供未截断的片段，否则用 `evidence_snippet`（每条截断 300 字），此时数字冲突判据自动跳过。

`--cross-negatives 1` 表示每题再构造 1 条跨题负例；真实跑批里本来就有的错答（`number_mismatch`）会自然进样本，一般够用，跨题负例用于补足负例数量。

样本 jsonl 每行一条，`label` 填人工结论（1=应命中，0=不应命中）：

```jsonl
{"kind":"fact","fact":"给水泵出口压力0.6MPa","answer":"泵出口压力为0.6MPa","label":1}
{"kind":"fact","fact":"水压试验压力18.84MPa","answer":"水压试验压力为18.8MPa","label":0}
{"kind":"evidence","evidence":"给水泵出口压力不低于0.6MPa","chunk":"泵的出口压力须保持在0.6MPa以上","label":1}
{"kind":"fuzzy","clause":"给水泵出口压力不低于0.6MPa","chunk":"泵出口压力保持0.6MPa以上","label":0}
```

建议每行只放一个要点 / 一个分句，标注才可解释。脚本输出：
- 终端 markdown 表：每个阈值的 TP/FP/FN/TN + precision/recall/F1，✅ 标出推荐值
- `results/calibration/<kind>_calibration.json`：逐条分数、混淆矩阵、推荐阈值、分数分布

推荐规则：在 `precision ≥ --min-precision` 且 `recall ≥ --min-recall` 的阈值中取 F1 最高（并列取更高阈值）；都不满足时退回 F1 最高并提示「先扩充/复核样本」。`fuzzy` 的结论对应代码常量 `CLAUSE_FUZZY_THRESHOLD`（`src/metrics_retrieval.py`），`fact`/`evidence` 对应配置项 `fact_label_threshold` / `evidence_threshold`。

> 注意：语义阈值依赖 embedding 质量与样本分布，**必须先跑校准再改配置**；样本少于 50 条时结论仅供参考。

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
