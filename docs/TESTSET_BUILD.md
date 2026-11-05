# 企业 RAG 测试集构建要求 → 本项目落地映射

来源：《企业RAG测试集构建要求_600MW规程案例.docx》V1.0（案例文件 `600MW机组集控运行规程20230831.docx`：约 8385 段落 / 104 表格 / 216 标题 / 32 内联图形 / 2 嵌入对象）。

核心红线：**以受控原文为事实基准，不得以 RAGFlow 解析结果或回答作为唯一标准答案来源；正式金标准必须经专业人员审核。**

## 七层结构（§4 Table3）与本项目对应

| 测试层级 | 验证问题 | 本项目产物 |
|---|---|---|
| 原文解析 | 原文是否完整正确被解析 | `extract_docx_baseline.py` → 基线；`parse_benchmark.py` → `parse_summary.json`（DATA-01 四项）+ `parse_anomalies.jsonl`（缺失/重复/错位/截断/无法定位） |
| 切块与索引 | 是否形成完整可定位可检索 Chunk | `list_all_chunks` 全量拉取 + 原文→Chunk 反向映射；异常 Chunk 清单 |
| 召回 | 正确 Chunk 是否进入候选集 | `retrieval_benchmark.py`：Recall@K / Hit@K（`expected_chunk` + `acceptable_chunks` 均算命中） |
| 重排 | 正确 Chunk 是否稳定排前列 | `rerank_compare: [off, on]` 同一批问题同候选集 A/B；`rerank_gain`（重排只调序不补召回） |
| 问答与引用 | 答案是否正确完整引用正确 | `run_all.py`：要点召回/顺序/单位规则版 + `review_queue` 半自动复核 |
| 拒答 | 无依据时是否停止回答 | `refusal_reason` 六类 + `REFUSAL-by-reason` 统计 |
| 版本与权限 | 是否只用有效且有权内容 | 版本冲突题（RAG-08/DATA-03）+ `platform_checks.py` 演练清单（越权零召回/零引用/零泄露） |

解析与问答分离（§3-3）：先跑通解析比对确认内容进库，再评价召回/重排/回答，避免把解析问题误判为模型问题。

## 用例字段（Table9）

字段规范见 [DATASET_SPEC.md](DATASET_SPEC.md)（V2 精简版：一义一键，旧键只读兼容）。
核心键：`question_id`、`test_stage`（七层）、`question_type`（十二类）、`question`、
`paraphrase_group`、`expected_answer_points`、`mandatory_points`、`forbidden_claims`、
`evidence_text`、`chapter_path`、`table_or_item`、`expected_document/chunk`、
`acceptable_chunks`、`should_refuse`、`refusal_reason`（六类）、`requires_order/unit`、
`source_version/hash`、`reviewer/review_status`。

## 问题类型与编写规则（Table8 + §8.2）

十二类：直接事实 / 参数限值 / 条件 / 步骤 / 事故现象 / 事故原因 / 事故处置 / 对比 / 表格 / 区间判断 / 跨段综合 / 拒答澄清。编写十条硬规则中最易犯的三条：表格题必须记行名列名单位单元格关联（禁孤立数值）；步骤题拆分前置条件/动作/判据/注意/禁止；同一知识点原话+口语+干扰表达归入同一 `paraphrase_group`（按组统计召回，不虚增覆盖）。

## 人机协同十步（§10）与三层集

锁定版本哈希 → 章节覆盖清单 → 专业标记重点 → 工具生成候选 → 自动格式检查 → 专业逐条审核核心集 → 签发金标准（记审核人/日期）→ 入库映射 Chunk 跑测试 → 八类失败归因 → 变更回归只复核差异。

三层集：`core`（120 条全审核，发布验收/回归基线）→ `extended`（扩至 300~500，工具生成+抽查）→ `adversarial`（模糊/错误前提/对抗持续补，发现问题转入核心）。

## 初始配额（§11 Table11）与验收（Table5/Table12）

配额：参数限值 20 / 启停条件 15 / 操作步骤顺序 20 / 事故现象原因处理 20 / 表格查询比较 15 / 口语改写 10 / 跨章节综合 8 / 无答案拒答 7 / 相似干扰错误前提 5，合计 120；章节覆盖机组启动/正常运行/停运/事故处理/升压站/主要辅机/继电保护/厂用电；表格≥20%。`scripts/coverage_check.py` 输出缺口清单。

验收：解析段落/标题/数字单位/表格关联/定位率（关键 100%）、召回 Recall@K/Hit@K、重排 MRR/nDCG（A/B 对比）、回答正确完整忠实、引用准确覆盖可定位、拒答三率、权限越权为 0。失败归因八类：原文/解析/切块/索引/召回/重排/生成/策略，`report.md` 输出 `_attribution` 分布。审核用附录 B 十二项检查表（来源/版本/对象/参数/条件/顺序/安全/引用/干扰/拒答/判定/结论）。
