# 问答测试集字段规范（DATASET_SPEC V2）

依据《企业RAG测试集构建要求》Table9，并对历史字段做精简收敛。
原则：**一义一键**——同一语义只保留一个规范键，旧键仅读兼容、新数据不再写。

## 字段表（`datasets/golden_sample.jsonl` 每行一个用例）

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `question_id` | 是 | string | 用例唯一编号（旧 `case_id` 可读代换） |
| `question` | 是 | string | 用户问题；拒答题也保留原问法 |
| `test_stage` | 是 | enum | `parse/chunk/retrieval/rerank/qa/refusal/permission`（默认 `qa`，拒答题默认 `refusal`） |
| `question_type` | 是 | enum | 十二类：`fact/parameter/condition/procedure/accident_symptom/accident_cause/accident_action/compare/table/range/multi_document/negative`（旧 `accident` 视为 `accident_action`） |
| `paraphrase_group` | 建议 | string | 同一知识点改写组；同组多题按组统计召回，不虚增覆盖（§8.2-10） |
| `difficulty` | 是 | enum | `easy/medium/hard` |
| `risk_level` | 是 | enum | `low/medium/high/critical`；高/关键题须填 `mandatory_points` |
| `answerable` | 是 | bool | 是否可答；与 `should_refuse` 恒互反，写其一即可 |
| `expected_answer_points` | 可答题是 | string[] | 标准答案要点：必须表达的事实、条件、结论（GEN-02 口径） |
| `mandatory_points` | 高风险题是 | string[] | 不可遗漏项：缺失会改变操作或安全含义（§3-5 逐条复核） |
| `forbidden_claims` | 建议 | string[] | 禁止内容：无依据/对象错/版本错的结论（GEN-09 一票否决） |
| `evidence_text` | 可答题是 | string | 原文证据（§8.2-1 每题须有明确证据） |
| `chapter_path` | 可答题是 | string | 章节路径，如 `1 机组启动 > 1.1 检修后的验收与试验`（§11 章节覆盖） |
| `table_or_item` | 表格题是 | string | 表格名/行列/条款编号，如 `水压试验表/省煤器行/超压试验压力列`（§8.2-6） |
| `expected_document/chunk` | 检索题是 | string | 入库后映射的目标文档/Chunk；与 `document_ids/chunk_ids` 自动合并 |
| `document_ids/chunk_ids` | 可答题是 | string[] | 预期命中集合；`acceptable_chunks` 中任一命中亦算命中 |
| `acceptable_chunks` | 按需 | string[] | 其他可接受 Chunk |
| `relevance` | 按需 | map | chunk→0/1/2，用于 NDCG；缺省全按 1 |
| `golden_answer` | 建议 | string | 自然语言参考答案（供人工/LLM 复核阅读，判分以要点为准） |
| `must_cite` | 否 | bool | 是否须引用，**默认 true**；GEN-10 只统计可答且未拒答题 |
| `should_refuse` | 拒答题是 | bool | 是否应拒答（GEN-05 口径） |
| `refusal_reason` | 拒答题是 | enum | `no_answer/insufficient/ambiguous/false_premise/stale_version/no_permission`（§7.2 六类） |
| `requires_order` | 步骤题是 | bool | 步骤顺序是否强制（GEN-ORDER 口径） |
| `requires_unit` | 参数题是 | bool | 数值是否须带单位/上下限方向（GEN-UNIT 口径） |
| `attachment_chunk_ids` | 附件题是 | string[] | 附件所在 Chunk；**非空即视为需附件**，不再另写布尔 |
| `source_version/hash` | 是 | string | 受控原文版本与文件哈希（可重复性绑定，§5.1/§10） |
| `reviewer/review_status` | 正式集是 | string/enum | 审核人；`pending/approved/rejected/deprecated`（`approved` 须有审核人） |
| `remarks/notes` | 按需 | string | 判定说明、干扰项备注 |

## 已废弃（读兼容、禁再写，`validate_dataset.py` 会告警）

`required_facts`→`expected_answer_points`、`forbidden_facts`→`forbidden_claims`、
`golden_facts`（无判读方）、`optional_facts`/`allowed_paraphrase`（改用改写组多题）、
`must_answer`（恒等于 `answerable && !should_refuse`）、`must_abstain`（→`should_refuse`）、
`requires_attachment`（→`attachment_chunk_ids` 非空推导）。

## 推导规则（`GoldenItem.from_dict`）

- `should_refuse = should_refuse ⏷ must_abstain ⏷ ¬answerable`；`answerable` 缺省取反。
- `expected_answer_points = expected_answer_points ⏷ required_facts ⏷ golden_facts`。
- `must_cite` 缺省 `true`；拒答/不可答题不计 GEN-10。
- `expected_chunk/document` 自动并入 `chunk_ids/document_ids`。

## 最小示例

```json
{"question_id": "PAR-TABLE-001", "test_stage": "qa", "question": "省煤器、水冷壁和过热器的超压试验压力是多少？",
 "question_type": "table", "difficulty": "medium", "risk_level": "high", "answerable": true,
 "expected_answer_points": ["超压试验压力23.55MPa"],
 "mandatory_points": ["数值23.55", "单位MPa", "对象省煤器/水冷壁/过热器"],
 "forbidden_claims": ["再热器超压试验压力"],
 "evidence_text": "水压试验表格中省煤器、水冷壁、过热器对应超压试验压力单元格",
 "chapter_path": "1 机组启动 > 1.1 检修后的验收与试验 > 1.1.5 水压试验",
 "table_or_item": "水压试验表/省煤器水冷壁过热器行/超压试验压力列",
 "expected_document": "DOC-600MW集控运行规程-20230831",
 "requires_unit": true, "must_cite": true, "source_version": "20230831", "review_status": "pending"}
```

```json
{"question_id": "AMB-OIL-001", "test_stage": "refusal", "question": "油压低了怎么办？",
 "question_type": "negative", "difficulty": "medium", "risk_level": "high", "answerable": false,
 "should_refuse": true, "refusal_reason": "ambiguous",
 "source_version": "20230831", "review_status": "pending",
 "remarks": "润滑油/EH油/密封油均可能油压低，不得自行假定"}
```

## 校验

`python scripts/validate_dataset.py <file>`：废弃键告警、拒答六类、表格行列、步骤顺序、参数单位、高风险要点、审核人规则；`python scripts/coverage_check.py <file>`：§11 配额与章节缺口。
