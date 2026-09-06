# 阶段 3：共享知识图谱构建

## 范围与配置

- 唯一正文输入：`data/processed/phase2/chunks.jsonl`（62 个 chunk）
- 输入 SHA-256：`a081ba0f863a37e71fae17a8c7b42f2aa46019e8d62ca1d98e818e4988811427`
- 模型：`deepseek-v4-pro`
- OpenAI 兼容接口：`https://api.deepseek.com`
- temperature：`0`；max_retries：`2`；thinking：`disabled`
- 原始响应按 chunk 缓存；后处理可完全离线重复运行。

本阶段只读取 phase2 `chunks.jsonl`，未读取问题、答案、`evidence` 或
`evidence_triple`。未生成问题、未训练分类器、未生成 Embedding，也未实现
Vector、LightRAG 或 PathRAG 检索。

## 图谱统计

- 节点：124
- 逻辑边：80
- 实体类型：`{'ANIMAL': 6, 'CHARACTER': 14, 'CREATIVE_WORK': 8, 'EVENT': 5, 'LOCATION': 26, 'OBJECT': 19, 'ORGANIZATION': 12, 'PERSON': 34}`
- 关系类型：`{'ARRESTS': 1, 'AUTHORED_BY': 7, 'EMPLOYED_BY': 6, 'FRIEND_OF': 2, 'HELPS': 1, 'LOCATED_AT': 31, 'MANAGED_BY': 1, 'MARRIED_TO': 1, 'OPPOSES': 2, 'OWNS': 2, 'PARENT_OF': 2, 'PARTICIPATES_IN': 8, 'PERFORMED_AT': 2, 'PORTRAYED_BY': 11, 'ROMANTIC_WITH': 1, 'SIBLING_OF': 2}`
- 弱连通分量：73
- 孤立节点：70
- 平均度：1.290323
- Chunk 结果：`{'success': 62}`
- 节点来源覆盖：`{'act_1': 132, 'act_2': 157, 'act_3_scene_1': 107, 'act_3_scene_2': 105, 'cast': 28, 'front_matter': 7, 'production_history': 19}`
- 边来源覆盖：`{'act_1': 20, 'act_2': 22, 'act_3_scene_1': 14, 'act_3_scene_2': 21, 'cast': 14, 'front_matter': 5, 'production_history': 6}`

对称关系在正式图谱中仅存一条逻辑边并设置 `directed=false`。重复关系合并
后保留全部引文、chunk 来源和描述；GraphML 数组字段使用 JSON 字符串。

## 别名归一化

- `Augustin Jedd` ← `Augustin`, `Dean`, `Dr. Jedd`, `Gus`, `Jedd`, `Pa`, `Papa`, `Papsey`, `poor Papa`, `THE DEAN`, `the Dean of St. Marvells`, `The Very Rev. Augustin Jedd, D.D.`
- `Georgiana Tidman` ← `Aunt George`, `Aunt Georgiana`, `Aunt Tidman`, `George`, `George Tidd`, `GEORGIANA`, `Mrs. Tidman`, `Tidd`
- `Sir Tristram Mardon` ← `Mardon`, `SIR TRISTRAM`, `Sir Tristram Mardon, Bart`, `Tris`, `Tris Mardon`, `Tristram`

- 审计映射条目：196
- 未解决别名：5
- `ALL` 仅作为集体台词标签，未成为实体；演员 PERSON 与角色 CHARACTER 未合并。
- `Dandy Dick` 固定为 ANIMAL；`Miss Jedd` 保持未解决，不猜测具体是哪位姐妹。

## LLM 使用

- 历史 API 尝试次数：64
- 本次运行 API 调用：0
- 本次缓存命中：62
- 历史 prompt tokens：96475
- 历史 completion tokens：82835
- 历史 total tokens：179310
- 历史请求耗时合计：823.993599 秒

密钥只从环境变量读取，从未写入代码、缓存、日志或本报告。

## 建图前后样例

建图前，各 chunk 独立包含模型输出的实体、关系及逐条原文引文。建图后，
别名被映射到稳定节点，重复逻辑边合并。例如 `THE DEAN`、`Augustin`、`Gus`
和 `Dr. Jedd` 归一为 `Augustin Jedd`，但每条边仍保留原始 chunk 与引文。

## 30 条关系人工抽查

- 抽样：fixed-seed stratified random sample（seed=42）
- 样本：30；已审：30；通过：30；通过率：100.00%
- 类别分布：`{'arrest_police_station': 4, 'ball_or_racing': 5, 'dandy_dick_ownership': 6, 'emotion': 3, 'family': 5, 'production': 7}`
- 只依据 phase2 原文支持性判断，不参考任何问题、答案或监督证据。

| # | 类别 | 关系 | Chunk | 原文引文 | 结论 |
|---:|---|---|---|---|---|
| 1 | family | Augustin Jedd —SIBLING_OF→ Georgiana Tidman | `dd_act01_c0009` | THE DEAN. My dear widowed sister, Georgiana Tidman. | 通过 |
| 2 | family | Hannah Topping —MARRIED_TO→ Noah Topping | `dd_act03s01_c0005` | Annah Topping, Knee Evans, wife o' the Constable | 通过 |
| 3 | family | Augustin Jedd —PARENT_OF→ Sheba Jedd | `dd_act01_c0009` | THE DEAN. [_Embracing his daughters._] A second mother to my girls. She will implant the precepts of retrenchment if their father cannot! SALOME. B… | 通过 |
| 4 | family | Augustin Jedd —PARENT_OF→ Salome Jedd | `dd_act01_c0009` | SALOME. Keep the expenses down! THE DEAN. [_Embracing his daughters._] A second mother to my girls. | 通过 |
| 5 | family | Salome Jedd —SIBLING_OF→ Sheba Jedd | `dd_act01_c0004` | SHEBA walk in together. SALOME has her arm round her sister's waist | 通过 |
| 6 | emotion | Augustin Jedd —FRIEND_OF→ Sir Tristram Mardon | `dd_act03s02_c0007` | SIR TRISTRAM. Jedd, you were once my friend, and you are to be my relative. | 通过 |
| 7 | emotion | Blore —FRIEND_OF→ Hannah Topping | `dd_act03s01_c0003` | HANNAH. [_Starting and replacing the book._] Oh don't! This is Mr. Blore from the Deanery come to see us--an old friend o' mine! | 通过 |
| 8 | emotion | Major Tarver —ROMANTIC_WITH→ Salome Jedd | `dd_act01_c0001` | I believe, Salome, that it is to _you_ Major Tarver is paying attention. | 通过 |
| 9 | dandy_dick_ownership | Georgiana Tidman —OWNS→ Dandy Dick | `dd_act03s02_c0007` | Dandy Dick, I denounce you! GEORGIANA. As the owner of the other half, _I_ denounce you! | 通过 |
| 10 | dandy_dick_ownership | Augustin Jedd —PARTICIPATES_IN→ Restoration Fund | `dd_act01_c0008` | the Dean of St. Marvells, whose anxiety for the preservation of the Minister Spire threatens to undermine his health, has subscribed the munificent… | 通过 |
| 11 | dandy_dick_ownership | Dandy Dick —PARTICIPATES_IN→ Durnstone Handicap | `dd_act01_c0015` | Dandy! SIR TRISTRAM. I brought him down with me in lavender. You know he runs for the Durnstone Handicap to-morrow. | 通过 |
| 12 | dandy_dick_ownership | Georgiana Tidman —PARTICIPATES_IN→ Durnstone Handicap | `dd_act01_c0016` | George--ha! ha! Well, now you know he's fit, of course, you're going to back Dandy Dick for the Durnstone Handicap. | 通过 |
| 13 | dandy_dick_ownership | Dandy Dick —LOCATED_AT→ The Deanery of St. Marvells | `dd_act03s02_c0004` | the horse who enjoyed the shelter of the Deanery last night---- SIR TRISTRAM. Dandy Dick | 通过 |
| 14 | arrest_police_station | Georgiana Tidman —LOCATED_AT→ St. Marvells | `dd_act03s01_c0009` | St. Marvells. NOAH. Hunlock that door! [_HANNAH unlocks the door, and admits GEORGIANA and SIR TRISTRAM, both dressed for the race-course._ GEORGIA… | 通过 |
| 15 | arrest_police_station | Noah Topping —ARRESTS→ Augustin Jedd | `dd_act03s01_c0012` | NOAH comes out of the cell with THE DEAN, who is in handcuffs. | 通过 |
| 16 | arrest_police_station | Sir Tristram Mardon —LOCATED_AT→ St. Marvells | `dd_act03s01_c0009` | St. Marvells. NOAH. Hunlock that door! [_HANNAH unlocks the door, and admits GEORGIANA and SIR TRISTRAM | 通过 |
| 17 | arrest_police_station | Noah Topping —LOCATED_AT→ St. Marvells | `dd_act03s01_c0002` | Mr. Topping's got the appointment of Head Constable at St. Marvells | 通过 |
| 18 | ball_or_racing | Georgiana Tidman —LOCATED_AT→ Newmarket | `dd_act01_c0011` | Ask after George Tidd at Newmarket--they'll tell you all about me. | 通过 |
| 19 | ball_or_racing | Sheba Jedd —PARTICIPATES_IN→ St. Marvells Spring Meeting | `dd_act03s02_c0007` | We have won fifty pounds. THE DEAN. What! SHEBA. At the Races! | 通过 |
| 20 | ball_or_racing | Salome Jedd —LOCATED_AT→ Durnstone Athenaeum | `dd_act02_c0016` | We blame officers for subjecting two motherless girls to the shock of alighting at the Durnstone Athenaeum to find a notice on the front door: "Bal… | 通过 |
| 21 | ball_or_racing | Major Tarver —PARTICIPATES_IN→ St. Marvells Spring Meeting | `dd_act03s02_c0001` | TARVER and DARBEY enter, dressed for the Races | 通过 |
| 22 | production | Lady Bountiful —AUTHORED_BY→ Arthur W. Pinero | `dd_front_c0001` | ARTHUR W. PINERO AUTHOR OF "SWEET LAVENDER," "THE TIMES," "THE CABINET MINISTER," "LADY BOUNTIFUL," | 通过 |
| 23 | production | Toole's Theatre —MANAGED_BY→ John Clayton | `dd_prod_c0002` | Mr. Clayton took a temporary lease of Toole's Theatre | 通过 |
| 24 | production | Georgiana Tidman —PORTRAYED_BY→ Mrs. John Wood | `dd_cast_c0001` | GEORGIANA TIDMAN (a Widow, MRS. JOHN WOOD. | 通过 |
| 25 | production | Sir Tristram Mardon —PORTRAYED_BY→ Edmund Maurice | `dd_cast_c0001` | SIR TRISTRAM MARDON, Bart MR. EDMUND MAURICE. | 通过 |
| 26 | production | Noah Topping —PORTRAYED_BY→ W. H. Denny | `dd_cast_c0001` | NOAH TOPPING (Constable at MR. W. H. DENNY. | 通过 |
| 27 | ball_or_racing | Augustin Jedd —LOCATED_AT→ St. Marvells | `dd_act01_c0008` | Dr. Jedd, the Dean of St. Marvells | 通过 |
| 28 | production | Sweet Lavender —AUTHORED_BY→ Arthur W. Pinero | `dd_front_c0001` | ARTHUR W. PINERO AUTHOR OF "SWEET LAVENDER," | 通过 |
| 29 | production | Salome Jedd —PORTRAYED_BY→ Miss Marie Lewes | `dd_cast_c0001` | SALOME } the Dean's Daughters { MISS MARIE LEWES. | 通过 |
| 30 | dandy_dick_ownership | John Fielder —HELPS→ Georgiana Tidman | `dd_act01_c0015` | GEORGIANA. Yes, directly I saw Dandy Dick marched out before the auctioneer I asked John Fielder to help me, and he did, like a Briton. | 通过 |

## 失败、异常与质量结论

- 抽取失败：0 个 chunk。
- 无事实：0 个 chunk。
- 未解决别名：5 条。
- 后处理警告：393 条，详见 `quality_report.json`。
- 后处理警告分布：`{'blocked_ambiguous': 6, 'model_alias_not_auto_merged': 16, 'relation_direction_corrected': 22, 'relation_endpoint_unresolved': 9, 'relation_rejected_dandy_owner_not_explicit': 5, 'relation_rejected_event_mentioned_not_participated': 2, 'relation_rejected_evidence_too_short': 10, 'relation_rejected_location_not_asserted_for_source': 13, 'relation_rejected_location_target_type_invalid': 6, 'relation_rejected_ownership_target_type_invalid': 5, 'relation_rejected_performance_endpoint_types_invalid': 3, 'relation_rejected_portrayal_endpoint_types_invalid': 3, 'relation_rejected_portrayal_not_cast_evidence': 1, 'relation_rejected_predicate_not_explicit_in_evidence': 134, 'relation_rejected_racing_tip_not_participation': 1, 'relation_rejected_source_not_named_in_evidence': 28, 'relation_rejected_target_not_named_in_evidence': 28, 'relation_rejected_unsupported_arrest_direction': 1, 'relation_rejected_unsupported_parent_direction': 15, 'relation_rejected_weak_relation_excluded': 64, 'type_conflict_relinked_to_known_entity': 21}`。其中大部分是保守规则主动拒绝弱关系或语义不足关系的审计记录，并非运行失败。
- 自动质量检查：`pass`。
- Graph JSON 与 GraphML 均已重新加载验证。

下一步是“实现三个共享数据基础上的检索器”；本阶段未自行开始。
