# 阶段 1：GraphRAG-Bench Novel 数据选择报告

本报告只覆盖官方数据下载、检查、选书、筛选和 8:2 分层划分。
未进行文本切块、知识图谱构建、检索、LLM 调用或分类器训练。

## 官方来源与版本

- 官方仓库：https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- 实际 remote：`https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git`
- Git commit：`fdbab5959b18c96532580877ffe27d112bccc0ec`
- 下载日期：2026-09-05（Asia/Shanghai）
- `Datasets/Corpus/novel.json`：4,848,282 bytes，SHA-256 `b0b4b2e7dc1b5d72f774783130d08b5d62e922fa9dc3e58612154b78b6089d15`
- `Datasets/Questions/novel_questions.json`：1,312,016 bytes，SHA-256 `d02b3db9c44d273371960f700f9ff9bf174dbc885542a7cadf636cded32c7af1`

## 原始结构与完整性

- 语料文件：JSON 数组，共 20 项；字段 `['context', 'corpus_name']`。
- 问题文件：JSON 数组，共 2010 行；字段 `['answer', 'evidence', 'evidence_triple', 'id', 'question', 'question_type', 'source']`。
- 实际 question_type：`{'Complex Reasoning': 610, 'Contextual Summarize': 362, 'Creative Generation': 67, 'Fact Retrieval': 971}`。
- corpus_name 全部唯一；无空文本、空问题、空答案或无法匹配的 source。
- 未在标题或全文中发现《红楼梦》及列出的中英文变体。
- **官方原始异常：**全量问题存在重复 ID：`Novel-55f0c0e2` 出现在 ['Novel-47558', 'Novel-4128']。
  原始文件未被修改；该重复项不属于最终选中的作品。选中作品、训练集和
  测试集均已重新断言 ID 唯一。

## 全部 20 个候选语料

每项至少检查文本开头 2,000 字符；表中的作品类型不是因位于 Novel 目录而自动判定。

| corpus_name | 标题 | 体裁 | 字符 | 近似词数 | 全部题 | Fact | Complex | 两类合计 | Summarize | Creative | 连续叙事 | 合格 | 排除原因 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Novel-10146 | Reminiscences of Pioneer Days in St. Paul | 地方史文章合集 | 244,216 | 42,433 | 126 | 65 | 42 | 107 | 16 | 3 | 否 | 否 | 由报刊文章汇编而成，涉及多组地方史主题，不是单一完整作品叙事 |
| Novel-10321 | Dragon's Blood | 小说 | 316,622 | 53,841 | 71 | 40 | 16 | 56 | 15 | 0 | 是 | 否 | 两类可用问题仅 56 道，少于 70；Complex Reasoning 仅 16 道，少于 25 |
| Novel-10356 | Travels in Morocco, Volume II | 旅行叙事 | 308,774 | 52,029 | 130 | 64 | 42 | 106 | 22 | 2 | 是 | 是 | — |
| Novel-10762 | Impressions of Theophrastus Such | 讽刺随笔／人物素描集 | 342,668 | 58,359 | 78 | 36 | 19 | 55 | 17 | 6 | 否 | 否 | 由独立社会观察和人物素描组成，不是连续情节叙事；两类可用问题仅 55 道，少于 70；Complex Reasoning 仅 19 道，少于 25 |
| Novel-2544 | From Sand Hill to Pine | 短篇小说集 | 344,399 | 60,973 | 114 | 50 | 37 | 87 | 23 | 4 | 否 | 否 | 目录包含多篇不同人物和情节的短篇故事 |
| Novel-25646 | Child's Health Primer | 健康教材 | 99,076 | 18,042 | 104 | 48 | 30 | 78 | 21 | 5 | 否 | 否 | 面向初级课堂的生理与健康教材，不是叙事作品 |
| Novel-26183 | Laurence Sterne in Germany | 文学研究专著 | 450,081 | 71,482 | 96 | 43 | 36 | 79 | 14 | 3 | 否 | 否 | 学术研究专著，不是叙事性文学作品 |
| Novel-29973 | The 'Patriotes' of '37 | 历史叙事 | 164,036 | 27,716 | 102 | 46 | 29 | 75 | 22 | 5 | 是 | 是 | — |
| Novel-30752 | Vestiges of the Mayas | 考古学论著 | 168,274 | 29,099 | 87 | 45 | 27 | 72 | 14 | 1 | 否 | 否 | 考古与跨文明联系论著，不是人物和情节连续的叙事作品 |
| Novel-40700 | Dandy Dick | 三幕喜剧／闹剧 | 137,923 | 23,593 | 102 | 43 | 35 | 78 | 20 | 4 | 是 | 是 | — |
| Novel-4128 | The Diary of Samuel Pepys: June-August 1661 | 日记体叙事 | 92,796 | 18,244 | 72 | 40 | 15 | 55 | 16 | 1 | 是 | 否 | 两类可用问题仅 55 道，少于 70；Complex Reasoning 仅 15 道，少于 25 |
| Novel-41603 | Toto's Merry Winter | 儿童小说 | 267,307 | 48,416 | 103 | 51 | 24 | 75 | 23 | 5 | 是 | 否 | Complex Reasoning 仅 24 道，少于 25 |
| Novel-44557 | An Unsentimental Journey Through Cornwall | 旅行叙事 | 237,566 | 41,843 | 103 | 49 | 31 | 80 | 18 | 5 | 是 | 是 | — |
| Novel-47558 | Pen Pictures of Eventful Scenes and Struggles of Life | 回忆性事件合集 | 143,800 | 25,728 | 108 | 49 | 35 | 84 | 19 | 5 | 否 | 否 | 作者明确汇集多人和多段事件经历，不是单一连续叙事 |
| Novel-47676 | The Amores; or, Amours | 诗歌／哀歌集 | 304,362 | 53,925 | 81 | 41 | 23 | 64 | 13 | 4 | 否 | 否 | 由多首独立哀歌组成，缺少贯穿全书的连续情节；两类可用问题仅 64 道，少于 70；Complex Reasoning 仅 23 道，少于 25 |
| Novel-51410 | Dr. Elsie Inglis | 传记 | 334,209 | 60,635 | 78 | 43 | 17 | 60 | 18 | 0 | 是 | 否 | 两类可用问题仅 60 道，少于 70；Complex Reasoning 仅 17 道，少于 25 |
| Novel-54537 | Musical Instruments | 博物馆艺术手册 | 190,409 | 31,751 | 120 | 53 | 39 | 92 | 19 | 9 | 否 | 否 | 乐器历史与藏品说明手册，不是叙事作品 |
| Novel-58553 | An Astronomer's Wife | 传记 | 191,743 | 33,786 | 112 | 49 | 38 | 87 | 23 | 2 | 是 | 是 | — |
| Novel-5956 | Gallegher and Other Stories | 短篇小说集 | 255,527 | 47,802 | 86 | 46 | 25 | 71 | 15 | 0 | 否 | 否 | 目录包含多篇互不连续的短篇故事 |
| Novel-8559 | Scientific American Supplement No. 360 | 科学期刊合集 | 225,822 | 39,911 | 137 | 70 | 50 | 120 | 14 | 3 | 否 | 否 | 同一期包含工程、化学、自然史等多篇文章，不是单一叙事作品 |

## 选择结果

- 选中：`Novel-40700`，*Dandy Dick*。
- 体裁：三幕喜剧／闹剧。
- 长度：137,923 字符，约 23,593 个空白分词。
- 原始问题 102 道；筛选后 78 道（Fact 43，Complex 35）。
- 选择理由：它是单一且情节连续的三幕戏剧，满足两类题量硬门槛，不属于禁用作品，并且是所有合格候选中近似词数最少的作品。
- 完整确定性规则：先满足非《红楼梦》、单一连贯叙事作品、Fact Retrieval + Complex Reasoning >= 70、Complex Reasoning >= 25、文本非空且 source 严格匹配；再优先单一完整作品；以近似词数最少为主，距最短作品不超过 10% 时优先Complex Reasoning 更多者，最后按 corpus_name 字典序。

## 分层划分

| 集合 | Fact Retrieval | Complex Reasoning | 合计 |
|---|---:|---:|---:|
| 筛选后 | 43 | 35 | 78 |
| 训练集 | 34 | 28 | 62 |
| 测试集 | 9 | 7 | 16 |

划分前按 ID 排序，再在每种 question_type 内以 seed=42 独立打乱；训练/测试 ID 无交集且并集等于全部筛选问题。未创建验证集。

## 数据泄漏防护

- 原始官方 JSON 保持只读，不进行修正或去重。
- `test_queries.jsonl` 只含 id、source、question、question_type。
- 完整测试标签只保存在 `test_questions_labeled.jsonl`，供未来最终评估读取。
- `indexing_input.json` 只含 corpus_name、title、context。
- 后续索引或建图不得读取 questions、answer、evidence 或 evidence_triple。

## 数据检查示例

> 以下答案和证据仅用于数据检查，不能进入建图输入。

### Fact Retrieval

- ID：`Novel-0047c320`
- 问题：Who is the character that recognizes The Dean during his imprisonment in the police station, identifying him despite his disguise?
- 答案：Hannah is the character who recognizes The Dean.
- 证据：Hannah is the character who recognizes The Dean during his imprisonment in the police station.; Hannah identifies The Dean despite his disguise.

### Complex Reasoning

- ID：`Novel-025c325c`
- 问题：In the narrative of 'Dandy Dick', by what other name is THE DEAN known?
- 答案：THE DEAN is also known as Augustin Jedd.
- 证据：The Dean is also known as Augustin Jedd.

## 后续边界

**下一步只能从文本清洗与切块开始，且只能读取 `data/processed/phase1/indexing_input.json`。当前阶段到此停止。**
