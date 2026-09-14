# LightRAG 知识图谱问答与可视化应用设计

> 2026-09-05 已按用户后续确认调整 Web 实施边界：检索采用可插拔接口，
> LightRAG 实现使用 `aquery_data()`，统一答案服务负责生成；原 CLI
> 继续使用 `aquery_llm()`。具体新协议、注册入口和限制以
> `docs/RETRIEVAL_PLUGINS.md` 为准。下文是原 V1 基线，不再要求 Web
> 必须使用一体化查询。没有额外实现新的 retrieval 算法。

日期：2026-09-03

状态：Frozen（V1 实施基线）

## 1. 项目定位

V1 是一个完整但精简的 LightRAG 前后端应用。它只验证一条端到端主链路：导入一份小文本，用 LightRAG 建立并复用索引，让用户手动选择 LightRAG 查询模式，然后展示答案、实际可获得的检索上下文和当前查询相关的局部知识图谱。

```text
GraphRAG-Bench 中的一份小文本
        ↓
     LightRAG
   ├─ indexing
   ├─ local / global / hybrid / mix
   ├─ retrieval context
   └─ answer generation
        ↓
      FastAPI
        ↓
 React + Cytoscape
```

GraphRAG-Bench 在 V1 中只是方便、真实且结构明确的文本来源。系统不计算 benchmark 分数，也不以实验排行榜为目标。

## 2. V1 功能范围

V1 只实现以下功能：

1. 从 GraphRAG-Bench Novel 或 Medical 中选择一份小文本。
2. 使用 LightRAG 原始文本插入能力建立知识图谱、文本块和向量索引。
3. 保存并稳定复用已经完成的 LightRAG workspace。
4. 接收用户手动输入的问题。
5. 允许用户选择 `local`、`global`、`hybrid` 或 `mix`。
6. 返回答案和当前 LightRAG 版本实际提供的检索上下文。
7. 展示可获得的 entities、relationships、chunks 和 references。
8. 使用 Cytoscape 展示当前查询相关的有限局部图并高亮直接命中内容。
9. 记录每次查询的 latency、mode、成功状态和错误摘要。
10. 将 DeepSeek、embedding、LightRAG 和索引状态错误转换为前端可理解的提示。

V1 不建设实验评测、批处理、模式并排比较或通用数据管理能力。页面一次只操作一个 active dataset 和一个 active workspace。

## 3. 数据输入

### 3.1 数据来源

输入只来自 GraphRAG-Bench 的 corpus 文件：

```text
GraphRAG-Benchmark/Datasets/Corpus/
├─ novel.json 或 novel.parquet
└─ medical.json 或 medical.parquet
```

内部只需要一个文档模型：

```python
class CorpusDocument:
    corpus_name: str
    context: str
    subset: Literal["novel", "medical"]
    character_count: int
    word_count: int
```

Novel 文件包含多本小说，用户可从列表中选择一本。为方便第一次运行，界面提供“选择最短文本”按钮，其规则仅为在 `context` 非空的文档中按字符数取最小值。Medical 作为一个完整文本选项保留，但不是首次演示默认值。

V1 不读取 GraphRAG-Bench Questions，也不依赖 answer、evidence 或 question type。用户直接在页面输入问题。

### 3.2 Active dataset

系统一次只维护一个 active dataset：

```text
data/active/
├─ corpus.json
└─ manifest.json
```

`corpus.json` 只保存选中的一个 `CorpusDocument`。`manifest.json` 记录：

- schema version
- subset 和 corpus_name
- 文本字符数与词数
- 原始文件路径
- 文本 SHA-256
- 选择时间

选择另一份文本时，后端先生成新的 fingerprint。若它与当前 active dataset 不同，前端明确提示需要切换 active dataset，并且不能继续使用旧 workspace 进行查询。

## 4. LightRAG 索引

### 4.1 唯一索引路径

```text
CorpusDocument.context
  → LightRAG ainsert()
  → chunking
  → LLM entity/relation extraction
  → graph/vector/chunk stores
  → graph_chunk_entity_relation.graphml
```

知识图谱完全由 LightRAG 从文本抽取。应用不预先生成三元组，也不向 LightRAG 注入已有图谱。

### 4.2 Workspace 目录

```text
data/workspace/
├─ manifest.json
└─ LightRAG 生成的存储文件
```

V1 只管理这一个 workspace。workspace manifest 记录：

- corpus fingerprint
- LightRAG 版本
- LLM provider、base URL 和 model name
- embedding model、维度和最大 token 数
- chunk size 和 overlap
- build status
- started_at、completed_at 和 latency
- error summary

### 4.3 状态与复用

索引状态固定为：

- `not_built`
- `building`
- `ready`
- `failed`
- `interrupted`

启动和查询前同时检查：

1. workspace manifest 的状态为 `ready`。
2. 当前 corpus fingerprint 与建图时一致。
3. 当前 embedding 配置与建图时一致。
4. LightRAG 必需存储文件存在且能够初始化。

全部满足时直接复用 workspace，不再次调用 `ainsert()`。任一关键条件不满足时禁止查询，并提示用户重新构建。构建操作使用进程内互斥锁，防止重复点击产生两个索引任务。

应用不自动删除旧 workspace。用户切换文本或配置后点击“重新构建”时，后端先将旧目录移动到带时间戳的备份目录，再创建新 workspace。

### 4.4 模型配置

- V1 开发基线固定为本机 LightRAG `v1.5.7-3-gc1248646`，README 和依赖文件记录相同版本或 commit。
- LLM：DeepSeek OpenAI-compatible API。
- Embedding：本地 Hugging Face BGE 模型。
- API key 只从后端环境变量读取。
- 模型名、base URL、超时、并发数和 embedding 参数由后端配置提供。
- 前端只展示非敏感配置状态，不接收或回显 API key。

### 4.5 图存储约束

V1 固定配置 `GRAPH_STORAGE=NetworkXStorage`。因此局部图服务可以读取该 storage 在 workspace 中持久化的 GraphML；默认知识图谱 namespace 对应 `graph_chunk_entity_relation.graphml`。

其他 graph backend 不属于 V1。GraphService 不扫描或猜测图文件，而是根据固定 workspace 和 namespace 解析 GraphML 路径。

## 5. 查询流程

### 5.1 请求

```json
{
  "query": "Who is the main character and what relationships shape the story?",
  "mode": "hybrid",
  "top_k": 5
}
```

`mode` 只能是 `local`、`global`、`hybrid`、`mix`。V1 始终生成答案，不增加 retrieval-only 或额外生成开关。

LightRAGAdapter 的唯一查询入口固定为 `aquery_llm()`：

```python
try:
    result = await rag.aquery_llm(
        query,
        param=QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=top_k,
            stream=False,
        ),
    )
except Exception as exc:
    raise map_lightrag_exception(exc) from exc

if result.get("status") != "success":
    raise LightRAGQueryError(result.get("message", "LightRAG query failed"))

llm_response = result.get("llm_response", {})
data = result.get("data", {})

answer = llm_response.get("content")
entities = data.get("entities", [])
relationships = data.get("relationships", [])
chunks = data.get("chunks", [])
references = data.get("references", [])
```

V1 不调用 `rag.query()` 后再尝试恢复 context。`stream=False` 显式保证非流式回答通过 `llm_response.content` 返回。Adapter 同时处理两类失败：直接抛出的 Python exception 由 `try/except` 映射为平台错误；正常返回但顶层 `status != "success"` 的 failure dict 在调用后检查。两层处理都不可省略。

LightRAG 的 `top_k` 主要控制 local mode 的实体数量和 global mode 的关系数量，文本块数量由 `chunk_top_k` 单独控制。V1 为保持单一、直观的前端控件，规定 `chunk_top_k = top_k`。界面标签仍为 “Top-K”，并在辅助文字中说明它会同时设置 KG 与 chunk 的候选上限；实际返回数量仍可能受去重、截断和可用结果影响。

### 5.2 执行顺序

```text
校验问题与 mode
  → 校验 workspace ready 且 fingerprint 匹配
  → 构造 LightRAG QueryParam
  → 调用一次 LightRAG 查询
  → 获取 answer 与可用 structured context
  → 转换为 QueryResult
  → 构造局部结果图
  → 记录 latency 或错误
  → 返回前端
```

最终答案由 LightRAG 内部配置的 DeepSeek provider 生成。后端不会把同一上下文再次发送给另一个答案生成函数。

### 5.3 四种模式

- `local`：以低层关键词、实体和邻域上下文为主。
- `global`：以高层关键词、关系和关联实体上下文为主。
- `hybrid`：使用 LightRAG 内部的 local/global 组合逻辑。
- `mix`：使用 LightRAG 内部的知识图谱与文本块组合逻辑。

LightRAG 还支持 `naive` 和 `bypass`，但 V1 不把它们加入 API 枚举或前端控件，只暴露以上四种。模式内部的候选生成、截断与合并全部由 LightRAG 负责。

## 6. QueryResult

后端用一个稳定模型隔离 LightRAG 版本细节：

```python
class QueryResult:
    run_id: str
    query: str
    mode: str
    answer: str | None
    latency_ms: float

    context_text: str | None
    entities: list[RetrievedEntity]
    relationships: list[RetrievedRelationship]
    chunks: list[RetrievedChunk]
    references: list[Reference]

    graph_nodes: list[GraphNode]
    graph_edges: list[GraphEdge]

    warnings: list[str]
    error: QueryError | None
```

`LightRAGAdapter` 只使用 `aquery_llm()` 返回的 `llm_response`、`data`、`metadata`、`status` 和 `message`。其中 `data` 是 entities、relationships、chunks、references 的唯一检索数据入口。映射原则为：

- LightRAG 返回什么，就展示什么。
- 不存在的集合返回空列表。
- 不存在的单值返回 `null`。
- 缺少预期字段时添加 warning。
- 不解析控制台 INFO 日志。
- 不推测 score、rank、引用或严格遍历顺序。

`context_text` 是用于前端展示的检索上下文文本。优先使用 LightRAG 正式接口提供的上下文；若仅提供结构化 entities、relationships、chunks 和 references，则由 Adapter 按固定模板序列化，仅用于展示，不宣称与 LightRAG 内部最终 prompt 完全一致。

## 7. 局部图谱

### 7.1 图谱来源

局部图使用两个可信来源：

1. QueryResult 中直接返回的 entities 和 relationships。
2. 当前 `NetworkXStorage` workspace 的 `graph_chunk_entity_relation.graphml`。

GraphService 先将直接返回的实体和关系标记为 `retrieved=true`，再从 GraphML 中补充最多一跳邻居用于理解上下文。补充内容标记为 `retrieved=false`，不能冒充检索命中。

如果 LightRAG 没有返回可映射的实体或关系，图谱区域显示空态和 warning，而不是根据问题文本猜测节点。

### 7.2 映射规则

- 优先使用 LightRAG 提供的实体标识。
- 没有标识时只允许规范化后的实体名称精确匹配 GraphML 节点名称。
- 不进行模糊匹配。
- 关系只有在起点、终点和关系类型可确认时才高亮。
- 每次最多返回一个较小的节点上限，避免浏览器绘制完整图谱。

### 7.3 Cytoscape 表现

- 节点按实体类型着色。
- 边显示方向与关系类型。
- 直接检索节点和边使用高亮描边与粗线。
- 补充邻居降低透明度。
- 支持缩放、拖拽、居中、节点搜索和类型筛选。
- 点击节点显示名称、类型、描述和来源。
- 支持“只看命中”和“显示一跳邻居”。

页面名称使用“当前查询相关图谱”，不将其描述成 LightRAG 的严格检索路线动画。

## 8. FastAPI 后端

### 8.1 组件

`GraphRAGBenchLoader`：读取 Novel/Medical JSON 或 Parquet，并列出可选文档。

`DatasetService`：选择一个 CorpusDocument、计算 fingerprint 并管理 active dataset。

`LightRAGService`：创建实例、建立索引、检查状态和复用 workspace。

`LightRAGAdapter`：只调用 `aquery_llm()`，检查顶层状态，并将 `llm_response` 与 `data` 转换为 QueryResult。

`QueryService`：完成校验、单次查询、计时、错误隔离和运行记录。

`GraphService`：读取 GraphML 并构造当前查询相关局部图。

### 8.2 API

- `POST /api/dataset/load-options`：读取 GraphRAG-Bench corpus 并返回可选文本摘要。
- `POST /api/dataset/select`：选择一个 active document。
- `GET /api/dataset/status`：返回 active dataset 摘要。
- `POST /api/index/build`：为 active dataset 建立 workspace。
- `GET /api/index/status`：返回索引状态与是否可复用。
- `POST /api/query`：执行一个问题和一个 LightRAG mode。
- `GET /api/query/{run_id}`：读取一次查询记录。
- `GET /api/health`：返回后端、配置和依赖状态，不暴露密钥。

`POST /api/query` 是唯一查询入口。它在一个响应中返回答案、上下文、检索详情和局部图，不要求前端再拼接多次查询结果。

## 9. React 前端

V1 使用一个主页面，减少导航和状态同步复杂度。

### 9.1 页面顶部

- 应用名称。
- 当前 corpus_name、文本规模和来源类型。
- LightRAG workspace 状态。
- “选择文本”和“建立/重新建立索引”按钮。

### 9.2 查询控制区

- 问题文本框。
- `local / global / hybrid / mix` 单选或下拉框。
- Top-K；同时传给 LightRAG 的 `top_k` 与 `chunk_top_k`。
- 提交按钮。
- 当前模式的简短说明。

索引未 ready 时禁用提交按钮，并显示具体原因。

### 9.3 结果区

- answer 卡片。
- latency 和 mode。
- context 原文区域。
- Entities、Relationships、Chunks、References 四个标签页。
- warnings 和错误提示。
- Cytoscape 当前查询相关图谱。

所有空数据都使用明确空态，例如“当前 LightRAG 接口未返回 reference”，避免用户误以为页面加载失败。

### 9.4 错误提示

前端根据结构化错误码展示：

- `DATASET_NOT_SELECTED`
- `INDEX_NOT_READY`
- `INDEX_CONFIG_MISMATCH`
- `LLM_AUTH_ERROR`
- `LLM_RATE_LIMITED`
- `LLM_TIMEOUT`
- `EMBEDDING_ERROR`
- `LIGHTRAG_ERROR`
- `INVALID_QUERY`

错误卡包含简短原因和下一步操作，例如检查环境变量、等待限流恢复或重新建索引。前端不显示完整 traceback 或密钥相关内容。

## 10. 状态、日志与安全

```text
data/raw/graphrag_bench/   GraphRAG-Bench corpus，不提交 Git
data/active/               当前文本与 manifest
data/workspace/            当前 LightRAG workspace
data/workspace_backups/    显式重建时的旧 workspace
app.db                     索引状态和 query runs
```

`query_runs` 至少记录：

- run_id
- query
- mode
- latency_ms
- success
- warning count
- error code 和安全摘要
- created_at

完整 answer/context 是否写入 SQLite 由配置控制，默认不保存，避免数据库快速膨胀。开发日志不得包含 API key。

## 11. 测试策略

### 11.1 单元测试

- Novel/Medical JSON 和 Parquet 能转换为 CorpusDocument。
- 最短非空文本选择正确。
- corpus fingerprint 稳定。
- workspace 复用检查能发现文本或 embedding 配置变化。
- Adapter 使用 `aquery_llm()`，显式设置 `stream=False`，并令 `chunk_top_k=top_k`。
- Adapter 抛出的 exception 和 `status != "success"` 的结构化失败都能转换为 QueryError。
- LightRAG 结果缺少任意结构化字段时，QueryResult 返回空值和 warning。
- 错误能转换成稳定错误码，且不泄露密钥。

### 11.2 集成测试

使用 GraphRAG-Bench 中一份足够小的文本：

1. 第一次构建产生 ready workspace 和 GraphML。
2. 第二次启动识别并复用同一 workspace，不再次插入文本。
3. 四种 mode 各完成一次查询。
4. 查询返回 answer、latency 和至少一种可用上下文信息。
5. 可映射实体或关系能生成 Cytoscape 元素。
6. 查询链路没有调用 `rag.query()` 或解析控制台日志。
7. 模拟直接抛出异常、failure dict、DeepSeek 超时、embedding 失败和索引不匹配时返回明确错误。

### 11.3 前端测试

- 未选择文本和索引未就绪时不能查询。
- mode 选择能正确进入请求。
- answer、context 和四类详情能正确展示正常、空值和错误状态。
- 图谱高亮、邻居淡化、搜索和筛选正确。
- 后端错误转换为可操作提示。

## 12. 项目结构

```text
lightrag-qa-app/
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ api/
│  │  ├─ domain/              # CorpusDocument、QueryResult、错误模型
│  │  ├─ adapters/
│  │  │  ├─ graphrag_bench.py
│  │  │  └─ lightrag.py
│  │  ├─ services/
│  │  │  ├─ dataset_service.py
│  │  │  ├─ lightrag_service.py
│  │  │  ├─ query_service.py
│  │  │  └─ graph_service.py
│  │  ├─ persistence/
│  │  └─ config.py
│  └─ tests/
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  ├─ api/
│  │  └─ types/
│  └─ tests/
├─ data/
├─ docs/
├─ .env.example
└─ README.md
```

## 13. 实施顺序

### Phase 1：数据与索引

1. 固定 LightRAG 版本、DeepSeek 配置和 embedding 配置。
2. 实现 GraphRAG-Bench corpus loader 与单文档选择。
3. 实现 active dataset manifest 和 fingerprint。
4. 跑通单份小文本的 LightRAG `ainsert()`。
5. 实现 workspace 状态检查与复用。

### Phase 2：单次查询后端

6. 建立 QueryResult 和错误模型。
7. 实现 LightRAGAdapter。
8. 实现四种 mode 的单次查询。
9. 实现 latency/error 记录。
10. 实现 GraphML 局部图构造。
11. 暴露 FastAPI 接口。

### Phase 3：单页前端

12. 实现文本选择和索引状态区。
13. 实现问题输入与 mode 选择。
14. 实现 answer/context/详情标签页。
15. 实现 Cytoscape 局部图与命中高亮。
16. 实现 loading、空态和错误提示。

### Phase 4：验证与课堂演示

17. 对四种 mode 各完成一次真实查询。
18. 验证重启后 workspace 复用。
19. 准备 3–5 个手动问题展示完整流程。
20. 编写启动、配置和演示说明。

## 14. V1 验收标准

1. 能选择一份小说或医学文本并建立 LightRAG workspace。
2. 能正常执行 `local`、`global`、`hybrid`、`mix` 四种 LightRAG mode。
3. 前端可以由用户手动选择 mode。
4. 能展示 answer、context、entities、relationships、chunks、references 中当前 LightRAG 实际可获得的信息。
5. 能展示当前查询相关的有限局部图，并区分直接命中和补充邻居。
6. 能稳定复用已建立且配置匹配的 workspace，不重复建图。
7. DeepSeek、embedding、LightRAG 或索引状态报错时，前端有明确且不泄密的提示。
