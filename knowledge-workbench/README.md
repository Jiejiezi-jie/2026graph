# 知图 · LightRAG 检索工作台

本仓库包含 FastAPI + React/Cytoscape 知识检索工作台。
检索方法采用插件接口；目前内置 LightRAG 的 local / global / hybrid / mix / vector（底层 `naive`），
尚未实现新的检索算法或多方法评测界面。

## 队友交接

代码位于 `backend/`、`frontend/`、`scripts/`。分块与图谱在
[exports/Novel-4128-chunks](exports/Novel-4128-chunks/README.md)，包括 51 个原文分块和完整 GraphML（729 节点、1,369 条关系）。
数据文件已纳入 Git；无需获取 `.worktrees` 文件夹。
完整可复用索引现已包含在 [exports/Novel-4128-runtime](exports/Novel-4128-runtime/README.md)：原文、分块、GraphML、三个向量库、文档状态及完整索引清单。
按包内说明安装固定版本依赖，然后在本目录运行 `python exports/Novel-4128-runtime/restore.py --target .`，即可恢复已建索引，无需重新建图。
API Key、个人 `.env`、embedding 模型权重、查询历史和依赖目录不随仓库交接；队友使用自己的 Key 并安装匹配模型。运行目录仍被 Git 忽略，Git 跟踪的是 exports 下的完整快照。

## 启动网页

以下命令在本仓库根目录运行。克隆后，将示例路径换成你自己的仓库位置；Python 路径也需换成符合下方 Runtime contract 的 Python 3.10 环境：

```powershell
cd 'D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱'
```

终端 1（后端，保留此窗口）：

```powershell
# Key 只在启动后端的环境中设置；也可使用本 worktree 下不被 Git 追踪的 .env。
$env:DEEPSEEK_API_KEY = '<your-current-valid-key>'
& 'D:\Anaconda\envs\lightrag\python.exe' scripts\serve_web.py
```

终端 2（前端）：

```powershell
cd 'D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱\frontend'
npm ci
npm run dev
```

打开 [检索工作台](http://127.0.0.1:5173/)；
[API 文档](http://127.0.0.1:8000/docs)。
首次安装后端依赖见下方 Runtime contract；不要另装 PyPI LightRAG 来替换本地 checkout。
若端口已占用，先停止此前启动的同一服务，不要运行两个 backend worker。

- 打开网页读取已有 workspace，不重新建图；现有 Novel-4128 可直接复用。
- “管理数据与索引”可以选择 Novel/Medical 中最短的一份文本。
- 建索引必须手动确认，会调用 DeepSeek；重建会先备份旧 workspace。
- 图谱可缩放、定位实体、查看节点/关系描述；命中高亮与补充邻居严格区分。
- 图谱右上角「展开图谱」隐藏回答栏并占满主内容宽度，自动适配画布；「收起图谱」恢复双栏和原浏览视角，保留选中实体及漂动暂停状态。
- 点击节点可平滑聚焦并突出相邻关系；展开后的详情浮在画布右侧，窄屏浮在底部。标签按缩放、重要性和可用空间显示，悬停可看完整名称。
- 查询记录保存答案、检索结果、耗时和安全错误码。
- 查看记录后点击「检索工作台」，会清空当前问题、答案和证据，并重新加载整体图谱预览；不会重新提交查询。
- `vector（纯向量）` 按语义相似度检索原文片段，不扩展图谱邻居；复用已有文本向量索引，无需重新建图。
- 侧栏「API 设置」可填写 API Key 和 Base URL，默认地址为 `https://api.deepseek.com`。
- 网页设置保存后立即用于后续查询、答案生成和建索引；只保存在后端内存中，不写入 `.env`、数据库或浏览器存储。重启后恢复启动时的环境配置，网页输入的 Key 需要重新填写。
- 已配置 Key 时留空保留当前 Key；输入新 Key 可替换。保存后不会回显密钥。查询或建索引期间不能修改连接设置。
- 单独更换 Key 不影响索引复用。URL 参与现有索引配置指纹，更换 URL 后可能提示索引配置不匹配；不会自动重建索引。
- 保存设置不调用模型、不验证鉴权。health 的已配置状态也不表示接口鉴权已通过。

新增 retrieval 的接口、示例、错误处理和运行限制见
[检索插件开发约定](docs/RETRIEVAL_PLUGINS.md)。
重点入口：`backend/app/retrieval/contracts.py` 与 `plugins.py`。

## Web 与原有 CLI 的区别

```text
网页：检索插件（LightRAG aquery_data）→ 统一答案生成 → 证据和图谱展示
原 CLI：LightRAG aquery_llm → LightRAG 原生答案和结构化数据
```

网页使用自己的证据约束提示词，不能把网页答案视作旧 benchmark 的同配置结果。
本轮真实图谱读取通过；真实问答联调得到当前 Key 的 401，未验证有效 Key 下的
端到端成功回答。接口与成功界面使用明确标记的离线桩数据验证，不作为模型效果展示。

## 验证

```powershell
& 'D:\Anaconda\envs\lightrag\python.exe' -m pytest backend\tests -q
cd frontend
npm run build
```

浏览器离线检查（在 worktree 根目录、前后端启动后运行）：

```powershell
npx --yes --package @playwright/cli playwright-cli -s=knowledge-web open http://127.0.0.1:5173
npx --yes --package @playwright/cli playwright-cli -s=knowledge-web run-code --filename frontend/qa/browser-smoke.js
npx --yes --package @playwright/cli playwright-cli -s=knowledge-web run-code --filename frontend/qa/workbench-navigation.js
npx --yes --package @playwright/cli playwright-cli -s=knowledge-web run-code --filename frontend/qa/graph-expanded.js
```

此脚本拦截查询请求、不调用付费 API、不向真实查询历史写入测试结果，
结束后恢复真实页面。检查四种模式控件、证据展示、错误保留证据、图谱定位及窄屏布局。
`workbench-navigation.js` 另外验证历史记录返回工作台时刷新图谱，以及 vector 标签与底层 naive 参数的对应关系。

## 原有 Phase 1 命令行

Phase 1 is a Python smoke application for validating one complete LightRAG path:

```text
GraphRAG-Bench shortest Novel document
  -> LightRAG indexing
  -> reusable workspace
  -> aquery_llm(local/global/hybrid/mix)
  -> answer plus structured retrieval data
```

The CLI remains available alongside the web application. Existing benchmark scripts are independent of the web query path.

## Runtime contract

- Conda environment: `lightrag` (Python 3.10)
- LightRAG source: local editable checkout at commit `c1248646e4eda4d89054926af2e094730daf23fe`
- LLM: DeepSeek OpenAI-compatible API
- Embedding: local `BAAI/bge-small-en-v1.5`, 384 dimensions
- Graph storage: `NetworkXStorage`
- Corpus source: the sibling GraphRAG-Benchmark checkout

The backend intentionally does not declare `lightrag-hku` as a PyPI dependency. Bind the environment to the reviewed local checkout instead:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pip install -e 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\LightRAG'
& 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pip install -e '.\backend[test]'
```

Verify the source before running:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -c "import lightrag; print(lightrag.__file__); print(lightrag.__version__)"
git -C 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\LightRAG' rev-parse HEAD
```

The import path must point into the local checkout, and Git must print the pinned commit above.

## Configuration

Copy `.env.example` to an untracked `.env`, or set the key only in the PowerShell process that launches the command:

```powershell
$env:DEEPSEEK_API_KEY = '<your-current-key>'
```

Never commit or paste the key into source files. `LLM_API_KEY` is accepted as a compatibility alias. Other non-secret defaults are listed in `.env.example`.

## Run one real query

From the repository root:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run --no-capture-output -n lightrag python scripts\smoke_lightrag.py `
  --corpus 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\GraphRAG-Benchmark\Datasets\Corpus\novel.json' `
  --subset novel `
  --mode mix `
  --top-k 5 `
  --query 'Who is the central character, and which relationships are most important to the story?'
```

The loader selects the shortest non-empty document deterministically. In the current local dataset this is `Novel-4128` (92,796 characters).

The first successful run downloads the embedding model if necessary and pays the one-time DeepSeek indexing cost. A successful second identical run reports `"index_action": "reused"` and does not call `ainsert()` again.

Generated state is stored under:

```text
data/active/       selected document and manifest
data/workspace/    LightRAG stores, GraphML, and workspace manifest
```

Both directories are ignored by Git. A workspace is reusable only when corpus and runtime fingerprints match, its manifest is `ready`, GraphML exists, document statuses are processed, and the chunk and vector stores pass minimal persisted-file checks.

## Result semantics

The CLI prints one JSON result containing the answer, latency, mode, index action, warnings, and the `entities`, `relationships`, `chunks`, and `references` returned by `aquery_llm()`.

`context_text` is a stable display serialization of those structured fields. It is not claimed to be byte-for-byte identical to LightRAG's internal final prompt.

`Top-K` is intentionally a single V1 control: the adapter passes the same value to both `QueryParam.top_k` and `QueryParam.chunk_top_k`, with `stream=False`.

## Failure behavior

Expected failures are converted to short codes such as:

- `MISSING_API_KEY`
- `LLM_AUTH_FAILED`
- `LLM_RATE_LIMITED`
- `LLM_TIMEOUT`
- `EMBEDDING_ERROR`
- `INDEX_CONFIG_MISMATCH`
- `INDEX_INCOMPLETE`
- `LIGHTRAG_ERROR`
- `INVALID_QUERY`

The application never stores provider exception messages in its manifest. If indexing or finalization fails, the workspace manifest is `failed`; it is never marked `ready` before storages are finalized and GraphML passes the integrity check.

## Offline verification

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend\tests -v
```
