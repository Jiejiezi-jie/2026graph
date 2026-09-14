# 最新工作台更新（2026-09-14）

Adaptive 使用 300 题实验的加权 BGE-M3 CLS + Logistic Regression 选择头，
开发集 276 题，阈值 0.60、容差 0.10。模型来自 main 提交 `5f374e2`。
图谱、Chunk 与向量索引不需要重新生成。

已有 Medical bundle：停止后端，用 Medical Python 环境在仓库根目录执行：

```powershell
python src/backend/scripts/install_medical_router.py --repo . --ref 5f374e2 --bundle src/backend/data/medical
```

新机器：`prepare_medical.py --repo . --ref origin/main` 默认安装新选择头。
导入旧提交需添加 `--legacy-router`，再运行上述安装命令。
2026-09 仓库重组后路径已整体变化（`results_api/`→`result/api/`、
`src/official_backends/`→`src/backend/{common,vector,lightRAG,pathRAG}/`），
两个导入脚本都保留了旧前缀回退，因此 `5f374e2`、`2ce4e23` 等重组前的提交仍可直接导入。
安装器保留旧模型和 `bundle-before-router-*.json`，回退时停止后端并恢复对应旧清单。
请使用 scikit-learn 1.7.2。BGE-M3 权重仍按下文下载，不将 API Key 或本机缓存提交。

关键词提取默认预算提高到 4096，可用环境变量 `MEDICAL_KEYWORD_MAX_TOKENS` 调整。
LightRAG 的 JSON 输出格式正确转发；两种图检索均检查截断、空响应和格式错误。
日志仅记录结束原因、token 数及响应长度，不记录密钥或思考正文。

图谱命中展示支持 PathRAG 路径的已验证节点和边；旧记录不能匹配当前图谱时显示提示。
重启后端并刷新网页，对新问题生效。下方保留原有环境与启动说明。

---

# Medical 知识库接入与启动

## 队友拉取后的数据准备

在仓库根目录执行 `git pull --ff-only origin main`（2026-09 整合后 workbench 与 Medical 评测内容全部位于 main）。
下文的新机器依赖、模型安装步骤均从此目录（仓库根）执行。
Medical 索引和路由模型已在同一仓库的队友提交中，使用固定提交导入，不需要重新建图：

```powershell
git fetch origin lightRAG
python src/backend/scripts/prepare_medical.py --repo . --ref 2ce4e23b98ac4405e3e1c80454e5daef8e5f3b95 --output src/backend/data/medical
```

已有导入目录时跳过导入。PathRAG 的共享索引兼容副本会在首次查询时自动创建。
模型权重、本地 Python 环境、API Key 和查询历史不随代码提交。
2026-09 仓库整合后 Novel-4128 数据与产物已从仓库移除，Medical 是工作台唯一数据源；
legacy 本地建图运行时代码保留但非默认路径。

网页查询使用 `POST /api/query/stream`：分类完成推送 routing，
检索完成推送 retrieval，最终回答推送 result。原 `POST /api/query` 仍可使用。

这份接入基于队友仓库的 `origin/lightRAG`，当前导入提交为
`2ce4e23b98ac4405e3e1c80454e5daef8e5f3b95`。
现有知图页面、回答生成和查询记录接口继续复用；legacy 本地建图运行时代码仍保留，仅 Medical 导入模式默认启用。

## 当前电脑：直接启动（2026-09-09 更新）

当前 LightRAG 与 PathRAG **共用 LightRAG 构建的 Medical 图谱**：
3,396 个节点、5,693 条关系、199 个 Chunk。PathRAG 使用同源实体、关系、
Chunk 向量和原文映射，首次运行在 `runtime/pathrag-shared/<内容哈希>/`
建立字节一致的兼容副本，隔离运行缓存。无需重新抽取或生成 embedding。
旧 `indexes/pathrag` 仅保留为导入归档，不再参与新查询或图谱展示。
更新后需重启后端、刷新网页，并重新填写会话 API Key。
旧查询记录仍是当时的结果，验收共享图谱请提交新问题。

BGE-M3 权重已经下载，大小为 2,271,145,830 字节，
SHA-256 为 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`，
与队友记录一致。模型目录为 `artifacts/models/bge-m3`。

本机已创建 `artifacts/medical-env` 专用虚拟环境，复用原 lightrag 环境的
PyTorch 等基础依赖，并在专用环境内安装了固定版本的 LightRAG、scikit-learn
和 Transformers；原环境没有卸载或升级。固定版本 PathRAG 位于
`artifacts/medical-upstream/PathRAG`。当前电脑无需再运行下方的新机器安装步骤。

已经验证：启动预检通过；BGE-M3 实际编码为 1024 维、L2 范数 1.0；
Vector 实际召回 3 个片段；路由模型成功加载；LightRAG 和 PathRAG
均载入原有图谱及实体/关系/Chunk 向量索引。上述检查没有调用 DeepSeek API。

**窗口一：后端**

```powershell
Set-Location 'D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱'
$env:LLM_MODEL = 'deepseek-chat'
& '.\artifacts\medical-env\Scripts\python.exe' src/backend/scripts/serve_web.py --port 18000 --medical-bundle src/backend/data/medical --medical-embedding-path artifacts/models/bge-m3 --medical-pathrag-root artifacts/medical-upstream/PathRAG
```

**窗口二：前端**

```powershell
Set-Location 'D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱\src\frontend'
$env:API_PROXY_TARGET = 'http://127.0.0.1:18000'
npm run dev
```

打开 `http://127.0.0.1:5173`，在侧栏“API 设置”填写 DeepSeek Key 并保存。
默认 URL 为 `https://api.deepseek.com`。两个终端保持开启，停止时各按 Ctrl+C。
当前电脑的 8000、8001 和 8080 端口绑定被 Windows 拒绝；已检查 18000 和 5173
可用，因此本机命令使用后端 18000、前端 5173。前端代理可由
`API_PROXY_TARGET` 环境变量指定，未设置时仍默认 8000。
已实际启动这两个服务，并通过前端代理验证了 health、四种检索方法列表和
PathRAG 图谱接口；`retrieval_ready=true`，`runtime_issues=[]`。验证后测试服务已停止。

## 已完成的接入

### PathRAG 关键词格式修复（2026-09-08）

上游关键词示例插入提示词后仍含双花括号，模型可能输出
`{{"high_level_keywords": [...], "low_level_keywords": [...]}}`。
上游 JSON 解析失败后返回固定失败文本，原接入把它误报为没有检索证据。
现接入层仅修正已知示例的花括号、对 PathRAG 关键词请求启用 JSON object
响应格式，并兼容已观察到的多包一层花括号。关键词缺失、格式错误或截断
会返回独立错误码，固定失败文本也不再记为查询成功。未修改导入的上游源码。

同时处理上游 Sources 合并输出的 `id,\tcontent` 格式，保留原文中的逗号和
换行，避免把同一个 Chunk 拆成多条记录。常规带引号 CSV 仍由原解析器处理。

对胆管肿瘤解剖层次问题，已用用户提供的原始关键词输出进行离线重放：
Top-K=5，实际返回 2 个可精确匹配 Chunk ID 的原文片段；没有调用模型 API，
也没有写回索引或检索缓存。诊断结果保存于
`artifacts/pathrag-diagnostics/bile-duct-replay.json`。
更新后需重启后端，并在网页重新保存本次会话的 API Key，然后重新提交问题。

| 网页方法 | 执行内容 | 图谱来源 |
| --- | --- | --- |
| Vector | 从队友独立的 BGE-M3 文本向量索引检索 Chunk | 不展示图谱 |
| LightRAG | 调用其 LightRAG 的 `aquery_data`，支持 local/global/hybrid/mix/naive | LightRAG 的 Medical 图谱 |
| PathRAG | 在共享索引上调用 `aquery(... only_need_context=True)` | LightRAG 的 Medical 共享图谱 |
| Adaptive | 加载训练好的 TF-IDF + Logistic Regression 分类器，只执行预测的一个后端 | 跟随实际选中的方法 |

结果统一进入本项目的答案生成器，再保存到查询记录。没有调用队友的整套
`query()` 后再重复生成答案，也没有把离线评估结果当作实时检索返回。
Adaptive 结果会显示实际方法、分类概率和生效参数。分类概率不是答案正确率。
该分类器训练于英文 Medical 问题，不应把中文路由效果视为已经验证。

原导入包包含两套独立构建的图；当前仅启用 LightRAG 那套。
共享方案沿用现有路由分类器，未重新训练；原两图实验的评估结果不能直接视作共享图效果。
索引清单声明哈希与存储原文哈希分别报告，不声称重新验证了完整构建过程。

## 文件与运行隔离

- 代码：`src/backend/app/medical/`
- 导入工具：`src/backend/scripts/prepare_medical.py`
- Medical 数据、源码快照、路由模型及查询历史：`src/backend/data/medical/`
- 原 Novel 数据与历史：2026-09 整合时已从 Git 移除（legacy 建图运行时代码保留）。
- 导入目录已被 Git 忽略。本次没有提交、推送或切换任何分支。

导入工具只读取指定 Git 提交中的索引、适配器源码、路由模型与许可文件，
记录来源提交和文件校验和；不复制 API 配置，不读取未提交的工作区内容，不覆盖已有目录。
查询期间上游可能更新检索缓存，因此这个目录是本机运行副本，不是不可变的原始归档。
源码快照及路由模型在载入前会核对导入时的 SHA-256。

## 当前电脑已经导入的数据

目录：

```text
D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱\src\backend\data\medical
```

Vector、LightRAG、PathRAG 各有 199 个 Chunk；查询编码必须使用
**BGE-M3、1024 维、CLS pooling、L2 normalize、max_length=2048**。
不能沿用 Novel 的 384 维 BGE-small，也不能直接更换 pooling 后继续查询旧索引。

当前两种图检索统一为 3,396 个节点 / 5,693 条关系。网页加载共享图谱全部节点与关系，默认以
局部阅读视角打开；「全局」适应完整图谱，「局部」恢复阅读比例。
没有连线的孤立节点均匀排列在最外围圆环，有连线的小群组仍保持内部结构。
圆环只表示显示位置，不添加关系。节点间距处理与社区布局仅计算一次并缓存，
不会修改 GraphML 或向量索引；布局版本变更后自动更新缓存。
查询后在同一完整图谱上更新实际命中高亮，Adaptive 跟随实际检索方法。
查询记录中的图谱快照仍限制为 80 节点 / 200 边，同时另存完整命中标识；
不会向每条历史记录重复写入几千节点的完整图谱。旧记录使用其已有命中标记，
无法凭空恢复旧记录中未保存的命中。
前端只让可视区域内最多 40 个节点轻微漂动，限制受影响的连线总数为 400；
鼠标悬停不暂停，使用暂停按钮控制。展开、收起和取消节点聚焦保留阅读视角。
更换此版本后重启后端、刷新网页即可；首次加载会重新生成显示布局缓存。
PathRAG 返回的 CSV 名称与 GraphML 中带引号的 ID 已做精确格式映射；
原始返回证据仍保留，不根据名字相似度猜测关系。

首次接入代码准备阶段没有下载模型。后续已按用户要求完成上述本机模型下载与环境安装，
没有重新建图、没有调用问答 API。以下安装/下载步骤保留给换电脑或重装环境时使用。

## 早期双图版本的验证记录（当前共图版本见文首）

已执行前端 TypeScript 检查和生产构建、后端自动测试，以及真实 Medical 数据的
FastAPI 内存客户端检查。四种方法均可列出；两份图谱预览返回 200；
Vector 与尚未执行查询的 Adaptive 返回带说明的空图。
还用队友已保存的 PathRAG 检索证据检查了实体/关系 ID 映射。

后续已在独立端口 18001 / 5180 完成浏览器视觉验收：两张图全量载入、
默认局部视角、全局圆环、节点再次点击取消高亮并保留视角。
PathRAG 的 961 个无连线节点位于外围圆环。截图位于
`output/playwright/fullgraph-ring-pathrag.png`。
独立环境已安装，BGE-M3 编码、Vector 检索、路由模型载入和两份上游索引初始化已实测通过；
LightRAG/PathRAG 的实时关键词提取和最终回答尚未调用 API 验证。

## 1. 新建 Medical 环境

在 PowerShell 中执行：

```powershell
Set-Location 'D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱'
conda create -n medical-workbench python=3.10 -y
conda run -n medical-workbench python -m pip install -r requirements-medical.txt
```

不要在现有 Novel 的环境里升级依赖。Medical 采用队友固定的 LightRAG 提交；
路由模型实际序列化于 `scikit-learn 1.7.2`，依赖文件已固定该版本。
本项目原 backend 的 Python 版本约束也保持不变。

PathRAG 使用下面的独立源码目录。首次安装时运行：

```powershell
git clone https://github.com/BUPT-GAMMA/PathRAG.git artifacts/medical-upstream/PathRAG
git -C artifacts/medical-upstream/PathRAG checkout --detach 32567bfc93605b8393996d5fa9ccdc0edbb865b2
```

若目录已存在，先检查它的提交和本地修改，不要覆盖或重置已有工作。

2026-09 重组后仓库已自带同一提交的精简上游（仅 `PathRAG/` 包 + 依赖清单，
见 `deps/PathRAG`），可直接用 `--medical-pathrag-root deps/PathRAG`，
不必再克隆一份。该目录不带 `.git`，因此固定提交由其中的 `UPSTREAM_COMMIT`
标记声明，preflight 只在目录是独立 Git 检出时才读 `git rev-parse HEAD`；
两条路径任一满足即可通过版本校验。
若目录既不是独立检出、也没有该标记（例如被手工删过文件），
preflight 会明确报告「无法确认 PathRAG 固定 Git 提交」而不会静默放过。

## 2. 下载 BGE-M3

可以从队友取得其完整模型目录，或者稍后下载官方模型：

```powershell
conda run -n medical-workbench python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='artifacts/models/bge-m3', allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model'])"
```

这一步会下载模型权重，不是重新构建语料索引。CPU 可运行，首次载入及推理可能较慢；
若安装了匹配的 CUDA/PyTorch，可在后面的启动命令加 `--medical-device cuda:0`。

## 3. 检查后启动后端

```powershell
conda run -n medical-workbench python src/backend/scripts/serve_web.py --medical-bundle src/backend/data/medical --medical-embedding-path artifacts/models/bge-m3 --medical-pathrag-root artifacts/medical-upstream/PathRAG --check
```

这个预检读取索引、模型配置和依赖版本，不加载模型权重、不调用 API。
索引校验通过但依赖不齐时会列出缺项并返回非零退出状态。
文件/版本预检通过不等于真实检索和模型回答已经测试通过。

环境准备好后：

```powershell
$env:LLM_MODEL = 'deepseek-chat'
conda run --no-capture-output -n medical-workbench python src/backend/scripts/serve_web.py --medical-bundle src/backend/data/medical --medical-embedding-path artifacts/models/bge-m3 --medical-pathrag-root artifacts/medical-upstream/PathRAG
```

后端地址为 `http://127.0.0.1:8000`。先停止占用此端口的旧后端。
API Key / URL 仍通过网页“API 设置”填写，Key 只在当前后端进程内保存。
检索关键词提取及最终回答会使用这里配置的 API。

如暂时只想检查图谱和下拉框，可用已有 Python 环境：

```powershell
& 'D:\Anaconda\envs\lightrag\python.exe' src/backend/scripts/serve_web.py --medical-bundle src/backend/data/medical --preview-only
```

缺少模型时页面会显示未就绪原因并禁用查询；预览不会假装完成真实检索。

## 4. 启动前端

另开一个 PowerShell：

```powershell
Set-Location 'D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱\src\frontend'
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。侧栏应显示 Medical，检索方法下拉框应有
LightRAG / Vector / PathRAG / Adaptive。切换方法会清除上一条结果并加载对应预览。
Adaptive 在执行前不展示某个固定方法的图，以免误导；查询后显示实际方法的图。
Medical 模式禁止通过原有“选择最短文本/重新建图”入口覆盖导入索引。

## 参数含义与复现边界

- Vector：Top-K 是独立向量索引中的 Chunk 候选数，默认 5。
- LightRAG：Top-K 传入 `chunk_top_k`；新增“图谱候选”传入图检索的
  `top_k`，默认 40；默认模式为队友采用的 hybrid。
- PathRAG：Top-K 传入其检索 `top_k`；路径检索模式为 hybrid，
  保留队友文本/局部/全局上下文预算 2000/1500/1500。若要对齐队友实验，手动设为 40。
- Adaptive：保留用户输入的 Top-K 并传给选中方法；若选中 LightRAG，
  固定 hybrid + 图谱候选 40。这不是对离线实验各方法独立 Top-K 的自动重放。

队友离线实验默认 Vector=5、LightRAG 图谱=40/Chunk=5、PathRAG=40。
当前统一工作台复用了本项目的生成提示词，因此不能把网页回答直接视为
队友原 benchmark 的完整复现，更不能据此宣称 Adaptive 一定优于固定方法。
如果要做严格对比，应固定问题、各方法参数、生成提示词、模型和评估配置。

## 换电脑/重新导入

先拉取队友仓库，再运行：

```powershell
python src/backend/scripts/prepare_medical.py --repo 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\2026graph' --ref origin/lightRAG --output src/backend/data/medical
```

为了复现本次快照，可将 `--ref` 改成上面记录的完整提交哈希。
如果输出目录已存在，导入会拒绝覆盖；更新快照时换一个输出目录，再用对应启动参数。

## 接口示例

```json
{"query":"Your English Medical question","method_id":"adaptive","top_k":5,"options":{}}
```

发送至 `POST /api/query`。返回数据仍包含 `answer`、`retrieval.chunks`、
`entities`、`relationships`、`graph` 和计时；
Adaptive 另外在 `retrieval.metadata` 返回 `selected_method`、
`routing_probabilities`、`effective_top_k`、`effective_options`。
图谱预览可请求 `GET /api/graph?method_id=pathrag` 或 `method_id=lightrag`。
